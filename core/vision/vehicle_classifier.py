"""
Vehicle Type Classifier
-----------------------
Primary  : EfficientNet-B0 fine-tuned on the Kaggle Vehicle Classification Dataset
Fallback : Claude Vision API (zero-shot) when local model is absent
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)

# ── ImageNet-normalised transform used at inference ────────────────────────────
INFER_TRANSFORM = T.Compose([
    T.Resize((settings.image_size, settings.image_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Model definition ───────────────────────────────────────────────────────────

class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 backbone with a custom linear head."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ── High-level interface ───────────────────────────────────────────────────────

class VehicleClassifier:
    """
    Classifies vehicle type from a PIL image.

    Usage::

        clf = VehicleClassifier()
        result = clf.predict(pil_image)
        # {"vehicle_type": "SUV", "confidence": 0.92, "all_scores": {...}, "source": "local_model"}
    """

    def __init__(self) -> None:
        self.classes = settings.vehicle_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[nn.Module] = None
        self._try_load_local_model()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def _try_load_local_model(self) -> None:
        path: Path = settings.vehicle_classifier_path
        if not path.exists():
            logger.info("No local vehicle classifier at %s — will use Claude Vision.", path)
            return
        try:
            mdl = EfficientNetClassifier(len(self.classes))
            mdl.load_state_dict(torch.load(path, map_location=self.device))
            mdl.to(self.device).eval()
            self._model = mdl
            logger.info("Local vehicle classifier loaded (%s classes).", len(self.classes))
        except Exception as exc:
            logger.warning("Failed to load local model: %s — falling back to Claude Vision.", exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, image: Image.Image) -> dict:
        """Return classification result dict."""
        if self._model is not None:
            return self._predict_local(image)
        return self._predict_claude(image)

    # ── Local inference ────────────────────────────────────────────────────────

    def _predict_local(self, image: Image.Image) -> dict:
        tensor = INFER_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

        scores = dict(zip(self.classes, probs))
        best = max(scores, key=scores.get)
        return {
            "vehicle_type": best,
            "confidence": round(scores[best], 4),
            "all_scores": {k: round(v, 4) for k, v in scores.items()},
            "source": "local_model",
        }

    # ── Claude Vision fallback ─────────────────────────────────────────────────

    def _predict_claude(self, image: Image.Image) -> dict:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        b64 = _pil_to_b64(image)
        class_list = ", ".join(self.classes)

        prompt = (
            f"You are a vehicle recognition expert. Classify the vehicle in this image "
            f"into EXACTLY ONE of these categories: {class_list}.\n\n"
            "Respond with valid JSON only:\n"
            '{"vehicle_type": "<class>", "confidence": <0-1 float>, "reasoning": "<brief>"}'
        )

        resp = client.messages.create(
            model=settings.claude_model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )

        data = _parse_json(resp.content[0].text)
        raw_type = data.get("vehicle_type", "Car")
        # Normalise to known class (case-insensitive match)
        matched = next((c for c in self.classes if c.lower() == raw_type.lower()), raw_type)
        conf = float(data.get("confidence", 0.80))

        return {
            "vehicle_type": matched,
            "confidence": conf,
            "all_scores": {matched: conf},
            "source": "claude_vision",
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)
