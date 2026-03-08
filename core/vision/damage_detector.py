"""
Vehicle Damage Detector
-----------------------
Primary  : EfficientNet-B0 fine-tuned on the eashankaushik car-damage dataset
           (stage-1: damaged / normal  →  stage-2: dent / scratch / shatter / dislocation)
Fallback : Claude Vision API (zero-shot) returns location-aware damage descriptions
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

from config import settings
from api.schemas import DamageItem, DamageResult

logger = logging.getLogger(__name__)

INFER_TRANSFORM = T.Compose([
    T.Resize((settings.image_size, settings.image_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# damage_classes index → severity bucket
SEVERITY_MAP = {
    "normal":      "none",
    "scratch":     "minor",
    "dent":        "moderate",
    "dislocation": "moderate",
    "shatter":     "severe",
}


class EfficientNetClassifier(nn.Module):
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


class DamageDetector:
    """
    Detects vehicle damage from a PIL image.

    Usage::

        detector = DamageDetector()
        result: DamageResult = detector.detect(pil_image)
    """

    def __init__(self) -> None:
        self.classes = settings.damage_classes          # ["dent","scratch","shatter","dislocation","normal"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[nn.Module] = None
        self._try_load_local_model()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def _try_load_local_model(self) -> None:
        path: Path = settings.damage_classifier_path
        if not path.exists():
            logger.info("No local damage classifier at %s — will use Claude Vision.", path)
            return
        try:
            mdl = EfficientNetClassifier(len(self.classes))
            mdl.load_state_dict(torch.load(path, map_location=self.device))
            mdl.to(self.device).eval()
            self._model = mdl
            logger.info("Local damage classifier loaded (%d classes).", len(self.classes))
        except Exception as exc:
            logger.warning("Failed to load damage model: %s — falling back to Claude Vision.", exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, image: Image.Image) -> DamageResult:
        if self._model is not None:
            return self._detect_local(image)
        return self._detect_claude(image)

    # ── Local inference ────────────────────────────────────────────────────────

    def _detect_local(self, image: Image.Image) -> DamageResult:
        tensor = INFER_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

        scores = dict(zip(self.classes, probs))
        best = max(scores, key=scores.get)
        conf = scores[best]
        threshold = settings.damage_confidence_threshold

        if best == "normal" or conf < threshold:
            return DamageResult(
                detected_damages=[],
                damage_items=[],
                overall_severity="none",
                source="local_model",
            )

        item = DamageItem(
            damage_type=best,
            location="vehicle body",      # local model can't localise — use Claude for that
            severity=SEVERITY_MAP.get(best, "minor"),
            confidence=round(conf, 4),
        )
        return DamageResult(
            detected_damages=[f"{item.location} {best}"],
            damage_items=[item],
            overall_severity=item.severity,
            source="local_model",
        )

    # ── Claude Vision fallback ─────────────────────────────────────────────────

    def _detect_claude(self, image: Image.Image) -> DamageResult:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        b64 = _pil_to_b64(image)

        prompt = (
            "You are an automotive damage assessment AI. Inspect this vehicle image carefully.\n\n"
            "Identify ALL visible damage. For each damage, provide:\n"
            "  - damage_type: one of [dent, scratch, shatter, dislocation, normal]\n"
            "  - location: specific body panel (e.g. 'rear bumper', 'driver-side door', 'hood')\n"
            "  - severity: one of [minor, moderate, severe]\n"
            "  - confidence: float 0–1\n\n"
            "If NO damage is visible, return an empty damages list.\n\n"
            "Respond with valid JSON only:\n"
            "{\n"
            '  "damages": [\n'
            '    {"damage_type": "dent", "location": "rear bumper", "severity": "moderate", "confidence": 0.9}\n'
            "  ],\n"
            '  "overall_severity": "moderate"\n'
            "}"
        )

        resp = client.messages.create(
            model=settings.claude_model,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )

        data = _parse_json(resp.content[0].text)
        raw_damages = data.get("damages", [])
        overall = data.get("overall_severity", "none")

        items: List[DamageItem] = []
        descriptions: List[str] = []
        for d in raw_damages:
            dtype = d.get("damage_type", "damage")
            loc   = d.get("location", "vehicle body")
            sev   = d.get("severity", "minor")
            conf  = float(d.get("confidence", 0.8))
            items.append(DamageItem(damage_type=dtype, location=loc, severity=sev, confidence=conf))
            descriptions.append(f"{loc} {dtype}")

        return DamageResult(
            detected_damages=descriptions,
            damage_items=items,
            overall_severity=overall if raw_damages else "none",
            source="claude_vision",
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    # Extract the first complete JSON object (handles trailing text from Claude)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)
