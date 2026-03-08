"""
Vehicle Analysis Routes
-----------------------
POST /api/v1/analyze          – full multi-modal analysis (image + text + metadata)
POST /api/v1/analyze/image    – image-only (vehicle type + damage, no intent)
POST /api/v1/analyze/text     – text-only (intent extraction, no image required)
GET  /api/v1/vehicle-types    – list supported vehicle categories
GET  /api/v1/damage-types     – list supported damage categories
"""
from __future__ import annotations

import json
import logging
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from api.schemas import (
    AnalyzeRequest,
    ClassifierResult,
    DamageResult,
    ErrorResponse,
    IntentResult,
    ServiceRecord,
    VehicleMetadataIn,
)
from config import settings
from core.pipeline import VehicleIntelligencePipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["vehicle"])

# Pipeline is initialised once on first use (lazy singleton)
_pipeline: Optional[VehicleIntelligencePipeline] = None


def _get_pipeline() -> VehicleIntelligencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VehicleIntelligencePipeline()
    return _pipeline


def _load_image(upload: UploadFile) -> Image.Image:
    """Read multipart image upload → PIL Image."""
    try:
        raw = upload.file.read()
        return Image.open(BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is not a valid image.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read image: {exc}",
        )


# ── POST /analyze  (main endpoint) ────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=ServiceRecord,
    summary="Full multi-modal vehicle analysis",
    description=(
        "Upload a vehicle image (CCTV frame) + customer text + optional metadata. "
        "Returns a complete service record with vehicle type, damages, intent and priority."
    ),
)
async def analyze_vehicle(
    image: UploadFile = File(..., description="JPEG/PNG image of the vehicle"),
    customer_text: str = Form(..., description="Customer's service request in plain text"),
    metadata_json: Optional[str] = Form(
        default=None,
        description="Optional JSON string matching VehicleMetadataIn schema",
    ),
) -> ServiceRecord:
    # Parse optional metadata
    raw_meta: dict = {}
    vehicle_id: Optional[str] = None
    license_plate: Optional[str] = None

    if metadata_json:
        try:
            meta_dict = json.loads(metadata_json)
            meta = VehicleMetadataIn(**meta_dict)
            vehicle_id    = meta.vehicle_id
            license_plate = meta.license_plate
            raw_meta = meta.model_dump(exclude_none=True)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid metadata_json: {exc}",
            )

    pil_image = _load_image(image)

    try:
        record = _get_pipeline().run(
            image=pil_image,
            customer_text=customer_text,
            raw_metadata=raw_meta or None,
            vehicle_id=vehicle_id,
            license_plate=license_plate,
        )
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )

    return record


# ── POST /analyze/image  (image-only) ─────────────────────────────────────────

@router.post(
    "/analyze/image",
    summary="Image-only analysis (vehicle type + damage)",
    response_model=dict,
)
async def analyze_image_only(
    image: UploadFile = File(...),
) -> dict:
    pil_image = _load_image(image)
    pipeline = _get_pipeline()

    clf = pipeline.vehicle_classifier.predict(pil_image)
    dmg = pipeline.damage_detector.detect(pil_image)

    return {
        "vehicle_type": clf["vehicle_type"],
        "vehicle_confidence": clf["confidence"],
        "all_scores": clf["all_scores"],
        "detected_damage": dmg.detected_damages,
        "damage_severity": dmg.overall_severity,
        "damage_details": [d.model_dump() for d in dmg.damage_items],
        "sources": {
            "classifier": clf["source"],
            "damage_detector": dmg.source,
        },
    }


# ── POST /analyze/text  (text-only) ───────────────────────────────────────────

@router.post(
    "/analyze/text",
    summary="Text-only intent extraction",
    response_model=IntentResult,
)
async def analyze_text_only(
    customer_text: str = Form(...),
) -> IntentResult:
    pipeline = _get_pipeline()
    return pipeline.intent_extractor.extract(customer_text)


# ── GET /vehicle-types ─────────────────────────────────────────────────────────

@router.get("/vehicle-types", summary="List supported vehicle categories")
async def get_vehicle_types() -> dict:
    return {"vehicle_types": settings.vehicle_classes}


# ── GET /damage-types ──────────────────────────────────────────────────────────

@router.get("/damage-types", summary="List supported damage categories")
async def get_damage_types() -> dict:
    return {"damage_types": settings.damage_classes}
