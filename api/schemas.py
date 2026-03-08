"""
Pydantic schemas for all API request / response models.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────────

class VehicleMetadataIn(BaseModel):
    """Optional structured metadata about the vehicle (from CarDekho CSV or manual entry)."""
    vehicle_id: Optional[str] = None
    license_plate: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    fuel_type: Optional[str] = None
    transmission: Optional[str] = None
    ownership: Optional[str] = None          # "First Owner", "Second Owner" …
    km_driven: Optional[int] = None
    last_service_date: Optional[str] = None  # ISO date string


class AnalyzeRequest(BaseModel):
    """
    Main analysis request.
    Image is uploaded as multipart/form-data; this body is sent as a JSON part.
    """
    customer_text: str = Field(
        ...,
        description="Free-form text from the customer describing their service need.",
        example="My car was rear-ended last week. The bumper is cracked and I need an insurance claim.",
    )
    metadata: Optional[VehicleMetadataIn] = Field(
        default=None,
        description="Optional structured vehicle metadata to enrich the service record.",
    )


# ── Internal result models ─────────────────────────────────────────────────────

class ClassifierResult(BaseModel):
    vehicle_type: str
    confidence: float
    all_scores: Dict[str, float]
    source: str  # "local_model" | "claude_vision"


class DamageItem(BaseModel):
    damage_type: str       # e.g. "dent", "scratch"
    location: str          # e.g. "rear bumper", "driver-side door"
    severity: str          # "minor" | "moderate" | "severe"
    confidence: float


class DamageResult(BaseModel):
    detected_damages: List[str]          # human-readable list  e.g. ["rear bumper dent"]
    damage_items: List[DamageItem]       # structured detail
    overall_severity: str                # "none" | "minor" | "moderate" | "severe"
    source: str                          # "local_model" | "claude_vision"


class IntentResult(BaseModel):
    customer_intent: str                 # primary intent label
    urgency: str                         # "low" | "medium" | "high"
    key_concerns: List[str]             # bullet points from customer text
    sentiment: str                       # "positive" | "neutral" | "negative"


# ── Final service record ───────────────────────────────────────────────────────

class ServiceRecord(BaseModel):
    # ── Core output (matches assignment spec) ─────────────────────────────
    vehicle_type: str
    detected_damage: List[str]
    customer_intent: str
    service_priority: str                # "low" | "medium" | "high"

    # ── Enriched signals ──────────────────────────────────────────────────
    vehicle_confidence: float
    damage_severity: str
    urgency_level: str
    key_customer_concerns: List[str]
    damage_details: List[DamageItem]
    vehicle_metadata: Optional[Dict[str, Any]] = None

    # ── Audit / traceability ──────────────────────────────────────────────
    processing_time_ms: float
    classifier_source: str
    damage_source: str


# ── Misc response models ───────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
