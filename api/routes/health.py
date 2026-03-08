"""Health-check endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from api.schemas import HealthResponse
from config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """Returns service status and component availability."""
    components: dict[str, str] = {}

    # Local vehicle classifier
    vc_path = settings.vehicle_classifier_path
    components["vehicle_classifier"] = "local_model" if vc_path.exists() else "claude_vision_fallback"

    # Local damage classifier
    dc_path = settings.damage_classifier_path
    components["damage_detector"] = "local_model" if dc_path.exists() else "claude_vision_fallback"

    # Claude API reachability (lightweight check — no actual call)
    components["llm_intent_extractor"] = (
        "ready" if settings.anthropic_api_key else "missing_api_key"
    )

    # Metadata CSV
    csv_path = settings.metadata_csv_path
    components["metadata_processor"] = "ready" if csv_path.exists() else "csv_not_found"

    return HealthResponse(status="ok", components=components)
