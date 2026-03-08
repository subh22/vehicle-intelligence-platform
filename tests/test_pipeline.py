"""
Integration tests for the Vehicle Intelligence Pipeline.

Run with:
    pytest tests/ -v

Note: tests that call Claude API require ANTHROPIC_API_KEY in environment.
      Mark these with @pytest.mark.integration to skip in CI without a key.
"""
from __future__ import annotations

import io
import json
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def _make_test_image(color: tuple = (100, 149, 237), size: tuple = (640, 480)) -> bytes:
    """Create a minimal valid JPEG image for testing."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


# ── Health check ───────────────────────────────────────────────────────────────

def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "components" in body


# ── Vehicle types / damage types ───────────────────────────────────────────────

def test_vehicle_types_endpoint(client):
    resp = client.get("/api/v1/vehicle-types")
    assert resp.status_code == 200
    data = resp.json()
    assert "vehicle_types" in data
    assert "Car" in data["vehicle_types"]


def test_damage_types_endpoint(client):
    resp = client.get("/api/v1/damage-types")
    assert resp.status_code == 200
    data = resp.json()
    assert "damage_types" in data
    assert "dent" in data["damage_types"]


# ── Text-only intent extraction (mocked) ──────────────────────────────────────

def _mock_intent_response():
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock()]
    mock_resp.content[0].text = json.dumps({
        "customer_intent": "insurance_claim",
        "urgency": "high",
        "key_concerns": ["rear bumper damage", "insurance documentation"],
        "sentiment": "negative",
        "reasoning": "Customer mentions accident and insurance.",
    })
    return mock_resp


def test_text_only_endpoint_mocked(client):
    with patch("anthropic.Anthropic.messages") as mock_messages:
        mock_messages.create.return_value = _mock_intent_response()
        with patch("core.nlp.intent_extractor.IntentExtractor.extract") as mock_extract:
            from api.schemas import IntentResult
            mock_extract.return_value = IntentResult(
                customer_intent="insurance_claim",
                urgency="high",
                key_concerns=["rear bumper damage"],
                sentiment="negative",
            )
            resp = client.post(
                "/api/v1/analyze/text",
                data={"customer_text": "My car was rear-ended. I need an insurance claim."},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["customer_intent"] == "insurance_claim"
    assert body["urgency"] == "high"


# ── Full pipeline — unit test with mocked components ──────────────────────────

@pytest.fixture
def mock_pipeline():
    from api.schemas import DamageItem, DamageResult, IntentResult
    from core.pipeline import VehicleIntelligencePipeline

    pipeline = MagicMock(spec=VehicleIntelligencePipeline)

    # Mock vehicle classifier
    pipeline.vehicle_classifier.predict.return_value = {
        "vehicle_type": "SUV",
        "confidence": 0.92,
        "all_scores": {"SUV": 0.92, "Car": 0.05},
        "source": "local_model",
    }

    # Mock damage detector
    pipeline.damage_detector.detect.return_value = DamageResult(
        detected_damages=["rear bumper dent"],
        damage_items=[
            DamageItem(damage_type="dent", location="rear bumper", severity="moderate", confidence=0.87)
        ],
        overall_severity="moderate",
        source="claude_vision",
    )

    # Mock intent extractor
    pipeline.intent_extractor.extract.return_value = IntentResult(
        customer_intent="insurance_claim",
        urgency="high",
        key_concerns=["accident damage", "insurance claim required"],
        sentiment="negative",
    )

    # Mock metadata processor
    pipeline.metadata_processor.get_by_id.return_value = None
    pipeline.metadata_processor.get_by_plate.return_value = None
    pipeline.metadata_processor.enrich.return_value = {"year": 2020, "km_driven": 55000}

    return pipeline


def test_full_pipeline_logic(mock_pipeline):
    """Test the priority calculation and record assembly directly."""
    from core.pipeline import _calculate_priority

    priority = _calculate_priority(
        damage_severity="moderate",
        damage_types=["dent"],
        intent="insurance_claim",
        urgency="high",
    )
    assert priority == "high"

    priority_low = _calculate_priority(
        damage_severity="none",
        damage_types=[],
        intent="regular_service",
        urgency="low",
    )
    assert priority_low == "low"

    priority_med = _calculate_priority(
        damage_severity="moderate",
        damage_types=["scratch"],
        intent="repair",
        urgency="medium",
    )
    assert priority_med == "medium"


def test_severe_damage_escalates_to_high():
    from core.pipeline import _calculate_priority

    priority = _calculate_priority(
        damage_severity="severe",
        damage_types=["shatter"],
        intent="regular_service",   # low-priority intent
        urgency="low",
    )
    assert priority == "high"


# ── Full API endpoint — mocked pipeline ───────────────────────────────────────

def test_analyze_endpoint_mocked(client, mock_pipeline):
    from api.schemas import ServiceRecord

    mock_record = ServiceRecord(
        vehicle_type="SUV",
        detected_damage=["rear bumper dent"],
        customer_intent="insurance_claim",
        service_priority="high",
        vehicle_confidence=0.92,
        damage_severity="moderate",
        urgency_level="high",
        key_customer_concerns=["accident damage"],
        damage_details=[],
        vehicle_metadata=None,
        processing_time_ms=342.5,
        classifier_source="local_model",
        damage_source="claude_vision",
    )

    with patch("api.routes.vehicle._get_pipeline", return_value=mock_pipeline):
        mock_pipeline.run.return_value = mock_record

        img_bytes = _make_test_image()
        resp = client.post(
            "/api/v1/analyze",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"customer_text": "My car was rear-ended. Insurance claim needed."},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["vehicle_type"] == "SUV"
    assert "rear bumper dent" in body["detected_damage"]
    assert body["customer_intent"] == "insurance_claim"
    assert body["service_priority"] == "high"


# ── Schema validation tests ────────────────────────────────────────────────────

def test_analyze_missing_image(client):
    resp = client.post(
        "/api/v1/analyze",
        data={"customer_text": "Oil change needed."},
    )
    assert resp.status_code == 422          # Unprocessable Entity — image required


def test_analyze_missing_text(client):
    img_bytes = _make_test_image()
    resp = client.post(
        "/api/v1/analyze",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 422


# ── Integration tests (requires real ANTHROPIC_API_KEY) ───────────────────────

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No API key set")
def test_intent_extractor_real():
    from core.nlp.intent_extractor import IntentExtractor

    extractor = IntentExtractor()
    result = extractor.extract(
        "I had an accident last night and my front bumper is completely smashed. "
        "I need to file an insurance claim urgently."
    )
    assert result.customer_intent == "insurance_claim"
    assert result.urgency == "high"
    assert len(result.key_concerns) > 0
