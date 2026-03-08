"""
Multi-Modal Inference Pipeline
-------------------------------
Orchestrates Vehicle Classifier  +  Damage Detector  +  Intent Extractor
+  Metadata Processor into a single structured ServiceRecord.

Priority logic
--------------
HIGH   : emergency / insurance_claim intent  OR  severe damage (shatter / dislocation)
MEDIUM : repair / warranty_claim intent  OR  moderate damage (dent / dislocation)
LOW    : regular_service / inspection  OR  minor / no damage
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from PIL import Image

from api.schemas import DamageResult, IntentResult, ServiceRecord
from config import settings
from core.vision.vehicle_classifier import VehicleClassifier
from core.vision.damage_detector import DamageDetector
from core.nlp.intent_extractor import IntentExtractor
from core.data.metadata_processor import MetadataProcessor

logger = logging.getLogger(__name__)


class VehicleIntelligencePipeline:
    """
    Singleton-friendly pipeline class.

    Usage::

        pipeline = VehicleIntelligencePipeline()

        record = pipeline.run(
            image=pil_image,
            customer_text="My car was rear-ended. I need an insurance claim.",
            raw_metadata={"year": 2020, "km_driven": 55000, "fuel_type": "Petrol"},
        )
    """

    def __init__(self) -> None:
        logger.info("Initialising Vehicle Intelligence Pipeline …")
        self.vehicle_classifier = VehicleClassifier()
        self.damage_detector    = DamageDetector()
        self.intent_extractor   = IntentExtractor()
        self.metadata_processor = MetadataProcessor()
        logger.info("Pipeline ready.")

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(
        self,
        image: Image.Image,
        customer_text: str,
        raw_metadata: Optional[Dict[str, Any]] = None,
        vehicle_id: Optional[str] = None,
        license_plate: Optional[str] = None,
    ) -> ServiceRecord:
        t0 = time.perf_counter()

        # ── 1. Vehicle type classification ─────────────────────────────────────
        clf_result = self.vehicle_classifier.predict(image)
        logger.debug("Vehicle: %s (%.2f)", clf_result["vehicle_type"], clf_result["confidence"])

        # ── 2. Damage detection ────────────────────────────────────────────────
        dmg_result: DamageResult = self.damage_detector.detect(image)
        logger.debug("Damages: %s | severity=%s", dmg_result.detected_damages, dmg_result.overall_severity)

        # ── 3. Customer intent extraction ──────────────────────────────────────
        intent_result: IntentResult = self.intent_extractor.extract(customer_text)
        logger.debug("Intent: %s | urgency=%s", intent_result.customer_intent, intent_result.urgency)

        # ── 4. Metadata enrichment ─────────────────────────────────────────────
        enriched_meta: Optional[Dict[str, Any]] = None
        if vehicle_id:
            enriched_meta = self.metadata_processor.get_by_id(vehicle_id)
        elif license_plate:
            enriched_meta = self.metadata_processor.get_by_plate(license_plate)

        if raw_metadata and not enriched_meta:
            enriched_meta = self.metadata_processor.enrich(raw_metadata)
        elif raw_metadata and enriched_meta:
            # Merge: API request values take precedence over CSV lookup
            enriched_meta = {**enriched_meta, **self.metadata_processor.enrich(raw_metadata)}

        # ── 5. Priority calculation ────────────────────────────────────────────
        priority = _calculate_priority(
            damage_severity=dmg_result.overall_severity,
            damage_types=[item.damage_type for item in dmg_result.damage_items],
            intent=intent_result.customer_intent,
            urgency=intent_result.urgency,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return ServiceRecord(
            # ── Core spec output ───────────────────────────────────────────────
            vehicle_type=clf_result["vehicle_type"],
            detected_damage=dmg_result.detected_damages,
            customer_intent=intent_result.customer_intent,
            service_priority=priority,
            # ── Enriched signals ───────────────────────────────────────────────
            vehicle_confidence=clf_result["confidence"],
            damage_severity=dmg_result.overall_severity,
            urgency_level=intent_result.urgency,
            key_customer_concerns=intent_result.key_concerns,
            damage_details=dmg_result.damage_items,
            vehicle_metadata=enriched_meta,
            # ── Audit ──────────────────────────────────────────────────────────
            processing_time_ms=elapsed_ms,
            classifier_source=clf_result["source"],
            damage_source=dmg_result.source,
        )


# ── Priority logic ─────────────────────────────────────────────────────────────

def _calculate_priority(
    damage_severity: str,
    damage_types: list[str],
    intent: str,
    urgency: str,
) -> str:
    # Immediately HIGH
    if urgency == "high":
        return "high"
    if intent in settings.high_priority_intents:
        return "high"
    if damage_severity == "severe":
        return "high"
    if any(d in settings.high_priority_damages for d in damage_types):
        return "high"

    # MEDIUM
    if urgency == "medium":
        return "medium"
    if intent in settings.medium_priority_intents:
        return "medium"
    if damage_severity == "moderate":
        return "medium"

    # Default LOW
    return "low"
