"""
CLI Inference Script
---------------------
Run the full pipeline from the command line without starting the API server.

Usage:
    python scripts/run_inference.py \
        --image path/to/vehicle.jpg \
        --text "My SUV was scratched in a parking lot. Need repair estimate." \
        --metadata '{"year": 2021, "km_driven": 40000, "fuel_type": "Diesel"}'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Vehicle Intelligence Platform — CLI inference")
    parser.add_argument("--image",    required=True,  type=Path, help="Path to vehicle image")
    parser.add_argument("--text",     required=True,  type=str,  help="Customer service request text")
    parser.add_argument("--metadata", default=None,   type=str,  help="JSON string of vehicle metadata")
    args = parser.parse_args()

    # Validate image
    if not args.image.exists():
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"ERROR: Cannot open image: {e}")
        sys.exit(1)

    # Parse metadata
    raw_metadata: dict | None = None
    if args.metadata:
        try:
            raw_metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid metadata JSON: {e}")
            sys.exit(1)

    # Run pipeline
    from core.pipeline import VehicleIntelligencePipeline

    print("\nInitialising Vehicle Intelligence Pipeline …")
    pipeline = VehicleIntelligencePipeline()

    print("Running inference …\n")
    record = pipeline.run(
        image=image,
        customer_text=args.text,
        raw_metadata=raw_metadata,
    )

    # Pretty-print result
    result = {
        "vehicle_type":          record.vehicle_type,
        "detected_damage":       record.detected_damage,
        "customer_intent":       record.customer_intent,
        "service_priority":      record.service_priority,
        "vehicle_confidence":    record.vehicle_confidence,
        "damage_severity":       record.damage_severity,
        "urgency_level":         record.urgency_level,
        "key_customer_concerns": record.key_customer_concerns,
        "vehicle_metadata":      record.vehicle_metadata,
        "processing_time_ms":    record.processing_time_ms,
        "sources": {
            "vehicle_classifier": record.classifier_source,
            "damage_detector":    record.damage_source,
        },
    }

    print("=" * 60)
    print("VEHICLE SERVICE RECORD")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))
    print("=" * 60)


if __name__ == "__main__":
    main()
