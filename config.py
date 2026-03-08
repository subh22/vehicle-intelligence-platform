"""
Centralised configuration — override any value via .env or environment variables.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Anthropic ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    claude_model: str = "claude-haiku-4-5-20251001"

    # ── Paths ──────────────────────────────────────────────────────────────────
    base_dir: Path = Path(__file__).parent
    model_dir: Path = base_dir / "models"
    vehicle_classifier_path: Path = model_dir / "vehicle_classifier.pth"
    damage_classifier_path: Path = model_dir / "damage_classifier.pth"

    vehicle_dataset_path: Path = base_dir / "data" / "vehicle_classification"
    damage_dataset_path: Path = base_dir / "data" / "car_damage_detection"
    metadata_csv_path: Path = base_dir / "data" / "car_data.csv"

    # ── API ────────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # ── Vision ─────────────────────────────────────────────────────────────────
    # Kaggle "Vehicle Classification Dataset" — 7 categories
    vehicle_classes: List[str] = [
        "Bus", "Car", "Motorcycle", "SUV", "Truck", "Van", "Bicycle"
    ]
    # eashankaushik damage dataset classes
    damage_classes: List[str] = [
        "dent", "scratch", "shatter", "dislocation", "normal"
    ]
    image_size: int = 224
    vehicle_confidence_threshold: float = 0.55
    damage_confidence_threshold: float = 0.45

    # ── Priority escalation rules ──────────────────────────────────────────────
    high_priority_damages: List[str] = [
        "shatter", "dislocation", "frame damage", "engine damage", "major collision"
    ]
    high_priority_intents: List[str] = [
        "emergency", "insurance claim", "accident repair", "brake failure"
    ]
    medium_priority_intents: List[str] = [
        "repair", "replacement", "warranty claim", "recurring issue"
    ]


settings = Settings()
