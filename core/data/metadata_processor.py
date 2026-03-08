"""
Vehicle Metadata Processor
---------------------------
Loads the CarDekho vehicle metadata CSV and enriches a service request
with structured vehicle profile information.

Dataset: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
Expected CSV columns: name, year, selling_price, km_driven, fuel, seller_type,
                      transmission, owner
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class MetadataProcessor:
    """
    Loads the CarDekho CSV once and provides vehicle profile look-ups.

    Usage::

        proc = MetadataProcessor()

        # Look up by vehicle_id (row index)
        profile = proc.get_by_id("42")

        # Look up by license plate (if column exists)
        profile = proc.get_by_plate("MH12AB1234")

        # Enrich a raw metadata dict
        enriched = proc.enrich({"make": "Maruti", "year": 2019, "km_driven": 45000})
    """

    # Column name aliases so we handle both raw CarDekho and cleaned CSVs
    _COLUMN_ALIASES: Dict[str, str] = {
        "name":          "vehicle_name",
        "year":          "year",
        "selling_price": "price",
        "km_driven":     "km_driven",
        "fuel":          "fuel_type",
        "seller_type":   "seller_type",
        "transmission":  "transmission",
        "owner":         "ownership",
    }

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None
        self._load()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        csv_path: Path = settings.metadata_csv_path
        if not csv_path.exists():
            logger.info("Metadata CSV not found at %s — metadata enrichment disabled.", csv_path)
            return
        try:
            df = pd.read_csv(csv_path)
            # Normalise column names
            df = df.rename(columns={k: v for k, v in self._COLUMN_ALIASES.items() if k in df.columns})
            # Add computed columns
            if "year" in df.columns:
                df["vehicle_age_years"] = 2024 - df["year"].fillna(2020).astype(int)
            if "km_driven" in df.columns:
                df["mileage_category"] = pd.cut(
                    df["km_driven"].fillna(0),
                    bins=[0, 30_000, 70_000, 150_000, float("inf")],
                    labels=["low", "medium", "high", "very_high"],
                )
            self._df = df.reset_index(drop=True)
            logger.info("Metadata CSV loaded: %d records.", len(self._df))
        except Exception as exc:
            logger.warning("Could not load metadata CSV: %s", exc)

    # ── Public helpers ─────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._df is not None and not self._df.empty

    def get_by_id(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve vehicle profile by integer row index."""
        if not self.available:
            return None
        try:
            idx = int(vehicle_id)
            if 0 <= idx < len(self._df):
                return self._clean_record(self._df.iloc[idx].to_dict())
        except (ValueError, IndexError):
            pass
        return None

    def get_by_plate(self, plate: str) -> Optional[Dict[str, Any]]:
        """Retrieve vehicle profile by license_plate column (if present)."""
        if not self.available or "license_plate" not in self._df.columns:
            return None
        mask = self._df["license_plate"].str.upper() == plate.upper()
        if mask.any():
            return self._clean_record(self._df[mask].iloc[0].to_dict())
        return None

    def enrich(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take a raw metadata dict (from the API request) and add computed fields.
        Returns the enriched dict (or the original if nothing to add).
        """
        out = {k: v for k, v in raw.items() if v is not None}

        year = out.get("year")
        if year:
            out["vehicle_age_years"] = 2024 - int(year)

        km = out.get("km_driven")
        if km is not None:
            km = int(km)
            if km < 30_000:
                out["mileage_category"] = "low"
            elif km < 70_000:
                out["mileage_category"] = "medium"
            elif km < 150_000:
                out["mileage_category"] = "high"
            else:
                out["mileage_category"] = "very_high"

        ownership = out.get("ownership", "")
        if ownership:
            out["ownership_flag"] = "multi_owner" if "Second" in ownership or "Third" in ownership else "first_owner"

        return out

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Drop NaN / NaT values and convert numpy types to Python natives."""
        import math
        cleaned = {}
        for k, v in record.items():
            if isinstance(v, float) and math.isnan(v):
                continue
            if hasattr(v, "item"):          # numpy scalar
                v = v.item()
            cleaned[k] = v
        return cleaned
