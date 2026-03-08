"""
Kaggle Dataset Downloader
--------------------------
Downloads all required datasets and organises them into the expected directory layout.

Prerequisites:
    pip install kaggle
    # Place kaggle.json in ~/.kaggle/  (from https://www.kaggle.com/settings/account)

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --dataset vehicles
    python scripts/download_datasets.py --dataset damage
    python scripts/download_datasets.py --dataset metadata
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path("data")

DATASETS = {
    "vehicles": {
        "kaggle_id":   "kaggle datasets download -d marquis03/vehicle-classification",
        "zip_name":    "vehicle-classification.zip",
        "output_dir":  DATA_DIR / "vehicle_classification",
        "description": "Vehicle Classification Dataset (~5600 images, 7 classes)",
    },
    "damage": {
        "kaggle_id":   "kaggle datasets download -d eashankaushik/car-damage-detection",
        "zip_name":    "car-damage-detection.zip",
        "output_dir":  DATA_DIR / "car_damage_detection",
        "description": "Car Damage Detection Dataset (dent, scratch, shatter, dislocation)",
    },
    "damage_stage1": {
        "kaggle_id":   "kaggle datasets download -d eashankaushik/car-damage-detectionstage1",
        "zip_name":    "car-damage-detectionstage1.zip",
        "output_dir":  DATA_DIR / "car_damage_stage1",
        "description": "Car Damage Stage 1 (binary: damaged / whole)",
    },
    "damage_stage2": {
        "kaggle_id":   "kaggle datasets download -d eashankaushik/car-damage-detectionstage2",
        "zip_name":    "car-damage-detectionstage2.zip",
        "output_dir":  DATA_DIR / "car_damage_stage2",
        "description": "Car Damage Stage 2 (multi-class damage types)",
    },
    "metadata": {
        "kaggle_id":   "kaggle datasets download -d nehalbirla/vehicle-dataset-from-cardekho",
        "zip_name":    "vehicle-dataset-from-cardekho.zip",
        "output_dir":  DATA_DIR / "cardekho",
        "description": "CarDekho Vehicle Metadata (model, year, fuel, ownership, price)",
    },
    "customer_support": {
        "kaggle_id":   "kaggle datasets download -d thoughtvector/customer-support-on-twitter",
        "zip_name":    "customer-support-on-twitter.zip",
        "output_dir":  DATA_DIR / "customer_support",
        "description": "Customer Support on Twitter (intent classification data)",
    },
}


def _check_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        print("ERROR: kaggle CLI not found. Install with: pip install kaggle")
        print("Then add your API token to ~/.kaggle/kaggle.json")
        sys.exit(1)


def _download(key: str) -> None:
    info = DATASETS[key]
    out_dir: Path = info["output_dir"]

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[{key}] Already exists at {out_dir} — skipping.")
        return

    print(f"\n[{key}] Downloading: {info['description']}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = info["kaggle_id"].split() + ["--path", str(DATA_DIR), "--unzip"]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Download failed for {key}.")
        return

    # Move extracted contents into output_dir if needed
    zip_path = DATA_DIR / info["zip_name"]
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        zip_path.unlink()

    print(f"[{key}] Done → {out_dir}")


def _post_process_metadata() -> None:
    """Copy the CarDekho CSV to the expected config path."""
    src_dir = DATA_DIR / "cardekho"
    if not src_dir.exists():
        return
    csv_files = list(src_dir.glob("*.csv"))
    if csv_files:
        target = DATA_DIR / "car_data.csv"
        shutil.copy(csv_files[0], target)
        print(f"Metadata CSV copied to {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kaggle datasets for Vehicle Intelligence Platform")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",     action="store_true",  help="Download all datasets")
    group.add_argument("--dataset", choices=list(DATASETS.keys()), help="Download a specific dataset")
    args = parser.parse_args()

    _check_kaggle_cli()

    if args.all:
        for key in DATASETS:
            _download(key)
        _post_process_metadata()
    else:
        _download(args.dataset)
        if args.dataset == "metadata":
            _post_process_metadata()

    print("\nAll requested datasets ready.")
    print("Next steps:")
    print("  Train vehicle classifier:  python -m training.train_classifier")
    print("  Train damage detector:     python -m training.train_damage_detector --stage 2")
    print("  Start API:                 uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
