"""
Dataset Preparation Utilities
------------------------------
Handles both:
  1. Vehicle Classification Dataset  (Kaggle ~5600 images, 7 classes)
  2. Car Damage Detection Dataset    (eashankaushik — stage1 + stage2)

Expected folder layout after Kaggle download:

  data/vehicle_classification/
      train/
          Bus/      Car/      Motorcycle/  SUV/  Truck/  Van/  Bicycle/
      val/          (same structure)
      test/         (same structure, optional)

  data/car_damage_detection/
      training/
          00-damage/   01-whole/
      validation/
          00-damage/   01-whole/

  (stage2 damage types — put in subfolders after manual labelling or use stage2 zip)
  data/car_damage_stage2/
      training/
          01-dent/  02-scratch/  03-shatter/  04-dislocation/
      validation/
          ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder

from config import settings

# ── Augmentation pipelines ─────────────────────────────────────────────────────

TRAIN_AUG = A.Compose([
    A.Resize(settings.image_size, settings.image_size),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
    A.OneOf([
        A.GaussianBlur(blur_limit=3),
        A.MotionBlur(blur_limit=5),
    ], p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

VAL_AUG = A.Compose([
    A.Resize(settings.image_size, settings.image_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ── Dataset wrapper (uses Albumentations instead of torchvision transforms) ────

class AlbuImageFolder(Dataset):
    """ImageFolder-compatible dataset that applies Albumentations transforms."""

    def __init__(self, root: Path, transform: A.Compose) -> None:
        self._base = ImageFolder(str(root))
        self.transform = transform
        self.classes = self._base.classes
        self.class_to_idx = self._base.class_to_idx

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self._base.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        aug = self.transform(image=img)
        return aug["image"], label


# ── DataLoader factories ───────────────────────────────────────────────────────

def get_vehicle_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    """
    Returns (train_loader, val_loader, class_names) for the vehicle classification task.
    Falls back to a random 80/20 split if no explicit val/ folder exists.
    """
    base: Path = settings.vehicle_dataset_path
    train_dir = base / "train"
    val_dir   = base / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_dir}. "
            "Download the Kaggle dataset and place it under data/vehicle_classification/."
        )

    train_ds = AlbuImageFolder(train_dir, TRAIN_AUG)

    if val_dir.exists():
        val_ds = AlbuImageFolder(val_dir, VAL_AUG)
    else:
        # 80/20 split from train folder
        n_val = max(1, int(0.2 * len(train_ds)))
        n_train = len(train_ds) - n_val
        train_ds, val_ds = random_split(train_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    class_names = (
        train_ds.classes
        if hasattr(train_ds, "classes")
        else train_ds.dataset.classes  # after random_split
    )
    return train_loader, val_loader, class_names


def get_damage_dataloaders(
    dataset_path: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    """
    Returns (train_loader, val_loader, class_names) for the damage detection task.
    Supports both stage1 (binary) and stage2 (multi-class) layouts.
    """
    base: Path = dataset_path or settings.damage_dataset_path
    train_dir = base / "training"
    val_dir   = base / "validation"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Damage dataset not found at {train_dir}. "
            "Download eashankaushik/car-damage-detection from Kaggle."
        )

    train_ds = AlbuImageFolder(train_dir, TRAIN_AUG)

    if val_dir.exists():
        val_ds = AlbuImageFolder(val_dir, VAL_AUG)
    else:
        n_val = max(1, int(0.2 * len(train_ds)))
        n_train = len(train_ds) - n_val
        train_ds, val_ds = random_split(train_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    class_names = (
        train_ds.classes
        if hasattr(train_ds, "classes")
        else train_ds.dataset.classes
    )
    return train_loader, val_loader, class_names


# ── Quick dataset stats ────────────────────────────────────────────────────────

def print_dataset_stats(loader: DataLoader, name: str) -> None:
    total = len(loader.dataset)
    print(f"\n{name}: {total} samples | {len(loader)} batches")
    if hasattr(loader.dataset, "classes"):
        for cls in loader.dataset.classes:
            count = sum(
                1 for _, lbl in loader.dataset.samples
                if lbl == loader.dataset.class_to_idx[cls]
            )
            print(f"  {cls:20s}: {count}")
