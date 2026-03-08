"""
Train Vehicle Type Classifier
------------------------------
Fine-tunes EfficientNet-B0 on the Kaggle Vehicle Classification Dataset.

Usage:
    python -m training.train_classifier \
        --epochs 20 --batch-size 32 --lr 1e-4 --output models/vehicle_classifier.pth

Dataset: ~5600 images, 7 vehicle categories
Expected accuracy: 90–95% after 15–20 epochs
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import settings
from core.vision.vehicle_classifier import EfficientNetClassifier
from training.data_prep import get_vehicle_dataloaders, print_dataset_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def train(
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    output_path: Path = settings.vehicle_classifier_path,
    num_workers: int = 4,
    patience: int = 5,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # ── Data ───────────────────────────────────────────────────────────────────
    train_loader, val_loader, classes = get_vehicle_dataloaders(
        batch_size=batch_size, num_workers=num_workers
    )
    print_dataset_stats(train_loader, "Train")
    print_dataset_stats(val_loader, "Val")
    logger.info("Classes: %s", classes)

    num_classes = len(classes)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = EfficientNetClassifier(num_classes).to(device)

    # Freeze backbone for first 5 epochs (feature extraction phase)
    for param in model.backbone.features.parameters():
        param.requires_grad = False

    # ── Loss / Optimiser / Scheduler ───────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_acc = 0.0
    no_improve   = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after 5 epochs for fine-tuning
        if epoch == 6:
            logger.info("Unfreezing backbone for fine-tuning …")
            for param in model.backbone.features.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=lr / 10, weight_decay=1e-4)

        # ── Train phase ────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.perf_counter()

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

        scheduler.step()

        # ── Validation phase ───────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss    += loss.item() * images.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total
        epoch_time = time.perf_counter() - t0

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.2f%% | val_loss=%.4f val_acc=%.2f%% | %.1fs",
            epoch, epochs,
            train_loss / train_total, train_acc * 100,
            val_loss / val_total, val_acc * 100,
            epoch_time,
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            logger.info("  ✓ Best model saved (val_acc=%.2f%%)", val_acc * 100)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    logger.info("Training complete. Best val_acc: %.2f%% | Model: %s", best_val_acc * 100, output_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train vehicle type classifier")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--output",      type=Path,  default=settings.vehicle_classifier_path)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output,
        num_workers=args.num_workers,
        patience=args.patience,
    )
