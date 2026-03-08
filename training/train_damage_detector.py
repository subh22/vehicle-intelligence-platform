"""
Train Vehicle Damage Classifier
---------------------------------
Fine-tunes EfficientNet-B0 on the eashankaushik car-damage-detection dataset.

Stage 1  (binary):   damaged  vs  whole
Stage 2  (multi):    dent  /  scratch  /  shatter  /  dislocation  /  normal

Usage:
    # Stage 1 — binary
    python -m training.train_damage_detector \
        --stage 1 --epochs 15 --output models/damage_classifier_stage1.pth

    # Stage 2 — multi-class (uses stage2 subfolder layout)
    python -m training.train_damage_detector \
        --stage 2 --dataset-path data/car_damage_stage2 \
        --epochs 20 --output models/damage_classifier.pth
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import settings
from core.vision.damage_detector import EfficientNetClassifier
from training.data_prep import get_damage_dataloaders, print_dataset_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def train(
    stage: int = 2,
    dataset_path: Path | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    output_path: Path = settings.damage_classifier_path,
    num_workers: int = 4,
    patience: int = 5,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training Stage-%d damage classifier on: %s", stage, device)

    # ── Data ───────────────────────────────────────────────────────────────────
    train_loader, val_loader, classes = get_damage_dataloaders(
        dataset_path=dataset_path, batch_size=batch_size, num_workers=num_workers
    )
    print_dataset_stats(train_loader, "Train")
    print_dataset_stats(val_loader, "Val")
    logger.info("Classes: %s", classes)
    num_classes = len(classes)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = EfficientNetClassifier(num_classes).to(device)
    for param in model.backbone.features.parameters():
        param.requires_grad = False

    # ── Loss / Optimiser ───────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_acc = 0.0
    no_improve   = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        if epoch == 6:
            logger.info("Unfreezing backbone …")
            for param in model.backbone.features.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=lr / 10, weight_decay=1e-4)

        # Train
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        t0 = time.perf_counter()

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss    += loss.item() * imgs.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)

        scheduler.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        v_loss, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                v_loss    += loss.item() * imgs.size(0)
                preds      = out.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += imgs.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_acc = v_correct / v_total
        logger.info(
            "Epoch %d/%d | train_acc=%.2f%% | val_acc=%.2f%% | %.1fs",
            epoch, epochs, (t_correct / t_total) * 100, val_acc * 100,
            time.perf_counter() - t0,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            logger.info("  ✓ Best model saved (val_acc=%.2f%%)", val_acc * 100)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping after %d stagnant epochs.", patience)
                break

    # ── Final classification report ────────────────────────────────────────────
    logger.info("Training complete. Best val_acc: %.2f%%", best_val_acc * 100)
    logger.info("\n%s", classification_report(all_labels, all_preds, target_names=classes))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train vehicle damage classifier")
    p.add_argument("--stage",        type=int,   default=2, choices=[1, 2])
    p.add_argument("--dataset-path", type=Path,  default=None)
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--patience",     type=int,   default=5)
    p.add_argument("--output",       type=Path,  default=settings.damage_classifier_path)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        stage=args.stage,
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output,
        num_workers=args.num_workers,
        patience=args.patience,
    )
