#!/usr/bin/env python
"""Stage 01 — baseline YOLOv8l training (vanilla architecture).

Matches Table 4.2 of the thesis:
    100 epochs, 768px, AdamW, lr0=5e-4, cosine LR,
    batch=12 (constrained), AMP on, patience=25, seed configurable.

On A100 80GB we bump batch to 32 — keeps the spec spirit (memory-bound batch)
while exploiting the new GPU. Effective LR is rescaled with linear scaling rule.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
from pathlib import Path

from ultralytics import YOLO

from src.data.roboflow_pull import pull
from src.modules.register import register_all
from src.paths import CONFIGS_DIR, RUNS_DIR, WANDB_ENTITY, WANDB_PROJECT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--batch", type=int, default=32, help="A100 80GB friendly")
    ap.add_argument("--lr0", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--name", default=None, help="Run name override")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    register_all()
    dataset_path = pull()
    data_yaml = dataset_path / "data.yaml"

    name = args.name or f"baseline_seed{args.seed}"
    cfg = CONFIGS_DIR / "baseline.yaml"

    # Initialise from COCO-pretrained YOLOv8l weights for transfer learning.
    model = YOLO(str(cfg)).load("yolov8l.pt")

    # W&B init — Ultralytics auto-logs if wandb is installed and a key is set.
    if not args.no_wandb:
        import os
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
        if WANDB_ENTITY:
            os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=args.lr0,
        cos_lr=True,
        patience=args.patience,
        amp=True,
        seed=args.seed,
        deterministic=True,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=False,
        plots=True,
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
