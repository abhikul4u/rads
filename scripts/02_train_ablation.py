#!/usr/bin/env python
"""Stage 02 — ablation training: train any one variant in isolation or combined.

Variants:
    cbam       — baseline + CBAM in PANet neck
    p2         — baseline + 4th detection head (stride 4)
    sizeaware  — baseline architecture + size-aware bbox loss
    combined   — all three enhancements together

The same hyperparameters as the baseline are used for fair comparison.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse

from ultralytics import YOLO

from src.data.roboflow_pull import pull
from src.losses.size_aware_loss import install_size_aware_loss
from src.modules.register import register_all
from src.paths import CONFIGS_DIR, RUNS_DIR, WANDB_ENTITY, WANDB_PROJECT

CFG_MAP = {
    "cbam": "cbam.yaml",
    "p2": "p2head.yaml",
    "sizeaware": "sizeaware.yaml",
    "combined": "combined.yaml",
}

# Variants that require size-aware loss to be installed at runtime.
SIZE_AWARE_VARIANTS = {"sizeaware", "combined"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant", required=True, choices=sorted(CFG_MAP.keys()),
        help="Which architectural enhancement to ablate"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr0", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--size-aware-alpha", type=float, default=1.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    register_all()

    if args.variant in SIZE_AWARE_VARIANTS:
        install_size_aware_loss(alpha=args.size_aware_alpha)

    dataset_path = pull()
    data_yaml = dataset_path / "data.yaml"

    name = args.name or f"{args.variant}_seed{args.seed}"
    cfg = CONFIGS_DIR / CFG_MAP[args.variant]

    # CBAM/P2 change architecture — we can still warm-start the backbone from
    # COCO YOLOv8l (matching layers load, mismatched ones reinit gracefully).
    model = YOLO(str(cfg)).load("yolov8l.pt")

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
