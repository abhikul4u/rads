#!/usr/bin/env python
"""Stage 03 — distill the YOLOv8l+ teacher into a YOLOv8n student.

Composite loss: 0.4 task + 0.4 KL classification (T=4) + 0.2 feature MSE.

Run after the combined (or best ablation) teacher has been trained.
Example:
    python scripts/03_train_distill.py \
        --teacher artifacts/runs/combined_seed42/weights/best.pt \
        --seed 42
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import os
from pathlib import Path

from src.data.roboflow_pull import pull
from src.distill.trainer import DistillationTrainer
from src.modules.register import register_all
from src.paths import CONFIGS_DIR, RUNS_DIR, WANDB_ENTITY, WANDB_PROJECT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path, help="Path to teacher best.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--batch", type=int, default=64, help="student is small, bigger batch ok")
    ap.add_argument("--lr0", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--task-w", type=float, default=0.4)
    ap.add_argument("--kl-w", type=float, default=0.4)
    ap.add_argument("--feat-w", type=float, default=0.2)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if not args.teacher.exists():
        raise SystemExit(f"Teacher weights not found: {args.teacher}")

    register_all()
    dataset_path = pull()
    data_yaml = dataset_path / "data.yaml"

    name = args.name or f"distill_seed{args.seed}"
    student_cfg = CONFIGS_DIR / "distill_student.yaml"

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
        if WANDB_ENTITY:
            os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)

    overrides = dict(
        model=str(student_cfg),
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
        # Custom kwargs picked up by DistillationTrainer
        teacher_weights=str(args.teacher),
        distill_task_w=args.task_w,
        distill_kl_w=args.kl_w,
        distill_feat_w=args.feat_w,
        distill_T=args.T,
    )

    trainer = DistillationTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
