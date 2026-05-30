#!/usr/bin/env python
"""Stage 03 — distill the YOLOv8l+ teacher into a YOLOv8n student.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 pipeline
---------------------------------
Knowledge distillation is how the RADS pipeline turns the accurate-but-heavy
enhanced detector into something deployable on a phone. The large enhanced
model trained in Stage 02 (typically the "combined" variant) acts as the
*teacher*; a tiny YOLOv8n *student* is trained to mimic it. The student keeps
most of the teacher's detection quality on the RADS classes
["MH", "PH", "WLPH"] at a fraction of the compute, which is what later gets
quantized (Stage 04) and benchmarked for mobile FPS (Stage 07).

Training uses a composite objective so the student learns from both the
ground-truth labels and the teacher's "dark knowledge":
    * 0.4 * task loss          — the normal YOLO detection loss vs. labels.
    * 0.4 * KL classification  — KL divergence between student/teacher class
                                 logits softened at temperature T=4.
    * 0.2 * feature MSE         — MSE between intermediate feature maps so the
                                 student's representations align with the
                                 teacher's.
The actual distillation mechanics live in
``src.distill.trainer.DistillationTrainer``; this script only wires up the
overrides dict that configures it.

Run after the combined (or best ablation) teacher has been trained.

Composite loss: 0.4 task + 0.4 KL classification (T=4) + 0.2 feature MSE.

Example:
    python scripts/03_train_distill.py \
        --teacher artifacts/runs/combined_seed42/weights/best.pt \
        --seed 42
"""
from __future__ import annotations
import sys, os
# Make the repo root importable so ``src.*`` resolves when run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import os
from pathlib import Path

from src.data.roboflow_pull import pull
from src.distill.trainer import DistillationTrainer
from src.modules.register import register_all
from src.paths import CONFIGS_DIR, RUNS_DIR, WANDB_ENTITY, WANDB_PROJECT


def main():
    """Parse CLI args and distill the teacher into a YOLOv8n student.

    Steps performed, in order:
      1. Validate that the teacher checkpoint exists (fail fast otherwise).
      2. Register custom modules and ensure the dataset is available.
      3. Assemble an ``overrides`` dict combining the standard Ultralytics
         training knobs with the distillation-specific kwargs
         (teacher path + the three loss weights + temperature) that
         DistillationTrainer consumes.
      4. Construct the custom DistillationTrainer with those overrides and
         run it.

    No value is returned; the student weights land under RUNS_DIR/<name>/.
    """
    ap = argparse.ArgumentParser()
    # --teacher (required): path to the trained teacher checkpoint (best.pt),
    #           usually the combined variant from Stage 02. Its logits and
    #           features supervise the student.
    ap.add_argument("--teacher", required=True, type=Path, help="Path to teacher best.pt")
    # --seed: RNG seed for reproducibility; also names the run.
    ap.add_argument("--seed", type=int, default=42)
    # --epochs: training epoch budget (100 to match the other stages).
    ap.add_argument("--epochs", type=int, default=100)
    # --imgsz: square training image size in px; kept at 768 for training.
    ap.add_argument("--imgsz", type=int, default=768)
    # --batch: images per step. The YOLOv8n student is tiny, so a larger
    #          batch (64) fits comfortably and stabilises distillation.
    ap.add_argument("--batch", type=int, default=64, help="student is small, bigger batch ok")
    # --lr0: initial AdamW learning rate (cosine schedule decays from here).
    ap.add_argument("--lr0", type=float, default=5e-4)
    # --patience: early-stopping patience in epochs without val improvement.
    ap.add_argument("--patience", type=int, default=25)
    # --task-w: weight on the standard detection (task) loss vs. ground truth.
    #           Default 0.4 per the thesis composite-loss recipe.
    ap.add_argument("--task-w", type=float, default=0.4)
    # --kl-w: weight on the temperature-softened KL classification loss that
    #         transfers the teacher's class-probability "dark knowledge". 0.4.
    ap.add_argument("--kl-w", type=float, default=0.4)
    # --feat-w: weight on the intermediate feature-map MSE alignment term. 0.2.
    ap.add_argument("--feat-w", type=float, default=0.2)
    # --T: softmax temperature for the KL term. Higher T softens the teacher's
    #      distribution, exposing inter-class similarity. Thesis uses T=4.
    ap.add_argument("--T", type=float, default=4.0)
    # --name: optional run-name override; defaults to "distill_seed<seed>".
    ap.add_argument("--name", default=None)
    # --no-wandb: disable Weights & Biases logging.
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # Fail fast with a clear message if the teacher weights are missing —
    # distillation is meaningless without a trained teacher to imitate.
    if not args.teacher.exists():
        raise SystemExit(f"Teacher weights not found: {args.teacher}")

    # Register custom modules (the student/teacher configs may reference them)
    # and ensure the dataset cache is populated.
    register_all()
    dataset_path = pull()
    data_yaml = dataset_path / "data.yaml"

    # Stable per-seed run name and the small-student architecture config.
    name = args.name or f"distill_seed{args.seed}"
    student_cfg = CONFIGS_DIR / "distill_student.yaml"

    # Optional W&B logging; setdefault lets externally-set env vars win.
    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
        if WANDB_ENTITY:
            os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)

    # Build the full config for DistillationTrainer: standard Ultralytics
    # training args plus the custom distill_* kwargs the trainer reads to set
    # up the teacher forward pass and weight the three loss components.
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

    # Instantiate the custom trainer with our overrides and run it. The trainer
    # internally loads the frozen teacher and injects the composite distillation
    # loss during each training step.
    trainer = DistillationTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
