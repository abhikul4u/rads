#!/usr/bin/env python
"""Stage 01 — baseline YOLOv8l training (vanilla architecture).

Author: Rutuja Kulkarni

Role in the RADS Layer 3 pipeline
---------------------------------
This stage trains the *reference* model for the Road Anomaly Detection System
(RADS): a vanilla Ultralytics YOLOv8l detector with no architectural
enhancements. Every other variant in the pipeline (cbam, p2, sizeaware,
combined, distill) is measured *against* this baseline, so reproducing it
faithfully — and re-running it across the three seeds (42, 1337, 2024) — is
what makes the later ablation comparisons meaningful.

The baseline is warm-started from COCO-pretrained YOLOv8l weights (transfer
learning) and fine-tuned on the 3-class RADS dataset ["MH", "PH", "WLPH"]
(manhole, pothole, water-logged pothole). Results land under RUNS_DIR/<name>
and are optionally streamed to Weights & Biases.

Matches Table 4.2 of the thesis:
    100 epochs, 768px, AdamW, lr0=5e-4, cosine LR,
    batch=12 (constrained), AMP on, patience=25, seed configurable.

On A100 80GB we bump batch to 32 — keeps the spec spirit (memory-bound batch)
while exploiting the new GPU. Effective LR is rescaled with linear scaling rule.

Example CLI invocation
----------------------
    # Train the baseline for the primary seed:
    python scripts/01_train_baseline.py --seed 42

    # Reproduce across the other two seeds for mean +/- std reporting:
    python scripts/01_train_baseline.py --seed 1337
    python scripts/01_train_baseline.py --seed 2024 --no-wandb
"""
from __future__ import annotations
import sys, os
# Make the repo root importable so ``src.*`` resolves when run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
from pathlib import Path

from ultralytics import YOLO

from src.data.roboflow_pull import pull
from src.modules.register import register_all
from src.paths import CONFIGS_DIR, RUNS_DIR, WANDB_ENTITY, WANDB_PROJECT


def main():
    """Parse CLI args and train the vanilla YOLOv8l baseline end to end.

    Steps performed, in order:
      1. Register all custom modules so any custom layers referenced by the
         model config can be resolved by Ultralytics' YAML parser.
      2. Ensure the RADS dataset is present (Stage 00 cache) and locate its
         data.yaml.
      3. Build the YOLOv8l model from the baseline config and warm-start it
         from COCO-pretrained weights (transfer learning).
      4. Optionally wire up Weights & Biases logging via environment vars.
      5. Run Ultralytics training with the thesis hyperparameters, seeded
         deterministically so the run is reproducible.

    No value is returned; the trained weights and plots are written under
    RUNS_DIR/<name>/.
    """
    ap = argparse.ArgumentParser()
    # --seed: RNG seed for full reproducibility; the pipeline sweeps 42, 1337,
    #         2024 and reports mean +/- std across them. Also names the run.
    ap.add_argument("--seed", type=int, default=42)
    # --epochs: number of training epochs. Thesis spec is 100; lower it only
    #           for quick smoke tests.
    ap.add_argument("--epochs", type=int, default=100)
    # --imgsz: training/inference square image size in px. 768 is the RADS
    #          baseline (small anomalies need the extra resolution).
    ap.add_argument("--imgsz", type=int, default=768)
    # --batch: images per optimisation step. Default 32 fills an A100 80GB;
    #          drop to ~12 on smaller GPUs (the original memory-bound spec).
    ap.add_argument("--batch", type=int, default=32, help="A100 80GB friendly")
    # --lr0: initial learning rate for AdamW (the cosine schedule decays from
    #        here). 5e-4 matches Table 4.2; rescale if you change batch a lot.
    ap.add_argument("--lr0", type=float, default=5e-4)
    # --patience: early-stopping patience in epochs with no val improvement.
    ap.add_argument("--patience", type=int, default=25)
    # --name: optional run-directory name override; defaults to
    #         "baseline_seed<seed>" so seeds don't collide on disk.
    ap.add_argument("--name", default=None, help="Run name override")
    # --no-wandb: disable Weights & Biases logging (handy offline / in CI).
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # Register custom modules (CBAM, P2 head, etc.) into Ultralytics' global
    # registry. The baseline doesn't use them, but registering is harmless and
    # keeps the entry points uniform across stages.
    register_all()
    # Guarantee the dataset is available (downloads/caches via Stage 00 logic)
    # and grab the YOLO data spec that defines splits + the class list.
    dataset_path = pull()
    data_yaml = dataset_path / "data.yaml"

    # Derive a stable run name keyed on the seed unless the user overrode it.
    name = args.name or f"baseline_seed{args.seed}"
    cfg = CONFIGS_DIR / "baseline.yaml"

    # Build the architecture from the baseline YAML, then load COCO-pretrained
    # YOLOv8l weights. Transfer learning from COCO gives the detector a strong
    # feature backbone, which matters on the comparatively small RADS dataset.
    model = YOLO(str(cfg)).load("yolov8l.pt")

    # W&B init — Ultralytics auto-logs if wandb is installed and a key is set.
    # We use setdefault so any pre-exported env vars take precedence over ours.
    if not args.no_wandb:
        import os
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
        if WANDB_ENTITY:
            os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)

    # Kick off training. cos_lr + AdamW + AMP + deterministic seeding together
    # reproduce the thesis baseline; plots/save persist artifacts for later
    # evaluation (Stage 05) and cross-seed aggregation (Stage 07).
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
