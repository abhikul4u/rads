#!/usr/bin/env python
"""Stage 02 — ablation training: train any one variant in isolation or combined.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 pipeline
---------------------------------
This stage produces the *enhanced* detectors that the thesis ablation study
compares against the Stage 01 baseline. Each enhancement targets a known
weakness when detecting the small, low-contrast road anomalies in the RADS
classes ["MH", "PH", "WLPH"]:
    cbam       — adds CBAM attention in the PANet neck to focus features.
    p2         — adds a 4th (stride-4) detection head for tiny objects.
    sizeaware  — keeps the baseline architecture but swaps in a size-aware
                 bbox loss that up-weights small-box errors.
    combined   — stacks all three enhancements together (this is also the
                 teacher used by the Stage 03 distillation step).

To keep the comparison fair, every variant is trained with the *same*
hyperparameters as the baseline (epochs, imgsz, optimizer, LR schedule, etc.)
and the same seeds (42, 1337, 2024). Only the architecture/loss changes.

Variants:
    cbam       — baseline + CBAM in PANet neck
    p2         — baseline + 4th detection head (stride 4)
    sizeaware  — baseline architecture + size-aware bbox loss
    combined   — all three enhancements together

The same hyperparameters as the baseline are used for fair comparison.

Example CLI invocation
----------------------
    # Train the CBAM variant for the primary seed:
    python scripts/02_train_ablation.py --variant cbam --seed 42

    # Train the combined teacher (used later by Stage 03 distillation):
    python scripts/02_train_ablation.py --variant combined --seed 42
"""
from __future__ import annotations
import sys, os
# Make the repo root importable so ``src.*`` resolves when run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse

from ultralytics import YOLO

from src.data.roboflow_pull import pull
from src.losses.size_aware_loss import install_size_aware_loss
from src.modules.register import register_all
from src.paths import CONFIGS_DIR, RUNS_DIR, WANDB_ENTITY, WANDB_PROJECT

# Maps each variant name to the Ultralytics model-config YAML that encodes its
# architecture. The size-aware variant reuses the baseline architecture (it
# only swaps the loss), which is why "sizeaware.yaml" mirrors the baseline.
CFG_MAP = {
    "cbam": "cbam.yaml",
    "p2": "p2head.yaml",
    "sizeaware": "sizeaware.yaml",
    "combined": "combined.yaml",
}

# Variants that require size-aware loss to be installed at runtime. The
# size-aware behaviour is a loss-level monkey-patch (not part of the YAML), so
# these variants must call install_size_aware_loss() before training.
SIZE_AWARE_VARIANTS = {"sizeaware", "combined"}


def main():
    """Parse CLI args and train a single ablation variant end to end.

    Mirrors the baseline trainer (Stage 01) but selects the architecture from
    CFG_MAP based on --variant, and additionally installs the size-aware loss
    for the variants that need it before kicking off training. Keeping every
    other hyperparameter identical to the baseline is deliberate: it isolates
    the effect of each enhancement for the ablation study.

    No value is returned; weights/plots are written under RUNS_DIR/<name>/.
    """
    ap = argparse.ArgumentParser()
    # --variant (required): which enhancement to train. One of the CFG_MAP
    #           keys (cbam | combined | p2 | sizeaware). Selects the model
    #           YAML and whether the size-aware loss gets installed.
    ap.add_argument(
        "--variant", required=True, choices=sorted(CFG_MAP.keys()),
        help="Which architectural enhancement to ablate"
    )
    # --seed: RNG seed for reproducibility; swept over 42, 1337, 2024. Also
    #         used to name the run directory.
    ap.add_argument("--seed", type=int, default=42)
    # --epochs: training epoch budget. Thesis spec is 100 (kept equal to the
    #           baseline for a fair comparison).
    ap.add_argument("--epochs", type=int, default=100)
    # --imgsz: square training image size in px; 768 matches the baseline.
    ap.add_argument("--imgsz", type=int, default=768)
    # --batch: images per step. 32 suits an A100 80GB; reduce on smaller GPUs.
    ap.add_argument("--batch", type=int, default=32)
    # --lr0: initial AdamW learning rate (cosine schedule decays from here).
    ap.add_argument("--lr0", type=float, default=5e-4)
    # --patience: early-stopping patience in epochs without val improvement.
    ap.add_argument("--patience", type=int, default=25)
    # --size-aware-alpha: strength of the size-aware loss term. Only meaningful
    #           for the sizeaware/combined variants; higher alpha penalises
    #           small-box localisation errors more. 1.0 is the default weight.
    ap.add_argument("--size-aware-alpha", type=float, default=1.0)
    # --name: optional run-name override; defaults to "<variant>_seed<seed>".
    ap.add_argument("--name", default=None)
    # --no-wandb: disable Weights & Biases logging.
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # Register custom modules (e.g. CBAM, P2 head) so the variant YAMLs that
    # reference them can be parsed/instantiated by Ultralytics.
    register_all()

    # Install the size-aware bbox loss only for the variants that need it. This
    # patches the loss before any model is built, so training picks it up.
    if args.variant in SIZE_AWARE_VARIANTS:
        install_size_aware_loss(alpha=args.size_aware_alpha)

    # Ensure dataset presence (Stage 00 cache) and locate its data spec.
    dataset_path = pull()
    data_yaml = dataset_path / "data.yaml"

    # Stable per-variant, per-seed run name unless overridden.
    name = args.name or f"{args.variant}_seed{args.seed}"
    cfg = CONFIGS_DIR / CFG_MAP[args.variant]

    # CBAM/P2 change architecture — we can still warm-start the backbone from
    # COCO YOLOv8l (matching layers load, mismatched ones reinit gracefully).
    model = YOLO(str(cfg)).load("yolov8l.pt")

    # Optional W&B logging; setdefault lets externally-set env vars win.
    if not args.no_wandb:
        import os
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
        if WANDB_ENTITY:
            os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)

    # Train with the exact baseline recipe (only architecture/loss differ) so
    # the resulting metrics are directly comparable in the ablation study.
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
