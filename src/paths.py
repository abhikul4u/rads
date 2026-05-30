"""Central path & environment resolution for RADS Layer 3.

Author: Rutuja Kulkarni

This module is the single source of truth for every filesystem location,
external-service credential, and project-wide constant used by the RADS
Layer 3 — Model Development Pipeline. Practically every other script in the
pipeline (data pull, training, ablations, knowledge distillation, INT8 PTQ
export, and evaluation) imports its paths and constants from here rather than
hard-coding strings, so that a single environment configuration controls the
behaviour of the whole pipeline.

Why it exists: the pipeline is designed to run identically on a RunPod A100
80GB pod (where `runpod_setup.sh` exports the relevant environment variables to
point at the persistent network volume) and on a local developer machine
(where those env vars are absent). By resolving each path from an environment
variable with a sensible local-dev fallback, the same code runs unmodified in
both places. Centralising `CLASS_NAMES` and `SEEDS` here also guarantees that
the data-loading, training, and evaluation stages all agree on the class set
and the determinism seeds used for the 3-seed ablation sweep.

All scripts import from here. Paths come from env vars (set by runpod_setup.sh)
with sensible local-dev fallbacks.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Core roots ---
# RADS_ROOT is the repository root. On RunPod it is supplied via the RADS_ROOT
# env var (pointing at the persistent volume); locally it falls back to two
# directories above this file (src/paths.py -> src -> repo root via parents[2]).
RADS_ROOT = Path(os.environ.get("RADS_ROOT", Path(__file__).resolve().parents[2]))
# RADS_ARTIFACTS holds every generated output (model runs, exports, results,
# calibration caches). Kept separate from source so it can live on a large
# persistent volume distinct from the (small, version-controlled) code tree.
RADS_ARTIFACTS = Path(os.environ.get("RADS_ARTIFACTS", RADS_ROOT / "artifacts"))
# RADS_DATA is where the Roboflow YOLOv8 dataset export is cached (see
# src/data/roboflow_pull.py). Separated from artifacts so the (immutable)
# dataset and the (regenerated) training outputs can be managed independently.
RADS_DATA = Path(os.environ.get("RADS_DATA", RADS_ROOT / "data"))

# --- Subdirs ---
# Static configs that live in the repo (custom model YAMLs, training hyper-params).
CONFIGS_DIR = RADS_ROOT / "configs"
# Ultralytics writes each training/ablation run here (weights, logs, plots).
RUNS_DIR = RADS_ARTIFACTS / "runs"
# Exported deployment artefacts: ONNX and INT8 TFLite models from the PTQ stage.
EXPORTS_DIR = RADS_ARTIFACTS / "exports"
# Aggregated evaluation outputs: mAP tables, per-class AP, confusion matrices, FPS.
RESULTS_DIR = RADS_ARTIFACTS / "results"
# Calibration image cache used by the INT8 post-training-quantisation stage.
CALIB_DIR = RADS_ARTIFACTS / "calibration"

# --- Roboflow ---
# Credentials + dataset coordinates for the source-of-truth annotated dataset.
# All are read from the environment so secrets never live in the repo; an empty
# default lets `--help` and import work even when the key is absent, with the
# actual download stage failing fast (see roboflow_pull.py) if it is unset.
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "")
ROBOFLOW_PROJECT = os.environ.get("ROBOFLOW_PROJECT", "")
# Dataset version pin — cast to int because env vars are always strings and the
# Roboflow SDK expects an integer version id. Pinning guarantees reproducibility.
ROBOFLOW_VERSION = int(os.environ.get("ROBOFLOW_VERSION", "1"))

# --- W&B ---
# Weights & Biases experiment-tracking target. ENTITY defaults to None so W&B
# falls back to the logged-in user's default entity when not explicitly set.
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "rads-layer3")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

# --- Class definitions (locked from Chapter 4) ---
# The three RADS detection classes: manhole, pothole, water-logged pothole.
# The ORDER is load-bearing: YOLO label files encode the class as an integer
# index into this list, so reordering would silently mislabel every annotation.
# roboflow_pull.py hard-fails if the downloaded dataset's class order disagrees.
CLASS_NAMES = ["MH", "PH", "WLPH"]  # order is load-bearing for label files
NUM_CLASSES = len(CLASS_NAMES)

# --- Determinism ---
# Three fixed RNG seeds. Every ablation configuration is trained once per seed so
# that reported metrics can be averaged across seeds with a variance estimate,
# isolating genuine architectural gains from run-to-run noise (thesis Ch. 4.5).
SEEDS = [42, 1337, 2024]

# Eagerly create all artefact subdirectories at import time so that any
# downstream stage can write to them without first checking for existence.
# parents=True creates intermediate dirs; exist_ok=True makes this idempotent.
for d in (RUNS_DIR, EXPORTS_DIR, RESULTS_DIR, CALIB_DIR):
    d.mkdir(parents=True, exist_ok=True)
