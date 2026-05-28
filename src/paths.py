"""Central path & environment resolution for RADS Layer 3.

All scripts import from here. Paths come from env vars (set by runpod_setup.sh)
with sensible local-dev fallbacks.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Core roots ---
RADS_ROOT = Path(os.environ.get("RADS_ROOT", Path(__file__).resolve().parents[2]))
RADS_ARTIFACTS = Path(os.environ.get("RADS_ARTIFACTS", RADS_ROOT / "artifacts"))
RADS_DATA = Path(os.environ.get("RADS_DATA", RADS_ROOT / "data"))

# --- Subdirs ---
CONFIGS_DIR = RADS_ROOT / "configs"
RUNS_DIR = RADS_ARTIFACTS / "runs"
EXPORTS_DIR = RADS_ARTIFACTS / "exports"
RESULTS_DIR = RADS_ARTIFACTS / "results"
CALIB_DIR = RADS_ARTIFACTS / "calibration"

# --- Roboflow ---
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "")
ROBOFLOW_PROJECT = os.environ.get("ROBOFLOW_PROJECT", "")
ROBOFLOW_VERSION = int(os.environ.get("ROBOFLOW_VERSION", "1"))

# --- W&B ---
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "rads-layer3")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

# --- Class definitions (locked from Chapter 4) ---
CLASS_NAMES = ["MH", "PH", "WLPH"]  # order is load-bearing for label files
NUM_CLASSES = len(CLASS_NAMES)

# --- Determinism ---
SEEDS = [42, 1337, 2024]

for d in (RUNS_DIR, EXPORTS_DIR, RESULTS_DIR, CALIB_DIR):
    d.mkdir(parents=True, exist_ok=True)
