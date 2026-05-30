#!/usr/bin/env bash
# =============================================================================
# RADS Layer 3 — Model Development Pipeline
# runpod_setup.sh — one-shot bootstrap for a fresh RunPod pod
#
# Author: Rutuja Kulkarni
#
# WHAT THIS SCRIPT DOES
#   Prepares a brand-new RunPod GPU pod (targeted at an A100 80GB) so that the
#   RADS (Road Anomaly Detection System) YOLOv8 training pipeline can run. It:
#     1. Installs OS-level system packages (git, media/OpenCV runtime libs,
#        tmux/htop/nvtop for monitoring).
#     2. Lays out the on-disk workspace directory tree on the persistent volume.
#     3. Installs the Python dependencies from requirements.txt.
#     4. Performs a sanity check confirming torch / Ultralytics / CUDA / GPU.
#     5. Runs a short smoke test to confirm the pipeline can actually train.
#     6. Prints a reminder of the credentials/env vars to export before training.
#
# WHEN TO RUN IT
#   Once, immediately after a fresh pod spins up (or after the container image
#   is rebuilt). It is idempotent — re-running it is safe and simply re-asserts
#   the same packages and directories.
#
# ENVIRONMENT VARIABLES IT HONORS (all optional; sensible defaults shown)
#   RADS_ROOT        Repo/checkout root          (default /workspace/rads-layer3)
#   RADS_ARTIFACTS   Output tree for runs/exports (default /workspace/artifacts)
#   RADS_DATA        Dataset download location    (default /workspace/data)
#   RADS_SKIP_SMOKE  Set to "1" to skip the smoke test in step 5.
#
# WHERE IT FITS IN THE PIPELINE
#   This is the very first step. After it succeeds (and the credentials are
#   exported as the final reminder instructs), the operator launches
#   scripts/06_run_full_pipeline.sh to perform the actual multi-seed training.
# =============================================================================

# Fail fast and loudly: -e abort on any error, -u abort on unset var,
# -o pipefail surface failures from any stage of a pipeline (not just the last).
set -euo pipefail

echo "==> RADS Layer 3 RunPod bootstrap starting"

# --- 1. System ---
# Refresh the apt package index quietly (-qq) so the install below sees current
# package versions.
apt-get update -qq
# Install the minimal set of OS packages the pipeline needs:
#   git/wget/curl/unzip — fetching code and the dataset
#   ffmpeg/libsm6/libxext6 — shared libs OpenCV (an Ultralytics dependency) needs
#   tmux/htop/nvtop — terminal multiplexer + CPU/GPU live monitors used during runs
# --no-install-recommends keeps the image lean; output is silenced for clean logs.
apt-get install -y --no-install-recommends \
    git wget curl unzip ffmpeg libsm6 libxext6 tmux htop nvtop \
    > /dev/null
echo "[ok] system packages"

# --- 2. Workspace layout on persistent volume ---
# RunPod persistent volume mounts at /workspace by default, so everything we
# want to survive a pod restart lives under it. ${VAR:-default} lets the caller
# override any path via the environment while falling back to the standard one.
export RADS_ROOT="${RADS_ROOT:-/workspace/rads-layer3}"
export RADS_ARTIFACTS="${RADS_ARTIFACTS:-/workspace/artifacts}"
export RADS_DATA="${RADS_DATA:-/workspace/data}"

# Create the artifact subtree the downstream scripts expect:
#   runs/        per-training Ultralytics output (weights, csv, plots)
#   exports/     quantized/exported models (ONNX, TFLite)
#   results/     evaluation JSONs and aggregated thesis tables
#   calibration/ calibration data used during quantization
# Brace expansion creates all four in one mkdir; -p makes parents and is a no-op
# if they already exist (this is what makes the script idempotent).
mkdir -p "$RADS_ARTIFACTS"/{runs,exports,results,calibration}
mkdir -p "$RADS_DATA"
echo "[ok] workspace dirs: $RADS_ARTIFACTS, $RADS_DATA"

# --- 3. Python deps ---
# Move into the repo directory (the folder this script lives in) so the relative
# requirements.txt path below resolves regardless of where the script is invoked.
cd "$(dirname "$0")"
# Upgrade the install toolchain first so wheels build cleanly; silence chatter.
python -m pip install --upgrade pip wheel setuptools > /dev/null
# Install the pinned project dependencies (torch, ultralytics, etc.). Left
# verbose on purpose so dependency-resolution problems are visible in the log.
python -m pip install -r requirements.txt
echo "[ok] python deps installed"

# --- 4. Sanity check ---
# Run an inline Python heredoc to print the resolved toolchain versions and,
# critically, confirm CUDA can see the GPU before we waste time on training.
# The 'PY' delimiter is single-quoted so the shell does not expand anything
# inside the heredoc — it is passed to Python verbatim.
python - <<'PY'
import torch, ultralytics, sys
print(f"  python      : {sys.version.split()[0]}")
print(f"  torch       : {torch.__version__}")
print(f"  ultralytics : {ultralytics.__version__}")
print(f"  CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
PY

# --- 5. Smoke test (skip with RADS_SKIP_SMOKE=1) ---
# A ~40s end-to-end mini training/inference run that proves the install actually
# works (not just imports). Skippable for fast re-bootstraps via RADS_SKIP_SMOKE=1.
if [ "${RADS_SKIP_SMOKE:-0}" != "1" ]; then
    echo ""
    echo "==> Running smoke test (~40s) — set RADS_SKIP_SMOKE=1 to skip"
    # The `|| { ... }` makes a smoke-test failure non-fatal: we warn but do not
    # abort, because the environment is otherwise fully set up and the operator
    # can investigate without re-running every previous step.
    python tests/smoke_test.py --quick || {
        echo "[WARN] smoke test failed — check output above. Setup is otherwise complete."
    }
fi

# --- 6. Credentials reminder ---
# The pipeline pulls the dataset from Roboflow and logs metrics to Weights &
# Biases, both of which need API keys that are NOT baked into this script for
# security. Print a copy-pasteable reminder so the operator can export them (and
# the resolved RADS_* paths) before launching training.
echo ""
echo "==> Before training, export these in your shell (or add to ~/.bashrc):"
echo "    export ROBOFLOW_API_KEY='...'"
echo "    export WANDB_API_KEY='...'"
echo "    export RADS_ROOT='$RADS_ROOT'"
echo "    export RADS_ARTIFACTS='$RADS_ARTIFACTS'"
echo "    export RADS_DATA='$RADS_DATA'"
echo ""
echo "==> Bootstrap complete."
