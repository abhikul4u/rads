#!/usr/bin/env bash
# RunPod pod bootstrap for RADS Layer 3.
# Run once after pod spins up. Idempotent.
set -euo pipefail

echo "==> RADS Layer 3 RunPod bootstrap starting"

# --- 1. System ---
apt-get update -qq
apt-get install -y --no-install-recommends \
    git wget curl unzip ffmpeg libsm6 libxext6 tmux htop nvtop \
    > /dev/null
echo "[ok] system packages"

# --- 2. Workspace layout on persistent volume ---
# RunPod persistent volume mounts at /workspace by default.
export RADS_ROOT="${RADS_ROOT:-/workspace/rads-layer3}"
export RADS_ARTIFACTS="${RADS_ARTIFACTS:-/workspace/artifacts}"
export RADS_DATA="${RADS_DATA:-/workspace/data}"

mkdir -p "$RADS_ARTIFACTS"/{runs,exports,results,calibration}
mkdir -p "$RADS_DATA"
echo "[ok] workspace dirs: $RADS_ARTIFACTS, $RADS_DATA"

# --- 3. Python deps ---
cd "$(dirname "$0")"
python -m pip install --upgrade pip wheel setuptools > /dev/null
python -m pip install -r requirements.txt
echo "[ok] python deps installed"

# --- 4. Sanity check ---
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
if [ "${RADS_SKIP_SMOKE:-0}" != "1" ]; then
    echo ""
    echo "==> Running smoke test (~40s) — set RADS_SKIP_SMOKE=1 to skip"
    python tests/smoke_test.py --quick || {
        echo "[WARN] smoke test failed — check output above. Setup is otherwise complete."
    }
fi

# --- 6. Credentials reminder ---
echo ""
echo "==> Before training, export these in your shell (or add to ~/.bashrc):"
echo "    export ROBOFLOW_API_KEY='...'"
echo "    export WANDB_API_KEY='...'"
echo "    export RADS_ROOT='$RADS_ROOT'"
echo "    export RADS_ARTIFACTS='$RADS_ARTIFACTS'"
echo "    export RADS_DATA='$RADS_DATA'"
echo ""
echo "==> Bootstrap complete."
