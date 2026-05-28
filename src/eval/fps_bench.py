"""FPS benchmark harness.

GPU side (V100 / A100 / RTX 4090): plain Ultralytics inference loop with CUDA
synchronisation around each forward pass. Reports mean FPS over `n_iters` after
a warmup phase.

Mobile side (Snapdragon 7xx): we can't run on the SoC from RunPod directly.
Instead, this module emits an `adb` command bundle the user runs on a connected
Android device with TFLite Benchmark Tool. The Chapter 4 spec calls for measuring
FPS on a Snapdragon 7-series — this is the official Google-sanctioned way.

Run modes:
    python -m src.eval.fps_bench --gpu --weights best.pt --imgsz 768
    python -m src.eval.fps_bench --mobile-bundle <out_dir> --tflite path.tflite
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np


# --- GPU side ----------------------------------------------------------------

def gpu_fps(
    weights: Path,
    imgsz: int = 768,
    n_iters: int = 200,
    n_warmup: int = 30,
    device: int = 0,
) -> dict:
    """Measure forward-pass FPS on the local CUDA device."""
    import torch
    from ultralytics import YOLO

    from src.modules.register import register_all
    register_all()

    torch.backends.cudnn.benchmark = True
    model = YOLO(str(weights)).to(f"cuda:{device}").model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz, device=f"cuda:{device}")

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    torch.cuda.synchronize()

    # Timed loop
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times_ms = [t * 1000 for t in times]
    fps = 1.0 / statistics.mean(times)
    return {
        "device": torch.cuda.get_device_name(device),
        "imgsz": imgsz,
        "n_iters": n_iters,
        "latency_ms_mean": statistics.mean(times_ms),
        "latency_ms_std": statistics.pstdev(times_ms),
        "latency_ms_p50": float(np.percentile(times_ms, 50)),
        "latency_ms_p95": float(np.percentile(times_ms, 95)),
        "fps_mean": fps,
    }


# --- Mobile bundle generator -------------------------------------------------

ADB_BENCHMARK_SCRIPT = r"""#!/usr/bin/env bash
# Run on host with a Snapdragon 7xx device connected via adb.
# Pushes the TFLite model and runs Google's official benchmark tool.
set -euo pipefail

MODEL_LOCAL="$1"
MODEL_NAME=$(basename "$MODEL_LOCAL")
DEVICE_TMP="/data/local/tmp"

# 1. Push model
adb push "$MODEL_LOCAL" "$DEVICE_TMP/$MODEL_NAME"

# 2. Push benchmark binary (download from TF release page if not present)
BENCH="benchmark_model"
if [ ! -x "$BENCH" ]; then
  echo "Download benchmark_model from:"
  echo "  https://www.tensorflow.org/lite/performance/measurement"
  exit 1
fi
adb push "$BENCH" "$DEVICE_TMP/$BENCH"
adb shell chmod +x "$DEVICE_TMP/$BENCH"

# 3. Run — CPU 4 threads
adb shell "$DEVICE_TMP/$BENCH" \
  --graph="$DEVICE_TMP/$MODEL_NAME" \
  --num_threads=4 \
  --warmup_runs=30 \
  --num_runs=200 \
  --enable_op_profiling=true \
  | tee "$(dirname "$MODEL_LOCAL")/fps_cpu4.log"

# 4. Run — GPU delegate
adb shell "$DEVICE_TMP/$BENCH" \
  --graph="$DEVICE_TMP/$MODEL_NAME" \
  --use_gpu=true \
  --warmup_runs=30 \
  --num_runs=200 \
  | tee "$(dirname "$MODEL_LOCAL")/fps_gpu.log"

# 5. Run — NNAPI (Hexagon DSP on Snapdragon)
adb shell "$DEVICE_TMP/$BENCH" \
  --graph="$DEVICE_TMP/$MODEL_NAME" \
  --use_nnapi=true \
  --warmup_runs=30 \
  --num_runs=200 \
  | tee "$(dirname "$MODEL_LOCAL")/fps_nnapi.log"

echo "Done. Logs alongside model."
"""


def write_mobile_bundle(out_dir: Path, tflite: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "run_snapdragon_fps.sh"
    script.write_text(ADB_BENCHMARK_SCRIPT)
    script.chmod(0o755)
    readme = out_dir / "README.txt"
    readme.write_text(
        "Mobile FPS benchmark bundle\n"
        "===========================\n\n"
        f"1. Plug a Snapdragon 7-series device in (USB debugging on).\n"
        f"2. Download `benchmark_model` ARM64 binary from TFLite docs.\n"
        f"3. Run:  ./run_snapdragon_fps.sh {tflite}\n\n"
        "Outputs three logs (CPU/GPU/NNAPI) next to the .tflite.\n"
    )
    return script


# --- CLI ---------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--n-iters", type=int, default=200)
    ap.add_argument("--mobile-bundle", type=Path, help="Output dir for adb script")
    ap.add_argument("--tflite", type=Path)
    args = ap.parse_args()

    if args.gpu:
        if not args.weights:
            raise SystemExit("--gpu requires --weights")
        import json
        result = gpu_fps(args.weights, args.imgsz, args.n_iters)
        print(json.dumps(result, indent=2))
    elif args.mobile_bundle:
        if not args.tflite:
            raise SystemExit("--mobile-bundle requires --tflite")
        path = write_mobile_bundle(args.mobile_bundle, args.tflite)
        print(f"[ok] bundle written: {path}")
    else:
        ap.print_help()
