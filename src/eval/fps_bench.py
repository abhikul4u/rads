"""FPS benchmark harness.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 (Model Development) pipeline
-----------------------------------------------------
Throughput (frames-per-second) is a first-class deliverable for RADS because the
detector must ultimately run on an edge/mobile device mounted in a vehicle. The
thesis (Chapter 4) reports FPS on two very different targets, and this module
covers both:

  * GPU side: the model is benchmarked directly on the RunPod CUDA device
    (A100 80GB during development, but also valid on V100 / RTX 4090). This gives
    the server-side / upper-bound latency figure.
  * Mobile side: we cannot execute on the Snapdragon SoC from inside RunPod, so
    instead of measuring here we EMIT a self-contained ``adb`` bundle. The user
    runs that bundle against a physically-connected Android phone using Google's
    official TFLite Benchmark Tool. This keeps the mobile numbers credible and
    reproducible rather than estimated.

Measurement methodology (important for the thesis): both paths use an explicit
warmup phase (to let CUDA kernels autotune / clocks settle / caches warm) before
any timed iterations, and time a fixed number of forward passes. The GPU path
brackets each forward pass with ``torch.cuda.synchronize()`` so the wall-clock
timer captures the true GPU execution time rather than just the asynchronous
kernel-launch return. We report mean/std latency plus the p50 and p95
percentiles, and derive mean FPS from the mean latency.

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
    """Measure forward-pass FPS / latency on the local CUDA device.

    Runs a synthetic single-image inference loop (batch size 1, which matches
    the real-time per-frame deployment scenario) and reports latency statistics
    plus mean throughput.

    Args:
        weights: Path to the ``.pt`` checkpoint to benchmark.
        imgsz: Square input resolution; must match how the model was trained
            (RADS uses 768 on the GPU side).
        n_iters: Number of TIMED forward passes contributing to the statistics.
        n_warmup: Number of UNTIMED warmup passes run first so cuDNN can
            autotune kernels and GPU clocks/caches stabilise — without this the
            first few iterations are artificially slow and skew the mean.
        device: CUDA device index.

    Returns:
        A dict with the device name, input size, iteration count, latency
        statistics (mean/std/p50/p95 in milliseconds) and mean FPS.

    Note:
        Imports torch / ultralytics lazily so that the mobile-bundle path (and
        ``--help``) works on machines without a CUDA stack installed.
    """
    import torch
    from ultralytics import YOLO

    # RADS ships custom modules (CBAM, P2 head, etc.); register them so the
    # checkpoint's architecture can be reconstructed before loading.
    from src.modules.register import register_all
    register_all()

    # Let cuDNN pick the fastest convolution algorithms for this fixed input
    # shape (the benchmark uses a single constant shape, so autotuning pays off).
    torch.backends.cudnn.benchmark = True
    # Pull out the raw nn.Module in eval mode (drops the Ultralytics wrapper and
    # disables dropout / BN updates) for a clean forward-pass measurement.
    model = YOLO(str(weights)).to(f"cuda:{device}").model.eval()
    # A constant random tensor is enough: we are timing compute, not accuracy,
    # so input content is irrelevant — only its shape matters.
    dummy = torch.randn(1, 3, imgsz, imgsz, device=f"cuda:{device}")

    # --- Warmup: untimed passes to stabilise kernels/clocks (see docstring) ---
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    # Block until all warmup kernels have actually finished before timing.
    torch.cuda.synchronize()

    # --- Timed loop ---
    # CUDA kernel launches are asynchronous, so we synchronise immediately
    # BEFORE starting the timer (ensure nothing is still in flight) and again
    # AFTER the forward pass (wait for it to truly complete) so that each
    # recorded interval is the genuine GPU execution time of one forward pass.
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    # Convert per-iteration seconds to milliseconds for the latency stats, and
    # derive throughput as the reciprocal of the MEAN per-frame latency.
    times_ms = [t * 1000 for t in times]
    fps = 1.0 / statistics.mean(times)
    return {
        "device": torch.cuda.get_device_name(device),
        "imgsz": imgsz,
        "n_iters": n_iters,
        "latency_ms_mean": statistics.mean(times_ms),
        # Population std (pstdev) because these samples ARE the full set of
        # timed iterations, not a sample drawn to estimate a wider population.
        "latency_ms_std": statistics.pstdev(times_ms),
        # p50/p95 percentiles characterise the latency distribution's tail,
        # which matters more than the mean for real-time guarantees.
        "latency_ms_p50": float(np.percentile(times_ms, 50)),
        "latency_ms_p95": float(np.percentile(times_ms, 95)),
        "fps_mean": fps,
    }


# --- Mobile bundle generator -------------------------------------------------

# Template for the shell script we drop into the mobile bundle. It is a complete
# adb-driven benchmark: push the model + Google's `benchmark_model` binary to the
# phone, then run it under three different accelerators (CPU, GPU delegate, NNAPI
# /Hexagon DSP) so the thesis can compare backends. Kept as a raw string so the
# bash `$VAR` references survive untouched into the emitted file.
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
    """Emit a self-contained adb benchmark bundle for the Snapdragon target.

    Because the mobile FPS measurement happens off-box (on a physical phone),
    this writes out the runnable shell script plus a short human-readable README
    explaining how to use it. The script itself is the ``ADB_BENCHMARK_SCRIPT``
    template above.

    Args:
        out_dir: Directory to create the bundle in (created if missing).
        tflite: Path to the INT8 ``.tflite`` model, surfaced in the README as
            the example argument to the script.

    Returns:
        The path to the written, executable shell script.

    Side effects:
        Creates ``out_dir``, writes ``run_snapdragon_fps.sh`` (chmod 0755) and
        ``README.txt``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "run_snapdragon_fps.sh"
    script.write_text(ADB_BENCHMARK_SCRIPT)
    # Make the emitted script directly executable on the host.
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
    # Two mutually-exclusive modes selected by flag: --gpu benchmarks here and
    # now, --mobile-bundle emits the offline adb bundle. With neither, print help.
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")              # run GPU benchmark
    ap.add_argument("--weights", type=Path)                    # .pt for GPU mode
    ap.add_argument("--imgsz", type=int, default=768)          # GPU input size
    ap.add_argument("--n-iters", type=int, default=200)        # timed iterations
    ap.add_argument("--mobile-bundle", type=Path, help="Output dir for adb script")
    ap.add_argument("--tflite", type=Path)                     # .tflite for mobile mode
    args = ap.parse_args()

    if args.gpu:
        # GPU mode needs a checkpoint to benchmark.
        if not args.weights:
            raise SystemExit("--gpu requires --weights")
        import json
        result = gpu_fps(args.weights, args.imgsz, args.n_iters)
        # Emit machine-readable JSON so results can be captured into the thesis.
        print(json.dumps(result, indent=2))
    elif args.mobile_bundle:
        # Mobile mode needs the model that will be pushed to the phone.
        if not args.tflite:
            raise SystemExit("--mobile-bundle requires --tflite")
        path = write_mobile_bundle(args.mobile_bundle, args.tflite)
        print(f"[ok] bundle written: {path}")
    else:
        # No mode chosen — show usage rather than doing nothing silently.
        ap.print_help()
