#!/usr/bin/env python
"""Stage 04 — INT8 PTQ + multi-format export.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 pipeline
---------------------------------
This stage takes the trained (usually distilled YOLOv8n) student and makes it
deployable. It applies INT8 *post-training quantization* (PTQ) — calibrating
activation ranges on a sample of real RADS images so the float32 weights can
be represented as 8-bit integers — and then exports the model to the formats
the rest of the system consumes: ONNX (for server/GPU runtimes) and TFLite
(for the on-device mobile benchmark in Stage 07). INT8 quantization is what
shrinks the model and speeds up CPU/NPU inference enough for real-time road
anomaly detection on a phone, ideally with minimal accuracy loss.

All conversion logic lives in ``src.quantize.ptq_int8.export_all``; this
script is just the thin CLI that feeds it.

Example:
    python scripts/04_quantize_export.py \
        --weights artifacts/runs/distill_seed42/weights/best.pt \
        --name rads_student_int8 \
        --imgsz 640
"""
from __future__ import annotations
import sys, os
# Make the repo root importable so ``src.*`` resolves when run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
from pathlib import Path

from src.quantize.ptq_int8 import export_all


def main():
    """Parse CLI args and run INT8 PTQ + ONNX/TFLite export.

    Thin wrapper: it forwards the four CLI values to ``export_all``, which
    performs calibration, quantization, and writes out the exported model
    files under the exports directory keyed by ``--name``.
    """
    ap = argparse.ArgumentParser()
    # --weights (required): path to the trained checkpoint to quantize,
    #           typically the distilled student's best.pt from Stage 03.
    ap.add_argument("--weights", required=True, type=Path)
    # --name (required): output bundle name; used to label the export folder
    #           and the resulting .onnx/.tflite files (e.g. rads_student_int8).
    ap.add_argument("--name", required=True)
    # --imgsz: square input size baked into the exported model. 640 is the
    #          mobile inference size (smaller than the 768 used for training).
    ap.add_argument("--imgsz", type=int, default=640)
    # --calib-n: number of calibration images used to estimate INT8 activation
    #            ranges. More images give more stable ranges; 200 is a good
    #            speed/accuracy trade-off for PTQ.
    ap.add_argument("--calib-n", type=int, default=200)
    args = ap.parse_args()
    # Hand off to the quantization/export pipeline.
    export_all(args.weights, args.name, args.imgsz, args.calib_n)


if __name__ == "__main__":
    main()
