#!/usr/bin/env python
"""Stage 04 — INT8 PTQ + multi-format export.

Example:
    python scripts/04_quantize_export.py \
        --weights artifacts/runs/distill_seed42/weights/best.pt \
        --name rads_student_int8 \
        --imgsz 640
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
from pathlib import Path

from src.quantize.ptq_int8 import export_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calib-n", type=int, default=200)
    args = ap.parse_args()
    export_all(args.weights, args.name, args.imgsz, args.calib_n)


if __name__ == "__main__":
    main()
