#!/usr/bin/env python
"""Stage 07 — collect per-seed evaluation results and produce the master CSV.

Also captures GPU FPS for the final student model. Mobile FPS bundle is
generated for the user to run via adb separately.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import json
from pathlib import Path

from src.eval.aggregate_seeds import aggregate
from src.eval.fps_bench import gpu_fps, write_mobile_bundle
from src.paths import EXPORTS_DIR, RESULTS_DIR, RUNS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--student-weights", type=Path,
                    default=RUNS_DIR / "distill_seed42" / "weights" / "best.pt")
    ap.add_argument("--student-tflite", type=Path,
                    default=EXPORTS_DIR / "rads_student_int8" / "rads_student_int8.int8.tflite")
    args = ap.parse_args()

    # 1. Per-seed aggregation → mean ± std
    summary_csv = RESULTS_DIR / "layer3_summary.csv"
    aggregate(args.runs_dir, summary_csv)

    # 2. GPU FPS on the local A100 (proxy for the V100 in the thesis spec)
    fps_results = {}
    for label, w in [
        ("baseline", RUNS_DIR / "baseline_seed42" / "weights" / "best.pt"),
        ("combined", RUNS_DIR / "combined_seed42" / "weights" / "best.pt"),
        ("student", args.student_weights),
    ]:
        if w.exists():
            print(f"\n[fps] {label}: measuring on GPU…")
            fps_results[label] = gpu_fps(w, imgsz=768 if label != "student" else 640)
            print(json.dumps(fps_results[label], indent=2))
        else:
            print(f"[fps] {label}: weights missing, skipped")
    (RESULTS_DIR / "fps_gpu.json").write_text(json.dumps(fps_results, indent=2))

    # 3. Mobile bundle for Snapdragon FPS
    if args.student_tflite.exists():
        bundle = write_mobile_bundle(RESULTS_DIR / "mobile_bundle", args.student_tflite)
        print(f"[ok] mobile bench bundle → {bundle}")
    else:
        print(f"[skip] no TFLite at {args.student_tflite}")

    print(f"\n==> All Layer 3 results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
