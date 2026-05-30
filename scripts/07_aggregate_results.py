#!/usr/bin/env python
"""Stage 07 — collect per-seed evaluation results and produce the master CSV.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 pipeline
---------------------------------
This is the final stage that turns the scattered per-seed run artifacts into
the reportable results for the thesis. It does three things:
  1. Aggregates the per-seed eval.json files (written by Stage 05) across the
     seeds (42, 1337, 2024) into a single master CSV with mean +/- std for
     every model/variant — the headline accuracy table.
  2. Measures GPU inference throughput (FPS) for the key models (baseline,
     combined teacher, distilled student) on the local A100, which stands in
     as a proxy for the V100 quoted in the thesis spec. The big models are
     benched at 768px and the deployable student at its 640px mobile size.
  3. Writes out a self-contained "mobile bundle" containing the INT8 TFLite
     student so the on-device (Snapdragon) FPS can be measured separately via
     adb — that step happens off this machine.

Outputs all land under RESULTS_DIR.

Also captures GPU FPS for the final student model. Mobile FPS bundle is
generated for the user to run via adb separately.

Example:
    # Use the default run/export locations:
    python scripts/07_aggregate_results.py

    # Point at a custom runs directory and student artifacts:
    python scripts/07_aggregate_results.py \
        --runs-dir artifacts/runs \
        --student-weights artifacts/runs/distill_seed42/weights/best.pt \
        --student-tflite artifacts/exports/rads_student_int8/rads_student_int8.int8.tflite
"""
from __future__ import annotations
import sys, os
# Make the repo root importable so ``src.*`` resolves when run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import json
from pathlib import Path

from src.eval.aggregate_seeds import aggregate
from src.eval.fps_bench import gpu_fps, write_mobile_bundle
from src.paths import EXPORTS_DIR, RESULTS_DIR, RUNS_DIR


def main():
    """Aggregate per-seed metrics, benchmark GPU FPS, and emit the mobile bundle.

    Produces three result artifacts under RESULTS_DIR:
      * layer3_summary.csv — cross-seed mean +/- std accuracy table.
      * fps_gpu.json       — measured GPU FPS for baseline/combined/student.
      * mobile_bundle/     — TFLite + harness for off-device mobile FPS.

    Missing inputs are skipped gracefully (with a log line) so a partial
    pipeline still produces whatever results are available.
    """
    ap = argparse.ArgumentParser()
    # --runs-dir: directory containing the per-run output folders (each holding
    #             an eval.json from Stage 05). Defaults to the pipeline RUNS_DIR.
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    # --student-weights: path to the distilled student checkpoint to GPU-bench.
    #             Defaults to the seed-42 distillation run's best.pt.
    ap.add_argument("--student-weights", type=Path,
                    default=RUNS_DIR / "distill_seed42" / "weights" / "best.pt")
    # --student-tflite: path to the INT8 TFLite export (from Stage 04) that
    #             goes into the mobile benchmark bundle. Defaults to the
    #             standard rads_student_int8 export location.
    ap.add_argument("--student-tflite", type=Path,
                    default=EXPORTS_DIR / "rads_student_int8" / "rads_student_int8.int8.tflite")
    args = ap.parse_args()

    # 1. Per-seed aggregation → mean ± std
    # Walks the run directories, reads each eval.json, and collapses the seeds
    # into one summary row per model with mean and standard deviation.
    summary_csv = RESULTS_DIR / "layer3_summary.csv"
    aggregate(args.runs_dir, summary_csv)

    # 2. GPU FPS on the local A100 (proxy for the V100 in the thesis spec)
    # Bench the three models that matter for the latency/accuracy story. The
    # student is measured at its 640px deployment size; the larger models at
    # the 768px training/eval size. Missing checkpoints are skipped, not fatal.
    fps_results = {}
    for label, w in [
        ("baseline", RUNS_DIR / "baseline_seed42" / "weights" / "best.pt"),
        ("combined", RUNS_DIR / "combined_seed42" / "weights" / "best.pt"),
        ("student", args.student_weights),
    ]:
        if w.exists():
            print(f"\n[fps] {label}: measuring on GPU…")
            # Student runs at 640 (mobile size); the heavy models at 768.
            fps_results[label] = gpu_fps(w, imgsz=768 if label != "student" else 640)
            print(json.dumps(fps_results[label], indent=2))
        else:
            print(f"[fps] {label}: weights missing, skipped")
    # Persist whatever FPS numbers we collected for the results section.
    (RESULTS_DIR / "fps_gpu.json").write_text(json.dumps(fps_results, indent=2))

    # 3. Mobile bundle for Snapdragon FPS
    # Package the INT8 TFLite model into a ready-to-run bundle; the actual
    # on-device measurement is done by the user via adb outside this script.
    if args.student_tflite.exists():
        bundle = write_mobile_bundle(RESULTS_DIR / "mobile_bundle", args.student_tflite)
        print(f"[ok] mobile bench bundle → {bundle}")
    else:
        print(f"[skip] no TFLite at {args.student_tflite}")

    print(f"\n==> All Layer 3 results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
