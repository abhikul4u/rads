#!/usr/bin/env bash
# =============================================================================
# RADS Layer 3 — Model Development Pipeline
# scripts/06_run_full_pipeline.sh — end-to-end multi-seed pipeline orchestrator
#
# Author: Rutuja Kulkarni
#
# WHAT THIS SCRIPT DOES
#   Drives the entire Layer 3 experiment from raw dataset to thesis-ready tables
#   by calling the numbered Python stage scripts (00..07) in the correct order:
#     Stage 00  pull/prepare the dataset
#     Stage 01  train the YOLOv8 baseline, once per seed, then evaluate
#     Stage 02  train each architectural ablation (cbam, p2, sizeaware,
#               combined), once per seed, then evaluate
#     Stage 03  knowledge distillation — train a student from each seed's
#               "combined" model used as the teacher, then evaluate
#     Stage 04  quantize+export the seed-42 student to INT8 (TFLite + ONNX)
#     Stage 07  aggregate every run's metrics across seeds into thesis tables
#   The 3 detection classes are manhole / pothole / water-logged pothole.
#   A full run is all variants × 3 seeds ≈ 80–100 GPU hours on an A100 80GB.
#
# WHEN TO RUN IT
#   After runpod_setup.sh has bootstrapped the pod and the Roboflow / W&B
#   credentials have been exported. Typically launched inside a tmux session
#   (e.g. `rads-full`) with output teed to artifacts/full_run.log so progress
#   can be inspected later with status_check.sh.
#
# ENVIRONMENT VARIABLES IT HONORS
#   QUICK   "1" => smoke mode: forces EPOCHS=5 and a single seed (42) so the
#           whole flow can be exercised in minutes instead of days.
#   SEEDS   space-separated list overriding the default "42 1337 2024".
#   EPOCHS  per-training epoch budget (default 100; ignored when QUICK=1).
#   IMGSZ   training image size in px (default 768).
#   BATCH   training batch size (default 32; distillation overrides to 64).
#
# USAGE
#   bash scripts/06_run_full_pipeline.sh           # full run, all variants × 3 seeds
#   QUICK=1 bash scripts/06_run_full_pipeline.sh   # smoke test: 5 epochs, 1 seed
# =============================================================================

# Fail fast: abort on any error (-e), on unset variables (-u), and on failures
# anywhere in a pipeline (-o pipefail). A long unattended run must not silently
# continue past a broken stage.
set -euo pipefail

# Run from the repo root regardless of where the script was invoked: the script
# lives in scripts/, so its dirname + ".." is the project root that all the
# relative scripts/... and artifacts/... paths below are resolved against.
cd "$(dirname "$0")/.."

# Resolve tunables from the environment with defaults. The SEEDS array uses
# ${SEEDS:-...} so a caller can pass e.g. SEEDS="42" to run a single seed.
SEEDS=(${SEEDS:-42 1337 2024})
EPOCHS=${EPOCHS:-100}
IMGSZ=${IMGSZ:-768}
BATCH=${BATCH:-32}

# QUICK smoke mode: shrink the experiment to one seed and 5 epochs so the full
# control flow (every stage + the inter-stage file checks) can be validated fast.
if [ "${QUICK:-0}" = "1" ]; then
  SEEDS=(42)
  EPOCHS=5
  echo "==> QUICK mode: epochs=$EPOCHS, seeds=${SEEDS[*]}"
fi

# Stage 00 — dataset
# Download/prepare the dataset once up front so every subsequent training stage
# trains against identical data.
python scripts/00_pull_dataset.py

# Stage 01 — baseline × N seeds
# Train the plain YOLOv8 baseline once per seed, then immediately evaluate that
# seed's best checkpoint. Repeating across seeds gives the variance needed for
# statistically meaningful thesis comparisons.
for s in "${SEEDS[@]}"; do
  python scripts/01_train_baseline.py --seed "$s" --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH"
  # Evaluate the just-trained checkpoint; best.pt is Ultralytics' best-epoch save.
  python scripts/05_evaluate.py --weights "artifacts/runs/baseline_seed${s}/weights/best.pt" --imgsz "$IMGSZ"
done

# Stage 02 — each ablation × N seeds
# Sweep the four architectural variants, each across all seeds, evaluating after
# every training. "combined" stacks the individual improvements and doubles as
# the distillation teacher in Stage 03.
for variant in cbam p2 sizeaware combined; do
  for s in "${SEEDS[@]}"; do
    python scripts/02_train_ablation.py --variant "$variant" --seed "$s" \
        --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH"
    python scripts/05_evaluate.py \
        --weights "artifacts/runs/${variant}_seed${s}/weights/best.pt" --imgsz "$IMGSZ"
  done
done

# Stage 03 — distill student from each seed's combined teacher
# For each seed, distill a lightweight student from that seed's "combined" model.
# The `if [ -f "$TEACHER" ]` guard skips seeds whose teacher is missing (e.g. a
# crashed/partial Stage 02), so a single failure doesn't abort the whole sweep.
# Distillation uses a larger batch (64) because the student is smaller/cheaper.
for s in "${SEEDS[@]}"; do
  TEACHER="artifacts/runs/combined_seed${s}/weights/best.pt"
  if [ -f "$TEACHER" ]; then
    python scripts/03_train_distill.py --teacher "$TEACHER" --seed "$s" \
        --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch 64
    python scripts/05_evaluate.py \
        --weights "artifacts/runs/distill_seed${s}/weights/best.pt" --imgsz "$IMGSZ"
  fi
done

# Stage 04 — quantize the seed-42 student to TFLite + ONNX
# Only the seed-42 student is exported as the deployable artifact (one canonical
# model is enough for the deployment story). The guard skips this if that student
# was never produced. Export uses imgsz 640 — the deployment/edge inference size,
# deliberately smaller than the 768 training size.
STUDENT="artifacts/runs/distill_seed42/weights/best.pt"
if [ -f "$STUDENT" ]; then
  python scripts/04_quantize_export.py --weights "$STUDENT" --name rads_student_int8 --imgsz 640
fi

# Stage 07 — aggregate across seeds
# Collect every run's per-seed metrics into mean±std summary tables for the thesis.
python scripts/07_aggregate_results.py

echo "==> Pipeline complete. See artifacts/results/ for thesis tables."
