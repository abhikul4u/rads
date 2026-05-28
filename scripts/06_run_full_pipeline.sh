#!/usr/bin/env bash
# End-to-end Layer 3 pipeline driver.
# Runs: baseline → ablations → combined → distill → quantize → evaluate → aggregate.
# All variants × 3 seeds. Total ~80–100 GPU hours on A100 80GB.
#
# Usage:
#   bash scripts/06_run_full_pipeline.sh           # full run, all variants × 3 seeds
#   QUICK=1 bash scripts/06_run_full_pipeline.sh   # smoke test: 5 epochs, 1 seed
set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS=(${SEEDS:-42 1337 2024})
EPOCHS=${EPOCHS:-100}
IMGSZ=${IMGSZ:-768}
BATCH=${BATCH:-32}

if [ "${QUICK:-0}" = "1" ]; then
  SEEDS=(42)
  EPOCHS=5
  echo "==> QUICK mode: epochs=$EPOCHS, seeds=${SEEDS[*]}"
fi

# Stage 00 — dataset
python scripts/00_pull_dataset.py

# Stage 01 — baseline × N seeds
for s in "${SEEDS[@]}"; do
  python scripts/01_train_baseline.py --seed "$s" --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH"
  python scripts/05_evaluate.py --weights "artifacts/runs/baseline_seed${s}/weights/best.pt" --imgsz "$IMGSZ"
done

# Stage 02 — each ablation × N seeds
for variant in cbam p2 sizeaware combined; do
  for s in "${SEEDS[@]}"; do
    python scripts/02_train_ablation.py --variant "$variant" --seed "$s" \
        --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH"
    python scripts/05_evaluate.py \
        --weights "artifacts/runs/${variant}_seed${s}/weights/best.pt" --imgsz "$IMGSZ"
  done
done

# Stage 03 — distill student from each seed's combined teacher
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
STUDENT="artifacts/runs/distill_seed42/weights/best.pt"
if [ -f "$STUDENT" ]; then
  python scripts/04_quantize_export.py --weights "$STUDENT" --name rads_student_int8 --imgsz 640
fi

# Stage 07 — aggregate across seeds
python scripts/07_aggregate_results.py

echo "==> Pipeline complete. See artifacts/results/ for thesis tables."
