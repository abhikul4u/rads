# RADS Layer 3 — Thesis Results

Snapshot of all small, thesis-critical artifacts from the full training run.

## Headline files

- **`layer3_summary.csv`** — Aggregated mean ± std across 3 seeds. This is
  Table 4.1 of the thesis (6 variants × 9 metric columns).
- **`fps_gpu.json`** — Inference latency benchmark on A100 80GB
  (baseline, combined teacher, distilled student).
- **`annotation_audit.csv`** — Per-issue audit of the Roboflow dataset
  (17 minor warnings out of 3,328 images = 0.5% error rate).
- **`preliminary_per_class.md`** — Per-class breakdown narrative (informal).

## Subdirectories

### `test_jsons/`
18 JSON files (6 variants × 3 seeds). Each is the test-set evaluation of
that run's `best.pt`. Contains: map50, map50_95, precision, recall, plus
per-class AP50 (MH, PH, WLPH).

### `training_curves/`
- `<variant>_seed<N>_results.csv` — per-epoch training metrics from
  Ultralytics. Use these to plot loss curves and val mAP over time.
- `<variant>_seed<N>_args.yaml` — the exact hyperparameters used for that
  run. Critical for reproducibility.

### `confusion_matrices/`
One confusion matrix per variant (seed 42 only). Both raw and normalized
versions where available.

### `exports/`
ONNX model files exported from the distilled student. Useful for downstream
deployment (Android via Layer 4 pipeline).

## What's NOT in here (too large for git)

- Full `best.pt` weights — see backup tarball downloaded to laptop
- All TensorBoard event files — available via W&B dashboard
- Confusion matrices for other seeds — derivable from W&B

## Reproducibility

To reproduce these numbers:
1. Clone repo, run `bash runpod_setup.sh`
2. Pull dataset: `python scripts/00_pull_dataset.py`
3. Set patience=25 (already in code), use 3 seeds: 42, 1337, 2024
4. Run full pipeline: `bash scripts/06_run_full_pipeline.sh`

Hardware: A100 80GB (RunPod Community Cloud).
Wall time: ~16-19 hours.

## W&B dashboard

Full training curves and live metrics:
https://wandb.ai/abhikul4u-hvpm-college-of-engineering-technology-amravati
