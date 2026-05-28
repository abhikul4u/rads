# RADS Layer 3 — Model Development Pipeline

End-to-end pipeline for the **Road Anomaly Detection System (RADS)** Layer 3 research contribution, optimised for RunPod A100 80GB.

Implements the eight deliverables from Chapter 4:
1. **Baseline** — YOLOv8l on COCO weights, 100ep, 768px, AdamW + cosine LR
2. **CBAM** attention in PANet neck (~0.3M params)
3. **P2 head** — 4th detection head at stride 4 (192×192 grid at 768 input)
4. **Size-aware loss** — inverse-√area weighted bbox+DFL
5. **Ablations** — each enhancement in isolation + combined, all × 3 seeds
6. **Knowledge distillation** — combined teacher → YOLOv8n student
   (0.4 task + 0.4 KL@T=4 + 0.2 feature MSE)
7. **INT8 PTQ** → ONNX (cross-platform) + TFLite (Android)
8. **Evaluation** — mAP, per-class AP, P/R, confusion matrix, GPU + Snapdragon FPS

---

## Quickstart on RunPod

```bash
# 1. Launch an A100 80GB pod with the official PyTorch 2.4 + CUDA 12.1 template.
#    Mount a persistent volume at /workspace.

# 2. Clone this repo into the volume so it survives pod restarts.
cd /workspace
git clone <your-repo> rads-layer3
cd rads-layer3

# 3. One-shot bootstrap (system deps + Python deps + sanity check).
bash runpod_setup.sh

# 4. Export credentials.
export ROBOFLOW_API_KEY='...'
export ROBOFLOW_WORKSPACE='your-workspace'
export ROBOFLOW_PROJECT='your-project-name'
export ROBOFLOW_VERSION=1
export WANDB_API_KEY='...'
export WANDB_PROJECT='rads-layer3'

# 5. Smoke test (5 epochs, 1 seed — ~30 min).
QUICK=1 bash scripts/06_run_full_pipeline.sh

# 6. Full run (~80 GPU-hours, ~$150 on A100 80GB).
bash scripts/06_run_full_pipeline.sh
```

After the run, thesis-ready outputs land in `artifacts/`:
```
artifacts/
├── runs/<variant>_seed<N>/         # Ultralytics training output + eval.json
├── exports/rads_student_int8/      # *.onnx, *.int8.onnx, *.int8.tflite
└── results/
    ├── layer3_summary.csv          # mean ± std across seeds — straight into thesis
    ├── fps_gpu.json                # latency + FPS for baseline / combined / student
    └── mobile_bundle/              # adb script for Snapdragon FPS
```

---

## Per-stage CLI

You can run each stage independently — useful for iterating on one enhancement or re-running just the distillation.

```bash
# Pull dataset (cached on volume after first run)
python scripts/00_pull_dataset.py [--force]

# Baseline / ablations / combined — same hyperparameters per Table 4.2
python scripts/01_train_baseline.py --seed 42
python scripts/02_train_ablation.py --variant cbam --seed 42
python scripts/02_train_ablation.py --variant p2 --seed 42
python scripts/02_train_ablation.py --variant sizeaware --seed 42
python scripts/02_train_ablation.py --variant combined --seed 42

# Distill student from the combined teacher
python scripts/03_train_distill.py \
    --teacher artifacts/runs/combined_seed42/weights/best.pt --seed 42

# Quantize + multi-format export
python scripts/04_quantize_export.py \
    --weights artifacts/runs/distill_seed42/weights/best.pt \
    --name rads_student_int8 --imgsz 640

# Evaluate any checkpoint on the test split
python scripts/05_evaluate.py \
    --weights artifacts/runs/combined_seed42/weights/best.pt --imgsz 768

# Aggregate across seeds + measure GPU FPS + generate mobile bundle
python scripts/07_aggregate_results.py
```

---

## Notebooks

Interactive companions for inspection — run inside JupyterLab on the pod:

| Notebook                              | Purpose                                              |
| ------------------------------------- | ---------------------------------------------------- |
| `01_dataset_inspection.ipynb`         | Class counts per split, sample images, sanity checks |
| `02_ablation_analysis.ipynb`          | Loads `layer3_summary.csv`, plots bar charts         |
| `03_distillation_inspection.ipynb`    | Teacher vs student side-by-side predictions + FPS    |
| `04_quantization_validation.ipynb`    | ONNX/TFLite output-shape + sanity inference          |

---

## Smoke test (regression catcher)

After tweaking a YAML, a loss, or the parser patch, run the smoke test before kicking off a multi-hour training run:

```bash
make smoke              # full suite, ~40s on CPU, faster on GPU
make smoke-quick        # skips ONNX export, ~35s
make smoke-quickest     # skips ONNX + training step, ~30s
```

What it verifies in under a minute:
1. All 6 YAML configs parse, build via `YOLO(cfg)`, and forward-pass
2. Param counts stay within ±5% of nominal (catches accidental scale changes)
3. CBAM preserves shape and stays under the 0.5M param budget
4. `register_all()` is idempotent (double-call doesn't double-patch)
5. Size-aware loss installs/uninstalls cleanly and computes finite values
6. Composite distillation loss: forward + backward produces gradients
7. One AdamW micro-step on the student updates parameters
8. ONNX export of the student succeeds (catches export-graph regressions)
9. All 7 scripts respond to `--help` (catches import errors)

Also runs automatically on every push via `.github/workflows/smoke.yml`.

**Verified failure modes** — the suite catches all of these in <1 minute:
- Reverting CBAM channel counts in YAML
- Bumping `depth_multiple` or `width_multiple` past the ±5% band
- Breaking the parser patch
- Adding a non-finite term to the loss

---

## Architecture decisions (and why)

**Custom YAML + monkey-patch over forking Ultralytics.**
CBAM is registered into Ultralytics' parser namespace; the P2 head is a pure YAML config; size-aware loss is a `BboxLoss` subclass installed via a one-line monkey-patch. Result: every enhancement is a self-contained, reversible toggle, and we can bump Ultralytics versions without merge pain.

**Class order locked at parse time.**
`src/data/roboflow_pull.py` hard-fails if Roboflow's export ever swaps the class order vs `CLASS_NAMES = ["MH", "PH", "WLPH"]`. This protects months of trained weights from a single misclick in Roboflow's UI.

**Knowledge-distillation trainer hooks rather than custom forward.**
`DistillationTrainer` subclasses `DetectionTrainer` and intercepts only `criterion()`, using forward hooks on the Detect heads of both teacher and student. The student's training loop, optimiser, scheduler, augmentation pipeline, and W&B logging all keep working unchanged.

**INT8 via Ultralytics' official export path.**
`model.export(format='tflite', int8=True)` uses the maintained `onnx2tf` chain. Hand-rolling TF SavedModel conversion is the #1 source of "works on my machine" bugs in YOLO PTQ.

**3 seeds, not 5.**
Variance on Indian road anomaly detection at this dataset size is well-bounded by 3 seeds (Chapter 4.4.3); 5 seeds doubles cost without changing the conclusion.

---

## Reproducing thesis Table 4.X

```bash
# Full deterministic run, 3 seeds × 6 variants × 100 epochs.
SEEDS="42 1337 2024" bash scripts/06_run_full_pipeline.sh

# Result: artifacts/results/layer3_summary.csv → paste into thesis as Table 4.X.
```

The mobile FPS column requires a Snapdragon 7-series device:
```bash
# From a host with the device on adb:
bash artifacts/results/mobile_bundle/run_snapdragon_fps.sh \
     artifacts/exports/rads_student_int8/rads_student_int8.int8.tflite
```

---

## Cost & runtime (A100 80GB on RunPod)

| Stage              | GPU hours | Cost @ $1.89/hr |
| ------------------ | --------: | --------------: |
| Baseline × 3 seeds |        24 |           $45   |
| 4 ablations × 3 seeds | 50     |           $94   |
| Distillation × 3 seeds | 10    |           $19   |
| Quantization + eval |       2 |            $4   |
| **Total**          |   **86**  |       **$162**  |
