# RunPod Operations Runbook

Single source of truth for setting up, running, and recovering the RADS Layer 3 pipeline on RunPod. Updated as we learn things the hard way.

**Last updated**: 2026-05-28 (after first QUICK validation run)

---

## Quick reference — most-used commands

```bash
# Connect to running training
tmux attach -t rads

# Detach without killing training
# Press: Ctrl+B then D

# Check if anything is running
tmux ls
ps aux | grep -E "(python|train)" | grep -v grep

# Tail the live training log without disturbing the job
tail -f /workspace/artifacts/quick_run.log
# Ctrl+C exits tail; does NOT stop training

# Print all test-set results in a table
for f in /workspace/artifacts/results/*_test.json; do
  name=$(basename "$f" .json)
  python -c "import json; d=json.load(open('$f')); print(f\"{'$name':25s}  mAP50={d['map50']:.4f}  mAP50-95={d['map50_95']:.4f}\")"
done

# Check disk usage (50GB volume)
df -h /workspace

# Check GPU utilization
nvidia-smi
```

---

## Section 1 — First-time pod setup

### 1.1 Create pod on RunPod
- GPU: **A100 80GB PCIe or SXM**
- Template: **`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`**
- Pod name: descriptive (e.g. `rads-training`)
- **Volume Disk: 50 GB**
- Container Disk: 20 GB (default)
- Volume Mount Path: `/workspace`
- GPU Count: 1
- Deploy On-Demand

### 1.2 Connect
Use **Open web terminal** from the Connect panel.

### 1.3 Verify environment
```bash
nvidia-smi              # A100 80GB
python --version        # 3.11.x
nvcc --version | tail -1
```

### 1.4 Clone the repo (private repo needs PAT from github.com/settings/tokens, `repo` scope)
```bash
cd /workspace
git clone https://github.com/abhikul4u/rads.git
cd rads
```

### 1.5 Run the bootstrap
```bash
bash runpod_setup.sh
```

Expected ending: `✓ 9 passed, ○ 1 skipped`.

### 1.6 Patch known issues (until upstream fixes)

```bash
# wandb upgrade (old version rejects new key format)
pip install --upgrade wandb

# Pipeline absolute paths (verify the FIXED commit has these)
grep "artifacts/runs/" scripts/06_run_full_pipeline.sh
# All paths should start with /workspace/artifacts/runs/
# If not:
sed -i 's|artifacts/runs/|/workspace/artifacts/runs/|g' scripts/06_run_full_pipeline.sh
sed -i 's|artifacts/exports/|/workspace/artifacts/exports/|g' scripts/06_run_full_pipeline.sh

# W&B enable in Ultralytics
yolo settings wandb=True
```

### 1.7 Credentials in `~/.bashrc`

```bash
cat >> ~/.bashrc << 'EOF'
export ROBOFLOW_API_KEY='paste_actual_key'
export ROBOFLOW_WORKSPACE='road-anomalies'
export ROBOFLOW_PROJECT='road_anamolies_yolov8-phref'
export ROBOFLOW_VERSION=6
export WANDB_API_KEY='paste_actual_key'
export WANDB_PROJECT='rads-layer3'
export RADS_ROOT='/workspace/rads'
export RADS_ARTIFACTS='/workspace/artifacts'
export RADS_DATA='/workspace/data'
EOF
source ~/.bashrc
```

⚠️ NEVER paste API keys into chat. If exposed, revoke and regenerate immediately.

### 1.8 Login to W&B
```bash
wandb login --relogin
# Paste WANDB_API_KEY when prompted
```

### 1.9 Pull the dataset
```bash
python scripts/00_pull_dataset.py
```

Expected: `[ok] dataset at /workspace/data/road_anamolies_yolov8-phref-v6`

---

## Section 2 — Running the pipeline

### 2.1 ALWAYS use tmux
```bash
tmux new -s rads                # create
# Ctrl+B then D                 # detach
tmux attach -t rads             # reattach
tmux kill-session -t rads       # kill (only when SURE)
```

### 2.2 QUICK validation (~30 min, ~$1) — 5 epochs × 1 seed
```bash
tmux new -s rads
cd /workspace/rads
QUICK=1 bash scripts/06_run_full_pipeline.sh 2>&1 | tee /workspace/artifacts/quick_run.log
# Ctrl+B D
```

### 2.3 Full production (~80 hr, ~$140-160) — 100 epochs × 3 seeds
```bash
tmux new -s rads-full
cd /workspace/rads
bash scripts/06_run_full_pipeline.sh 2>&1 | tee /workspace/artifacts/full_run.log
```

### 2.4 Single-variant retry
```bash
python scripts/02_train_ablation.py --variant cbam --seed 42
python scripts/05_evaluate.py --weights /workspace/artifacts/runs/cbam_seed42/weights/best.pt --imgsz 768
```

---

## Section 3 — Monitoring

- **W&B**: <https://wandb.ai/abhikul4u-hvpm-college-of-engineering-technology-amravati> (best)
- **Log tail**: `tail -f /workspace/artifacts/quick_run.log` (Ctrl+C is safe)
- **Reattach tmux**: `tmux attach -t rads` then `Ctrl+B D`
- **GPU**: `nvidia-smi` or `watch -n 2 nvidia-smi`
- **Disk**: `df -h /workspace`, `du -sh /workspace/artifacts/*`

---

## Section 4 — Recovery scenarios

### 4.1 "Connection closed" in web terminal
Open new web terminal → `tmux attach -t rads`. Training survived.

### 4.2 Pod restarted (`tmux ls` → "no server running")
```bash
ls /workspace/artifacts/runs/    # see what completed
# Resume from the missing variant:
python scripts/02_train_ablation.py --variant <missing> --seed 42
```

### 4.3 Pipeline crashed mid-run
```bash
tail -100 /workspace/artifacts/quick_run.log
```
Triage:
- OOM → `--batch 16`
- Roboflow auth expired → rerun `00_pull_dataset.py`
- Disk full → clean old runs

### 4.4 W&B not logging
```bash
yolo settings | grep wandb       # must show: wandb: True
yolo settings wandb=True         # if not
# Restart training — old runs can't be retroactively logged
```

### 4.5 Smoke test passes locally, fails on pod
Usually a file missing from git. Check `.gitignore` — `data/` matches `src/data/` too. Use `/data/` to anchor to root.

### 4.6 Distill "NoneType is not iterable"
Fixed in commit (see Section 11 iteration log). If you still see it:
```python
# In src/distill/trainer.py at top of __init__:
if cfg is None:
    from ultralytics.cfg import DEFAULT_CFG
    cfg = DEFAULT_CFG
```

### 4.7 Deterministic warnings on CBAM
`adaptive_max_pool2d_backward_cuda` lacks deterministic impl. Safe to ignore — results reproducible within ±0.001 mAP.

---

## Section 5 — Cost control

| State | GPU cost | Storage cost |
|---|---|---|
| Running | $1.74/hr | included |
| **Stopped** | $0 | ~$5/mo for 50GB |
| Terminated | $0 | $0 (data lost) |

**Stop the pod** between sessions. Don't terminate until you've downloaded:
- `/workspace/artifacts/results/*.csv`
- `/workspace/artifacts/runs/*/weights/best.pt`
- `/workspace/artifacts/exports/*`

Budget: ~$1.45 per variant×seed. Full run ~$40-50 expected (vs $160 original estimate).

---

## Section 6 — Troubleshooting reference

| Symptom | Fix |
|---|---|
| `nvidia-smi` no GPU | Stop + restart pod |
| `git pull` asks for password | PAT expired — generate new at github.com/settings/tokens |
| `wandb: API key must be 40 chars` | `pip install --upgrade wandb` |
| `ModuleNotFoundError: src.X` | File not in git — check `.gitignore` |
| `CUDA out of memory` | `--batch 16` |
| `tmux ls` "no server" | Pod restarted — resume from `ls /workspace/artifacts/runs/` |
| Eval can't find `best.pt` | See Bug 2 in Section 1.6 |
| W&B not logging | `yolo settings wandb=True` |
| Distill `NoneType not iterable` | See 4.6 |
| Deterministic warnings | Safe to ignore (4.7) |

---

## Section 7 — Safe iteration

```
1. Edit locally in VS Code
2. python tests/smoke_test.py    (should be 10/10)
3. git commit + push
4. On pod: git pull
5. python tests/smoke_test.py    (should be 10/10 here too)
6. QUICK=1 bash scripts/06_run_full_pipeline.sh
```

---

## Section 8 — Pending TODOs

- [ ] `requirements.txt` pin `wandb>=0.20`
- [ ] Smoke test: add a 1-epoch train+eval integration check
- [ ] `runpod_setup.sh`: auto-run `yolo settings wandb=True`, `pip install --upgrade wandb`
- [x] DistillationTrainer cfg=DEFAULT_CFG default
- [x] Pipeline paths absolute
- [x] Size-aware clamp 10x → 4x
- [ ] Auto-skip already-completed variants on pipeline rerun
- [ ] CleanLab label-error mining integrated into audit

---

## Section 9 — Useful one-liners

```bash
# Comparison table of all test results
for f in /workspace/artifacts/results/*_test.json; do
  name=$(basename "$f" .json)
  python -c "
import json
d = json.load(open('$f'))
print(f\"  {'$name':25s}  mAP50={d['map50']:.4f}  mAP50-95={d['map50_95']:.4f}  P={d.get('precision', 0):.3f}  R={d.get('recall', 0):.3f}\")
"
done

# Disk usage by run
du -sh /workspace/artifacts/runs/* | sort -h

# Find large files (cleanup candidates)
find /workspace -size +500M 2>/dev/null

# Env vars without revealing values
env | grep -E "(ROBOFLOW|WANDB|RADS)" | sed 's/=.*/=<hidden>/'

# Watch live training without blocking
watch -n 5 "tail -20 /workspace/artifacts/quick_run.log"
```

---

## Section 10 — Dataset quality auditing

### 10.1 Automated audit (30s)
```bash
cd /workspace/rads
python tools/audit_annotations.py \
  --dataset /workspace/data/road_anamolies_yolov8-phref-v6 \
  --export /workspace/artifacts/results/annotation_audit.csv
```

Catches: bad class IDs, OOB coordinates, tiny/huge boxes, duplicates, missing labels, class imbalance.

### 10.2 Auto-fix safe issues
```bash
python tools/audit_annotations.py \
  --dataset /workspace/data/road_anamolies_yolov8-phref-v6 \
  --fix-easy
```

Safe-fixes: clip OOB coords, remove duplicates, drop zero-size boxes. Backups in `.bak` files.

Revert if needed:
```bash
find /workspace/data -name "*.txt.bak" -exec sh -c 'mv "$1" "${1%.bak}"' _ {} \;
```

### 10.3 Visual spot-check (10 min)
```bash
mkdir -p /workspace/artifacts/audit_samples
python -c "
import cv2, random
from pathlib import Path
random.seed(42)
ds = Path('/workspace/data/road_anamolies_yolov8-phref-v6')
names = {0: 'MH', 1: 'PH', 2: 'WLPH'}
colors = {0: (0,255,255), 1: (0,255,0), 2: (255,0,0)}
out = Path('/workspace/artifacts/audit_samples')
images = list((ds / 'train' / 'images').iterdir())
for p in random.sample(images, 50):
    im = cv2.imread(str(p))
    if im is None: continue
    h, w = im.shape[:2]
    lbl = ds / 'train' / 'labels' / (p.stem + '.txt')
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            if not line.strip(): continue
            cls_id, cx, cy, bw, bh = line.split()
            cls_id = int(cls_id)
            cx, cy, bw, bh = float(cx)*w, float(cy)*h, float(bw)*w, float(bh)*h
            x1, y1 = int(cx-bw/2), int(cy-bh/2)
            x2, y2 = int(cx+bw/2), int(cy+bh/2)
            cv2.rectangle(im, (x1,y1), (x2,y2), colors[cls_id], 2)
            cv2.putText(im, names[cls_id], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[cls_id], 2)
    cv2.imwrite(str(out / p.name), im)
"
```

Then browse `/workspace/artifacts/audit_samples/` in Jupyter Lab.

### 10.4 Model-based error mining (after first decent baseline)
```python
from ultralytics import YOLO
model = YOLO('/workspace/artifacts/runs/baseline_seed42/weights/best.pt')
results = model.predict(
    '/workspace/data/road_anamolies_yolov8-phref-v6/train/images',
    conf=0.7, save_txt=True, save_conf=True,
)
```

Compare predictions to labels. Strong disagreement = candidate label error. Automate with [CleanLab](https://docs.cleanlab.ai/).

---

## Section 11 — Iteration log

Track each QUICK run's results.

### Run 1 — initial QUICK (2026-05-28)

**Setup**: out-of-the-box code, alpha=1.0, clamp=10x

| Variant | mAP50 | mAP50-95 | Notes |
|---|---|---|---|
| baseline | 0.704 | 0.422 | strong |
| cbam | 0.679 | 0.375 | -0.025 (expected at 5 epochs) |
| p2 | 0.685 | 0.387 | -0.019 |
| sizeaware | 0.604 | 0.339 | -0.100 ← clamp too aggressive |
| combined | 0.408 | 0.229 | -0.296 ← compounds sizeaware |
| distill | — | — | crashed (cfg=None) |

**Bugs found**:
1. ✓ Path issue in 06_run_full_pipeline.sh (relative → absolute) — fixed
2. ✗ DistillationTrainer crashes with cfg=None — fix queued
3. ✗ Size-aware clamp at 10x too aggressive — fix queued
4. ✓ W&B disabled by Ultralytics default — fixed via `yolo settings wandb=True`

### Run 2 — after distill + size-aware fixes (TBD)
Pending.

---

## Section 12 — Bug-fix iteration cycle

```
1. Find bug in QUICK output
2. Patch on laptop (VS Code)
3. python tests/smoke_test.py        # local sanity
4. git add + commit + push
5. On pod: git pull
6. Clean partial runs: rm -rf /workspace/artifacts/runs/* /workspace/artifacts/exports/*
7. tmux new -s rads
8. QUICK=1 bash scripts/06_run_full_pipeline.sh 2>&1 | tee /workspace/artifacts/quick_runN.log
9. Detach (Ctrl+B D)
10. ~30 min later: triage results, repeat
```

Each cycle: ~$1, ~30 min. Iterate freely.

---

*Version-controlled in the RADS Layer 3 repo. Push edits to GitHub; pull on the pod.*