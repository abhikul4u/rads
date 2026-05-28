"""Evaluation helpers built on top of Ultralytics' validator.

`run_eval(weights, data_yaml)` returns a flat dict with:
    map50, map50_95, precision, recall,
    per_class: {MH: AP50, PH: AP50, WLPH: AP50},
    confusion_matrix_path: <png>,
    val_dir: <ultralytics val output dir>

Also exposes `collect_runs(runs_dir)` to gather all runs into a single dataframe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.modules.register import register_all
from src.paths import CLASS_NAMES, RESULTS_DIR


def run_eval(
    weights: Path,
    data_yaml: Path,
    imgsz: int = 768,
    split: str = "test",
    run_name: str | None = None,
) -> Dict:
    """Run validation on `split` and return a flat metrics dict."""
    register_all()
    model = YOLO(str(weights))
    res = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        split=split,
        plots=True,
        save_json=True,
        name=run_name or f"eval_{weights.stem}",
    )

    # `res.box` exposes per-class AP arrays in Ultralytics 8.3+
    p_arr = np.asarray(res.box.mp if np.ndim(res.box.mp) > 0 else [res.box.mp]).reshape(-1)
    r_arr = np.asarray(res.box.mr if np.ndim(res.box.mr) > 0 else [res.box.mr]).reshape(-1)

    out = {
        "map50": float(res.box.map50),
        "map50_95": float(res.box.map),
        "precision": float(p_arr.mean()) if p_arr.size else float("nan"),
        "recall": float(r_arr.mean()) if r_arr.size else float("nan"),
    }

    # Per-class AP50 — `res.box.ap50` is shape (nc,)
    ap50 = np.asarray(res.box.ap50)
    for i, name in enumerate(CLASS_NAMES):
        out[f"AP50_{name}"] = float(ap50[i]) if i < len(ap50) else float("nan")

    out["val_dir"] = str(res.save_dir)
    cm_png = Path(res.save_dir) / "confusion_matrix.png"
    if cm_png.exists():
        out["confusion_matrix_path"] = str(cm_png)
    return out


def collect_runs(runs_dir: Path) -> pd.DataFrame:
    """Gather all `results.csv` files under `runs_dir` into one DataFrame."""
    rows: List[Dict] = []
    for csv in runs_dir.rglob("results.csv"):
        try:
            df = pd.read_csv(csv)
            last = df.iloc[-1].to_dict()
            last["run"] = csv.parent.name
            last["path"] = str(csv.parent)
            rows.append(last)
        except Exception as e:
            print(f"[warn] failed to read {csv}: {e}")
    return pd.DataFrame(rows)


def save_results(metrics: Dict, name: str) -> Path:
    out = RESULTS_DIR / f"{name}.json"
    import json
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[ok] metrics → {out}")
    return out
