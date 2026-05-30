"""Evaluation helpers built on top of Ultralytics' validator.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 (Model Development) pipeline
-----------------------------------------------------
This module is the single source of truth for the detection accuracy metrics
reported in the thesis. Rather than re-implementing mAP / precision / recall
(which is notoriously error-prone), it delegates the heavy lifting to
Ultralytics' built-in validator (``model.val(...)``) and then flattens the
returned ``Results`` object into a plain, JSON-serialisable dict using our own
canonical key names (``map50``, ``map50_95``, ``precision``, ``recall`` and the
per-class ``AP50_MH`` / ``AP50_PH`` / ``AP50_WLPH``). Those flat dicts are what
``src.eval.aggregate_seeds`` later averages across the three seeds.

It also asks Ultralytics to render plots (confusion matrix, PR curves) and dump
COCO-style JSON, and records the paths to those artifacts so downstream tooling
and the thesis write-up can locate them. ``collect_runs`` and ``save_results``
are small convenience helpers for harvesting / persisting these metrics.

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
    """Run validation on ``split`` and return a flat metrics dict.

    Loads the checkpoint, runs Ultralytics' validator on the requested data
    split, then extracts the headline detection metrics and per-class AP@0.50
    into a flat dict that is easy to JSON-serialise and later aggregate.

    Args:
        weights: Path to the ``.pt`` checkpoint to evaluate.
        data_yaml: Ultralytics dataset YAML describing class names and splits.
        imgsz: Square evaluation resolution (RADS uses 768).
        split: Which split to evaluate — defaults to ``"test"`` so the reported
            numbers reflect held-out generalisation, not the val set used during
            training.
        run_name: Name for the Ultralytics output sub-directory; defaults to
            ``eval_<weights-stem>``.

    Returns:
        Flat dict of metrics (see module docstring) plus the paths to the
        validation output directory and confusion-matrix PNG.

    Side effects:
        Runs a full inference pass over the split and writes Ultralytics'
        plots + JSON predictions under its run directory.
    """
    # Register RADS custom modules so a custom-architecture checkpoint loads.
    register_all()
    model = YOLO(str(weights))
    # plots=True renders the confusion matrix / PR curves; save_json=True dumps
    # COCO-format predictions for any external re-scoring.
    res = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        split=split,
        plots=True,
        save_json=True,
        name=run_name or f"eval_{weights.stem}",
    )

    # `res.box` exposes per-class AP arrays in Ultralytics 8.3+.
    # mp/mr (mean precision/recall) may come back as a scalar (single class) or
    # an array (multi-class); wrap scalars in a list and flatten so the
    # downstream .mean() works uniformly in both cases.
    p_arr = np.asarray(res.box.mp if np.ndim(res.box.mp) > 0 else [res.box.mp]).reshape(-1)
    r_arr = np.asarray(res.box.mr if np.ndim(res.box.mr) > 0 else [res.box.mr]).reshape(-1)

    out = {
        # mAP@0.50 and the stricter COCO mAP@0.50:0.95 (res.box.map).
        "map50": float(res.box.map50),
        "map50_95": float(res.box.map),
        # Dataset-level mean precision/recall; NaN if the arrays came back empty
        # (e.g. a degenerate run with no detections) so it's visibly missing.
        "precision": float(p_arr.mean()) if p_arr.size else float("nan"),
        "recall": float(r_arr.mean()) if r_arr.size else float("nan"),
    }

    # Per-class AP50 — `res.box.ap50` is shape (nc,), one entry per class in the
    # CLASS_NAMES order (MH, PH, WLPH). Guard the index in case the model has
    # fewer classes than expected, recording NaN rather than crashing.
    ap50 = np.asarray(res.box.ap50)
    for i, name in enumerate(CLASS_NAMES):
        out[f"AP50_{name}"] = float(ap50[i]) if i < len(ap50) else float("nan")

    # Record artifact locations so the thesis / downstream tooling can find them.
    out["val_dir"] = str(res.save_dir)
    cm_png = Path(res.save_dir) / "confusion_matrix.png"
    if cm_png.exists():
        out["confusion_matrix_path"] = str(cm_png)
    return out


def collect_runs(runs_dir: Path) -> pd.DataFrame:
    """Gather all ``results.csv`` files under ``runs_dir`` into one DataFrame.

    Recursively finds every Ultralytics ``results.csv`` and takes its final row
    (the converged metrics), tagging each with the run directory name and path.
    Useful for a quick cross-run overview / sanity check.

    Args:
        runs_dir: Root directory to search recursively.

    Returns:
        A DataFrame with one row per run (empty if none found). Unreadable CSVs
        are skipped with a warning rather than aborting the whole scan.
    """
    rows: List[Dict] = []
    for csv in runs_dir.rglob("results.csv"):
        try:
            df = pd.read_csv(csv)
            # Last row = final epoch's metrics.
            last = df.iloc[-1].to_dict()
            # Annotate provenance so rows remain identifiable once combined.
            last["run"] = csv.parent.name
            last["path"] = str(csv.parent)
            rows.append(last)
        except Exception as e:
            # One bad CSV shouldn't kill the whole collection pass.
            print(f"[warn] failed to read {csv}: {e}")
    return pd.DataFrame(rows)


def save_results(metrics: Dict, name: str) -> Path:
    """Persist a metrics dict to ``RESULTS_DIR/<name>.json``.

    This produces the per-run ``eval.json`` artifact that
    ``src.eval.aggregate_seeds`` later prefers when building the summary table.

    Args:
        metrics: Flat metrics dict, typically the return value of ``run_eval``.
        name: Base filename (without extension) for the JSON file.

    Returns:
        The path to the written JSON file.

    Side effects:
        Writes a pretty-printed JSON file to disk and prints its location.
    """
    out = RESULTS_DIR / f"{name}.json"
    import json
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[ok] metrics → {out}")
    return out
