"""Aggregate per-seed results into mean ± std rows for thesis Table 4.X.

Usage:
    python -m src.eval.aggregate_seeds \
        --runs-dir artifacts/runs \
        --out artifacts/results/layer3_summary.csv
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Run names follow:  {variant}_seed{N}
NAME_RE = re.compile(r"^(?P<variant>[A-Za-z0-9_+\-]+?)_seed(?P<seed>\d+)$")

METRICS = ["map50", "map50_95", "precision", "recall",
           "AP50_MH", "AP50_PH", "AP50_WLPH"]


def _load_run_metrics(run_dir: Path) -> dict | None:
    """Try a few well-known locations for a run's final metrics."""
    # 1. Our own eval JSON in results/
    eval_json = run_dir / "eval.json"
    if eval_json.exists():
        return json.loads(eval_json.read_text())
    # 2. Ultralytics' results.csv — take the last row.
    csv = run_dir / "results.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]
        mapper = {
            "metrics/mAP50(B)": "map50",
            "metrics/mAP50-95(B)": "map50_95",
            "metrics/precision(B)": "precision",
            "metrics/recall(B)": "recall",
        }
        out = {v: float(last[k]) for k, v in mapper.items() if k in last}
        return out if out else None
    return None


def aggregate(runs_dir: Path, out_csv: Path) -> pd.DataFrame:
    by_variant = defaultdict(list)
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        m = NAME_RE.match(d.name)
        if not m:
            continue
        metrics = _load_run_metrics(d)
        if not metrics:
            print(f"[skip] no metrics in {d.name}")
            continue
        metrics["_seed"] = int(m.group("seed"))
        by_variant[m.group("variant")].append(metrics)

    rows = []
    for variant, runs in by_variant.items():
        row = {"variant": variant, "n_seeds": len(runs)}
        for metric in METRICS:
            vals = [r[metric] for r in runs if metric in r]
            if not vals:
                continue
            row[f"{metric}_mean"] = sum(vals) / len(vals)
            if len(vals) > 1:
                m = row[f"{metric}_mean"]
                row[f"{metric}_std"] = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            else:
                row[f"{metric}_std"] = 0.0
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("variant")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\n[ok] summary → {out_csv}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    aggregate(args.runs_dir, args.out)
