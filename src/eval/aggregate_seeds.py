"""Aggregate per-seed results into mean ± std rows for thesis Table 4.X.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 (Model Development) pipeline
-----------------------------------------------------
This module sits at the very tail of the evaluation stage. Every RADS variant
(baseline, cbam, p2, sizeaware, combined, distill) is trained and evaluated
across THREE random seeds (42, 1337, 2024) to make the reported numbers
statistically defensible rather than the product of a single lucky run. Each
individual run leaves behind a per-seed metrics artifact (either our own
``eval.json`` produced by ``src.eval.metrics`` or Ultralytics' ``results.csv``).

The job of this file is to crawl the run directory, group runs by variant,
average each metric across the seeds, and compute the (sample) standard
deviation. The resulting "mean ± std" rows are written to a CSV that maps
directly onto the thesis results table (Table 4.X) comparing the architectural
variants. In other words: many noisy per-seed JSONs in, one clean comparison
table out.

The metrics aggregated are the standard detection figures: mAP@0.50,
mAP@0.50:0.95, mean precision/recall, and the per-class AP@0.50 for the three
RADS classes — MH (manhole), PH (pothole) and WLPH (water-logged pothole).

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

# Run directories are named ``{variant}_seed{N}`` (e.g. "cbam_seed1337").
# The regex captures the variant name (non-greedy, so the trailing "_seed{N}"
# is peeled off correctly even when the variant itself contains underscores,
# plus signs or hyphens, as in "p2+cbam") and the integer seed separately.
NAME_RE = re.compile(r"^(?P<variant>[A-Za-z0-9_+\-]+?)_seed(?P<seed>\d+)$")

# The exact set of metrics we surface in the thesis comparison table. The four
# aggregate scores plus the three per-class AP@0.50 numbers for MH/PH/WLPH.
METRICS = ["map50", "map50_95", "precision", "recall",
           "AP50_MH", "AP50_PH", "AP50_WLPH"]


def _load_run_metrics(run_dir: Path) -> dict | None:
    """Load a single run's final metrics from whatever artifact is present.

    A run may have been evaluated by our own ``src.eval.metrics`` (which writes
    a rich ``eval.json`` including per-class AP) or it may only have Ultralytics'
    training-time ``results.csv``. We prefer the former because it is richer,
    falling back to the latter so that runs without an explicit eval pass are
    still counted.

    Args:
        run_dir: A single run directory, e.g. ``artifacts/runs/cbam_seed42``.

    Returns:
        A flat ``{metric_name: value}`` dict, or ``None`` if neither artifact
        exists / yields any usable metric (the caller then skips this run).
    """
    # 1. Preferred source: our own eval JSON written by src.eval.metrics.
    #    It already uses the canonical metric keys (map50, AP50_MH, ...).
    eval_json = run_dir / "eval.json"
    if eval_json.exists():
        return json.loads(eval_json.read_text())
    # 2. Fallback: Ultralytics' per-epoch results.csv. The final epoch (last
    #    row) holds the converged validation metrics, so we read that row.
    csv = run_dir / "results.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        # Ultralytics pads column headers with spaces; strip so lookups match.
        df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]
        # Translate Ultralytics' verbose column names into our canonical keys.
        # The "(B)" suffix denotes the bounding-box (detection) metrics.
        # Note: results.csv has no per-class AP columns, so the fallback path
        # only contributes the four aggregate scores.
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
    """Aggregate every per-seed run under ``runs_dir`` into mean ± std rows.

    Walks the run directory, groups runs by variant, and for each metric in
    ``METRICS`` computes the cross-seed mean and the sample standard deviation.
    The combined table is written to ``out_csv`` and also echoed to stdout.

    Args:
        runs_dir: Directory holding one sub-directory per run, each named
            ``{variant}_seed{N}``.
        out_csv: Destination CSV path. Parent directories are created if
            missing.

    Returns:
        The aggregated DataFrame (one row per variant), sorted by variant name.

    Side effects:
        Writes ``out_csv`` to disk and prints progress / the final table.
    """
    # Bucket each run's metrics dict under its variant so seeds sit together.
    by_variant = defaultdict(list)
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        # Only consider directories whose name matches the {variant}_seed{N}
        # convention; anything else (logs, scratch dirs) is ignored.
        m = NAME_RE.match(d.name)
        if not m:
            continue
        metrics = _load_run_metrics(d)
        if not metrics:
            # No usable artifact — warn and move on rather than crashing the
            # whole aggregation (a single missing run shouldn't block the table).
            print(f"[skip] no metrics in {d.name}")
            continue
        # Tag with the seed (prefixed "_" to mark it as bookkeeping, not a
        # reported metric) and file under the variant bucket.
        metrics["_seed"] = int(m.group("seed"))
        by_variant[m.group("variant")].append(metrics)

    rows = []
    for variant, runs in by_variant.items():
        # n_seeds records how many seeds actually contributed — useful for
        # spotting variants where a run failed and the std is less trustworthy.
        row = {"variant": variant, "n_seeds": len(runs)}
        for metric in METRICS:
            # Collect this metric across seeds, tolerating runs that lack it
            # (e.g. results.csv fallback runs have no per-class AP).
            vals = [r[metric] for r in runs if metric in r]
            if not vals:
                continue
            # Cross-seed mean = the headline number reported in the thesis.
            row[f"{metric}_mean"] = sum(vals) / len(vals)
            if len(vals) > 1:
                # Sample (Bessel-corrected, n-1) standard deviation. We use the
                # sample estimator because the seeds are a small sample drawn to
                # estimate run-to-run variability, not the whole population.
                m = row[f"{metric}_mean"]
                row[f"{metric}_std"] = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            else:
                # A single seed has no spread to report.
                row[f"{metric}_std"] = 0.0
        rows.append(row)

    # Sort by variant for a stable, reproducible table ordering.
    df = pd.DataFrame(rows).sort_values("variant")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\n[ok] summary → {out_csv}")
    return df


if __name__ == "__main__":
    # CLI entry point: point at the runs directory and a CSV destination, then
    # hand off to aggregate(). Both arguments are required.
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)  # dir of *_seed* runs
    ap.add_argument("--out", type=Path, required=True)       # summary CSV path
    args = ap.parse_args()
    aggregate(args.runs_dir, args.out)
