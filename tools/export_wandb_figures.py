"""Export RADS Layer 3 W&B run metrics and generate thesis figures (v2).

v2 fixes:
  - Use _step (W&B's step counter) as x-axis since W&B doesn't log 'epoch'
  - Fix regex to handle variant names containing digits (e.g. 'p2')
  - Better error message when columns are missing

Pulls all runs from the W&B project, downloads their full per-step history,
and produces publication-quality figures for thesis Chapter 4.

Usage:
    python tools/export_wandb_figures.py \\
        --entity abhikul4u-hvpm-college-of-engineering-technology-amravati \\
        --project "-workspace-artifacts-runs" \\
        --output thesis_figures \\
        --summary-csv thesis_results/layer3_summary.csv

    # Re-run without re-downloading from W&B:
    python tools/export_wandb_figures.py ... --no-download
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:
    import wandb
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    print("Install: pip install wandb pandas matplotlib numpy", file=sys.stderr)
    sys.exit(1)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

VARIANT_COLORS = {
    "baseline":  "#666666",
    "cbam":      "#1f77b4",
    "p2":        "#2ca02c",
    "sizeaware": "#ff7f0e",
    "combined":  "#d62728",
    "distill":   "#9467bd",
}

VARIANT_ORDER = ["baseline", "cbam", "p2", "sizeaware", "combined", "distill"]

# Allow digits in the variant name (e.g. "p2")
NAME_RE = re.compile(r"^(?P<variant>[a-z0-9_]+?)_seed(?P<seed>\d+)$")

# W&B step counter is what we use as x-axis (Ultralytics logs once per epoch)
STEP_COL = "_step"


def parse_run_name(name: str):
    m = NAME_RE.match(name)
    if not m:
        return None
    return m.group("variant"), int(m.group("seed"))


def download_runs(entity: str, project: str, out_dir: Path) -> pd.DataFrame:
    print(f"Connecting to W&B: {entity}/{project}")
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}"))
    print(f"Found {len(runs)} runs")

    all_dfs = []
    csv_dir = out_dir / "per_run_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    seen = defaultdict(int)  # name → count of dup downloads
    for run in runs:
        parsed = parse_run_name(run.name)
        if not parsed:
            print(f"  Skipping {run.name} (not in variant_seedN format)")
            continue
        variant, seed = parsed

        # Skip duplicate run names (W&B can have multiple runs with same display name)
        # Keep the one with the most rows
        print(f"  Downloading {run.name}... ", end="", flush=True)
        try:
            hist = run.history(pandas=True, samples=10000)
        except Exception as e:
            print(f"FAILED ({e})")
            continue
        if hist.empty:
            print("EMPTY")
            continue

        hist["variant"] = variant
        hist["seed"] = seed
        hist["run_name"] = run.name

        # Compute pseudo-epoch from _step (Ultralytics logs once per epoch)
        if STEP_COL in hist.columns:
            hist["epoch"] = hist[STEP_COL]
        else:
            hist["epoch"] = range(len(hist))

        # If we already have this run name with more rows, skip this one
        existing_path = csv_dir / f"{run.name}.csv"
        if existing_path.exists():
            existing = pd.read_csv(existing_path)
            if len(existing) >= len(hist):
                print(f"{len(hist)} rows -> skipped (existing has {len(existing)})")
                continue
            else:
                print(f"{len(hist)} rows -> overwriting (was {len(existing)})")
        else:
            print(f"{len(hist)} rows")

        hist.to_csv(existing_path, index=False)
        all_dfs.append(hist)
        seen[run.name] += 1

    if not all_dfs:
        # Try loading from existing CSVs if nothing was downloaded
        csv_paths = list(csv_dir.glob("*.csv"))
        if csv_paths:
            print(f"Loading {len(csv_paths)} existing CSVs")
            for p in csv_paths:
                parsed = parse_run_name(p.stem)
                if not parsed:
                    continue
                variant, seed = parsed
                d = pd.read_csv(p)
                d["variant"] = variant
                d["seed"] = seed
                d["run_name"] = p.stem
                if "epoch" not in d.columns and STEP_COL in d.columns:
                    d["epoch"] = d[STEP_COL]
                all_dfs.append(d)

    if not all_dfs:
        raise RuntimeError("No runs downloaded successfully and no existing CSVs found")

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(out_dir / "all_runs_long.csv", index=False)
    print(f"\nCombined dataset: {len(df)} rows across {df['run_name'].nunique()} runs")
    return df


def load_from_existing_csvs(out_dir: Path) -> pd.DataFrame:
    csv_dir = out_dir / "per_run_csv"
    csv_paths = list(csv_dir.glob("*.csv"))
    if not csv_paths:
        raise RuntimeError(f"No CSVs in {csv_dir}")
    print(f"Loading {len(csv_paths)} existing CSVs from {csv_dir}")
    dfs = []
    for p in csv_paths:
        parsed = parse_run_name(p.stem)
        if not parsed:
            continue
        variant, seed = parsed
        d = pd.read_csv(p)
        d["variant"] = variant
        d["seed"] = seed
        d["run_name"] = p.stem
        if "epoch" not in d.columns and STEP_COL in d.columns:
            d["epoch"] = d[STEP_COL]
        dfs.append(d)
    if not dfs:
        raise RuntimeError("No matching CSVs found")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows across {df['run_name'].nunique()} runs")
    return df


def _aggregate_by_epoch(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns or "epoch" not in df.columns:
        return pd.DataFrame()
    # Coerce to numeric and drop NaN (W&B sometimes logs string artifacts mid-column)
    sub = df[["variant", "epoch", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=[metric])
    if sub.empty:
        return pd.DataFrame()
    grouped = sub.groupby(["variant", "epoch"])[metric].agg(["mean", "std", "count"]).reset_index()
    return grouped


def plot_metric_per_variant(df: pd.DataFrame, metric: str, title: str, ylabel: str,
                              out_path: Path, smooth: bool = True,
                              legend_loc: str = "auto") -> None:
    agg = _aggregate_by_epoch(df, metric)
    if agg.empty:
        print(f"  Skipping {out_path.name}: metric '{metric}' missing or all-NaN")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_any = False
    for variant in VARIANT_ORDER:
        sub = agg[agg["variant"] == variant].sort_values("epoch")
        if sub.empty:
            continue
        x = sub["epoch"].values
        y = sub["mean"].values
        yerr = sub["std"].fillna(0).values
        if smooth and len(y) > 5:
            window = max(3, len(y) // 30)
            y_smooth = pd.Series(y).rolling(window, min_periods=1, center=True).mean().values
        else:
            y_smooth = y
        color = VARIANT_COLORS[variant]
        ax.plot(x, y_smooth, label=variant, color=color, linewidth=2)
        ax.fill_between(x, y_smooth - yerr, y_smooth + yerr, color=color, alpha=0.15)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(f"  Skipping {out_path.name}: no data plotted")
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if legend_loc == "auto":
        loc = "upper right" if "loss" in metric.lower() else "lower right"
    else:
        loc = legend_loc
    ax.legend(loc=loc, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  + {out_path.name}")


def plot_loss_curves(df: pd.DataFrame, out_dir: Path) -> None:
    print("\nF1: Training and validation loss curves")
    losses = [
        ("train/box_loss", "Training Box Loss vs Epoch", "Box loss"),
        ("train/cls_loss", "Training Cls Loss vs Epoch", "Classification loss"),
        ("train/dfl_loss", "Training DFL Loss vs Epoch", "DFL loss"),
        ("val/box_loss",   "Validation Box Loss vs Epoch", "Box loss"),
        ("val/cls_loss",   "Validation Cls Loss vs Epoch", "Classification loss"),
        ("val/dfl_loss",   "Validation DFL Loss vs Epoch", "DFL loss"),
    ]
    for metric, title, ylabel in losses:
        safe = metric.replace("/", "_").replace("(", "").replace(")", "")
        plot_metric_per_variant(df, metric, title, ylabel,
                                  out_dir / f"F1_{safe}.png",
                                  legend_loc="upper right")


def plot_map_curves(df: pd.DataFrame, out_dir: Path) -> None:
    print("\nF2: Validation mAP curves (headline plot)")
    plot_metric_per_variant(df, "metrics/mAP50(B)",
                              "Validation mAP@0.5 vs Epoch (mean ± std across seeds)",
                              "mAP@0.5",
                              out_dir / "F2_val_mAP50.png",
                              legend_loc="lower right")
    plot_metric_per_variant(df, "metrics/mAP50-95(B)",
                              "Validation mAP@0.5:0.95 vs Epoch",
                              "mAP@0.5:0.95",
                              out_dir / "F2_val_mAP50-95.png",
                              legend_loc="lower right")
    plot_metric_per_variant(df, "metrics/precision(B)",
                              "Validation Precision vs Epoch",
                              "Precision",
                              out_dir / "F2_val_precision.png",
                              legend_loc="lower right")
    plot_metric_per_variant(df, "metrics/recall(B)",
                              "Validation Recall vs Epoch",
                              "Recall",
                              out_dir / "F2_val_recall.png",
                              legend_loc="lower right")


def plot_per_class_ap(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-class metrics aren't in W&B for our setup — info only."""
    print("\nF3: Per-class AP50 curves")
    print("  (Per-class metrics weren't logged to W&B during training;")
    print("   they're available in the test eval JSONs and shown in F4b bar chart.)")


def plot_lr_schedule(df: pd.DataFrame, out_dir: Path) -> None:
    print("\nF5: Learning rate schedule")
    lr_col = next((c for c in df.columns if c.startswith("lr/pg")), None)
    if not lr_col:
        print("  No lr/pg* column found, skipping")
        return
    if "epoch" not in df.columns:
        print("  No 'epoch' column, skipping")
        return
    sub = df[(df["variant"] == "baseline") & (df["seed"] == 42)].sort_values("epoch")
    if sub.empty:
        sub = df.sort_values(["run_name", "epoch"]).groupby("run_name").head(100)
    sub = sub.dropna(subset=[lr_col])
    if sub.empty:
        print(f"  All {lr_col} values are NaN, skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sub["epoch"], sub[lr_col], color="#1f77b4", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Cosine LR Schedule (lr0=5e-4)")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "F5_lr_schedule.png")
    plt.close(fig)
    print("  + F5_lr_schedule.png")


def plot_final_bar(summary_csv: Path, out_dir: Path) -> None:
    if not summary_csv.exists():
        print(f"\nF4: skipped (no {summary_csv})")
        print("  Get layer3_summary.csv from the pod and re-run --no-download")
        return
    print("\nF4: Final mAP bar chart")
    df = pd.read_csv(summary_csv)
    df = df.set_index("variant").loc[[v for v in VARIANT_ORDER if v in df.index]]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    w = 0.35
    b1 = ax.bar(x - w/2, df["map50_mean"], w, yerr=df["map50_std"],
                 label="mAP@0.5", color="#1f77b4", capsize=4)
    b2 = ax.bar(x + w/2, df["map50_95_mean"], w, yerr=df["map50_95_std"],
                 label="mAP@0.5:0.95", color="#ff7f0e", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=0)
    ax.set_ylabel("mAP")
    ax.set_title("Test-Set mAP by Variant (mean ± std, n=3 seeds)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    for bar in b1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width()/2, h),
                    ha="center", va="bottom", fontsize=9)
    fig.savefig(out_dir / "F4_final_map_bars.png")
    plt.close(fig)
    print("  + F4_final_map_bars.png")


def plot_per_class_bar(summary_csv: Path, out_dir: Path) -> None:
    if not summary_csv.exists():
        return
    print("\nF4b: Per-class AP50 bar chart")
    df = pd.read_csv(summary_csv)
    df = df.set_index("variant").loc[[v for v in VARIANT_ORDER if v in df.index]]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.27
    ax.bar(x - w, df["AP50_MH_mean"], w, yerr=df["AP50_MH_std"],
            label="MH (manhole)", color="#666666", capsize=3)
    ax.bar(x,     df["AP50_PH_mean"], w, yerr=df["AP50_PH_std"],
            label="PH (pothole)", color="#1f77b4", capsize=3)
    ax.bar(x + w, df["AP50_WLPH_mean"], w, yerr=df["AP50_WLPH_std"],
            label="WLPH (waterlogged)", color="#2ca02c", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=0)
    ax.set_ylabel("AP@0.5")
    ax.set_title("Per-Class AP@0.5 by Variant (mean ± std, n=3 seeds)")
    ax.legend(loc="lower right", ncol=3)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_dir / "F4b_per_class_bars.png")
    plt.close(fig)
    print("  + F4b_per_class_bars.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--output", type=Path, default=Path("thesis_figures"))
    ap.add_argument("--summary-csv", type=Path,
                    default=Path("thesis_results/layer3_summary.csv"))
    ap.add_argument("--no-download", action="store_true",
                    help="Skip W&B download; use existing per_run_csv/*.csv")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.no_download:
        df = load_from_existing_csvs(args.output)
    else:
        df = download_runs(args.entity, args.project, args.output)

    plot_loss_curves(df, args.output)
    plot_map_curves(df, args.output)
    plot_per_class_ap(df, args.output)
    plot_lr_schedule(df, args.output)
    plot_final_bar(args.summary_csv, args.output)
    plot_per_class_bar(args.summary_csv, args.output)

    print(f"\n✓ Done. Figures in {args.output}/")
    pngs = list(args.output.glob("*.png"))
    print(f"  Generated {len(pngs)} PNG files")


if __name__ == "__main__":
    main()