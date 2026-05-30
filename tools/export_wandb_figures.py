"""Export RADS Layer 3 W&B run metrics and generate thesis figures (v2).

Author: Rutuja Kulkarni

Role in the RADS Layer 3 pipeline
---------------------------------
This is the *results-reporting* stage that turns raw training telemetry into the
publication-quality figures used in Chapter 4 of the thesis. During Layer 3, each
YOLOv8 variant is trained for multiple seeds on RunPod A100s, with Ultralytics
logging per-epoch metrics (losses, mAP, precision/recall, learning rate) to
Weights & Biases. This script pulls all of those runs back down, aligns and
aggregates them across seeds (mean ± std), and renders consistent, styled
matplotlib figures so the thesis tells one coherent visual story.

The six experimental variants it knows about are the Layer 3 ablation ladder:
``baseline`` (vanilla YOLOv8l), ``cbam`` (attention), ``p2`` (extra high-res
detection head), ``sizeaware`` (size-aware box loss), ``combined`` (the stacked
improvements), and ``distill`` (the small distilled student). Each is assigned a
fixed colour and plot order so every figure is directly comparable.

W&B runs are expected to be named ``<variant>_seed<N>`` (e.g. ``cbam_seed42``);
:data:`NAME_RE` parses that. Because W&B does NOT log an explicit ``epoch`` for
this setup, the script uses W&B's internal ``_step`` counter as the x-axis proxy
(Ultralytics logs once per epoch, so step ≈ epoch) — this is the central fix that
distinguishes the v2 of this exporter.

Downloaded run histories are cached to ``per_run_csv/*.csv`` so figures can be
re-rendered offline with ``--no-download`` without re-querying W&B. Some figures
(F4 bar charts) instead read a separately-produced test-set summary CSV, because
per-class test metrics were not logged to W&B during training.

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

# Global matplotlib styling applied to every figure for a consistent, thesis-ready
# look: sans-serif fonts, no top/right spines, and higher DPI for print quality.
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

# Fixed colour per variant so the same variant is always the same colour across
# every figure in the thesis (aids cross-figure comparison).
VARIANT_COLORS = {
    "baseline":  "#666666",
    "cbam":      "#1f77b4",
    "p2":        "#2ca02c",
    "sizeaware": "#ff7f0e",
    "combined":  "#d62728",
    "distill":   "#9467bd",
}

# Canonical left-to-right plotting/legend order (the ablation ladder).
VARIANT_ORDER = ["baseline", "cbam", "p2", "sizeaware", "combined", "distill"]

# Run-name parser: "<variant>_seed<N>". The variant group is non-greedy and allows
# digits (the v2 fix) so names like "p2_seed42" split correctly into ("p2", 42)
# instead of swallowing the trailing digit of the variant.
NAME_RE = re.compile(r"^(?P<variant>[a-z0-9_]+?)_seed(?P<seed>\d+)$")

# W&B step counter used as the x-axis. W&B doesn't log an 'epoch' column for this
# setup, but Ultralytics logs once per epoch, so _step is a faithful epoch proxy.
STEP_COL = "_step"


def parse_run_name(name: str):
    """Split a W&B run display name into ``(variant, seed)``.

    Centralises the naming convention so both the download path and the
    load-from-CSV path agree on which runs are ours and which to ignore.

    Args:
        name: The W&B run display name (or a CSV stem), e.g. ``"combined_seed7"``.

    Returns:
        A ``(variant_str, seed_int)`` tuple if the name matches the expected
        ``<variant>_seed<N>`` pattern, otherwise ``None`` (caller skips it).
    """
    m = NAME_RE.match(name)
    if not m:
        return None
    return m.group("variant"), int(m.group("seed"))


def download_runs(entity: str, project: str, out_dir: Path) -> pd.DataFrame:
    """Download every matching run's full per-step history from W&B into one frame.

    Connects to the W&B project, iterates its runs, keeps only those whose names
    match ``<variant>_seed<N>``, and pulls each run's complete metric history. Each
    run is cached to ``<out_dir>/per_run_csv/<run_name>.csv`` and tagged with its
    variant/seed so later aggregation can group by them. When the same display name
    appears more than once (W&B allows duplicate names), the longer history wins —
    this avoids a truncated/crashed run overwriting a complete one.

    If nothing could be downloaded (e.g. offline), it falls back to whatever CSVs
    already exist on disk so the user is not left empty-handed.

    Args:
        entity: W&B entity (user or team) owning the project.
        project: W&B project name to read runs from.
        out_dir: Output directory; per-run CSVs and the combined CSV land here.

    Returns:
        A single long-format DataFrame concatenating every kept run's history.

    Raises:
        RuntimeError: If no runs download successfully and no cached CSVs exist.
    """
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

        # Derive a pseudo-epoch column. Prefer W&B's _step (≈ one per epoch);
        # if that column is somehow absent, fall back to row index as the x-axis.
        if STEP_COL in hist.columns:
            hist["epoch"] = hist[STEP_COL]
        else:
            hist["epoch"] = range(len(hist))

        # De-duplicate by display name: keep whichever copy has more rows (i.e.
        # the longer / more complete training history), discard the shorter one.
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
        # Nothing came down from W&B (offline / empty project) — recover by
        # reading any previously-cached per-run CSVs so figures can still render.
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

    # Stack every run into one long-format table and persist it for convenience.
    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(out_dir / "all_runs_long.csv", index=False)
    print(f"\nCombined dataset: {len(df)} rows across {df['run_name'].nunique()} runs")
    return df


def load_from_existing_csvs(out_dir: Path) -> pd.DataFrame:
    """Rebuild the combined DataFrame from cached per-run CSVs (the --no-download path).

    Mirrors the tail of :func:`download_runs` but skips W&B entirely, letting the
    user iterate on figure styling without re-querying the API. Re-derives the
    variant/seed/run_name tags (and the ``epoch`` proxy) from each CSV's filename
    and contents.

    Args:
        out_dir: The output directory that contains the ``per_run_csv/`` cache.

    Returns:
        The concatenated long-format DataFrame across all cached runs.

    Raises:
        RuntimeError: If the cache directory has no CSVs, or none of them match
            the expected run-name pattern.
    """
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
    """Collapse a metric across seeds into per-(variant, epoch) mean/std/count.

    This is the seed-averaging step that gives every curve its mean line and its
    ±std shaded band. It is defensive about messy W&B data: missing columns return
    an empty frame (so the caller can skip the plot gracefully), and non-numeric
    cells are coerced to NaN and dropped because W&B occasionally logs string
    artifacts in the middle of an otherwise-numeric column.

    Args:
        df: The combined long-format run history.
        metric: The column name to aggregate (e.g. ``"metrics/mAP50(B)"``).

    Returns:
        A DataFrame with columns ``[variant, epoch, mean, std, count]``, or an
        empty DataFrame if the metric is absent or entirely non-numeric.
    """
    if metric not in df.columns or "epoch" not in df.columns:
        return pd.DataFrame()
    # Work on a copy of just the columns we need so we don't mutate the caller's df.
    sub = df[["variant", "epoch", metric]].copy()
    # Coerce to numeric and drop NaN (W&B sometimes logs string artifacts mid-column).
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=[metric])
    if sub.empty:
        return pd.DataFrame()
    # One row per (variant, epoch): mean & std are across the seeds at that epoch;
    # count records how many seeds contributed (useful for sanity-checking n=3).
    grouped = sub.groupby(["variant", "epoch"])[metric].agg(["mean", "std", "count"]).reset_index()
    return grouped


def plot_metric_per_variant(df: pd.DataFrame, metric: str, title: str, ylabel: str,
                              out_path: Path, smooth: bool = True,
                              legend_loc: str = "auto") -> None:
    """Plot one metric vs. epoch as overlaid per-variant mean curves with std bands.

    The workhorse figure renderer. For each variant (in canonical order) it draws
    the seed-mean curve in that variant's fixed colour and shades a ±std band so
    the reader can see both the central trend and the run-to-run variability. If
    the metric is missing or no variant has data, it skips writing the file (and
    closes the figure) rather than emitting a blank plot.

    Args:
        df: Combined long-format run history.
        metric: Column to plot (e.g. ``"val/box_loss"``).
        title: Figure title.
        ylabel: Y-axis label.
        out_path: PNG path to save to (filename echoed to stdout on success).
        smooth: If True, apply a light centred rolling mean to de-noise the curve
            (window scales with series length); the std band is left un-smoothed.
        legend_loc: matplotlib legend location, or ``"auto"`` to pick a sensible
            corner based on whether the metric is a loss (top) or a score (bottom).
    """
    agg = _aggregate_by_epoch(df, metric)
    if agg.empty:
        print(f"  Skipping {out_path.name}: metric '{metric}' missing or all-NaN")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_any = False  # track whether any variant contributed, to avoid blank PNGs
    for variant in VARIANT_ORDER:
        sub = agg[agg["variant"] == variant].sort_values("epoch")
        if sub.empty:
            continue  # this variant has no data for this metric
        x = sub["epoch"].values
        y = sub["mean"].values
        yerr = sub["std"].fillna(0).values  # single-seed points have NaN std -> 0 band
        if smooth and len(y) > 5:
            # Window grows with series length (~1/30th) but never below 3 points;
            # center=True keeps the smoothed curve aligned with the raw one.
            window = max(3, len(y) // 30)
            y_smooth = pd.Series(y).rolling(window, min_periods=1, center=True).mean().values
        else:
            y_smooth = y
        color = VARIANT_COLORS[variant]
        ax.plot(x, y_smooth, label=variant, color=color, linewidth=2)
        # Translucent ±std band around the mean line.
        ax.fill_between(x, y_smooth - yerr, y_smooth + yerr, color=color, alpha=0.15)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)  # release the empty figure so we don't leak it
        print(f"  Skipping {out_path.name}: no data plotted")
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Losses descend (legend best top-right); scores ascend (legend best bottom-right).
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
    """Render the F1 figure family: train & val box/cls/dfl loss curves.

    Produces six ``F1_*.png`` files (the three YOLOv8 loss components for both
    train and val splits), each comparing all variants. These show training
    dynamics and convergence/overfitting behaviour for Chapter 4.

    Args:
        df: Combined run history.
        out_dir: Directory the F1 PNGs are written to.
    """
    print("\nF1: Training and validation loss curves")
    # (metric_column, figure_title, y_axis_label) for each loss component/split.
    losses = [
        ("train/box_loss", "Training Box Loss vs Epoch", "Box loss"),
        ("train/cls_loss", "Training Cls Loss vs Epoch", "Classification loss"),
        ("train/dfl_loss", "Training DFL Loss vs Epoch", "DFL loss"),
        ("val/box_loss",   "Validation Box Loss vs Epoch", "Box loss"),
        ("val/cls_loss",   "Validation Cls Loss vs Epoch", "Classification loss"),
        ("val/dfl_loss",   "Validation DFL Loss vs Epoch", "DFL loss"),
    ]
    for metric, title, ylabel in losses:
        # Sanitise the metric name into a filesystem-safe filename fragment.
        safe = metric.replace("/", "_").replace("(", "").replace(")", "")
        plot_metric_per_variant(df, metric, title, ylabel,
                                  out_dir / f"F1_{safe}.png",
                                  legend_loc="upper right")  # losses -> top-right legend


def plot_map_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """Render the F2 figure family: validation mAP / precision / recall curves.

    These are the headline accuracy plots — mAP@0.5, mAP@0.5:0.95, precision and
    recall over training — comparing every variant with mean ± std across seeds.

    Args:
        df: Combined run history.
        out_dir: Directory the F2 PNGs are written to.
    """
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
    """F3 placeholder: per-class AP-over-epoch curves are intentionally not drawn.

    Per-class AP was not logged to W&B during training for this setup, so there is
    nothing to plot per-epoch. This function exists to document that gap in the
    figure sequence and to point the reader to where per-class numbers DO live
    (the test-eval JSONs, summarised in the F4b bar chart). It only prints a note.

    Args:
        df: Combined run history (unused; kept for a uniform plot-fn signature).
        out_dir: Output directory (unused here).
    """
    print("\nF3: Per-class AP50 curves")
    print("  (Per-class metrics weren't logged to W&B during training;")
    print("   they're available in the test eval JSONs and shown in F4b bar chart.)")


def plot_lr_schedule(df: pd.DataFrame, out_dir: Path) -> None:
    """Render F5: the cosine learning-rate schedule actually used during training.

    Reads back the logged LR (the ``lr/pg*`` param-group column) rather than
    re-deriving it analytically, so the figure reflects exactly what the optimizer
    saw. It prefers the canonical ``baseline`` / ``seed 42`` run for a clean single
    curve; if that specific run is absent it falls back to the first ~100 rows per
    run. Skips gracefully (just prints why) if the LR column or epoch axis is
    missing, or if every LR value is NaN.

    Args:
        df: Combined run history.
        out_dir: Directory the ``F5_lr_schedule.png`` is written to.
    """
    print("\nF5: Learning rate schedule")
    # Find the first learning-rate param-group column (Ultralytics names them lr/pg0,...).
    lr_col = next((c for c in df.columns if c.startswith("lr/pg")), None)
    if not lr_col:
        print("  No lr/pg* column found, skipping")
        return
    if "epoch" not in df.columns:
        print("  No 'epoch' column, skipping")
        return
    # The schedule is identical across runs, so plot one representative run.
    sub = df[(df["variant"] == "baseline") & (df["seed"] == 42)].sort_values("epoch")
    if sub.empty:
        # Fallback when the canonical run isn't present: take the head of each run.
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
    """Render F4: a grouped bar chart of final test-set mAP@0.5 and mAP@0.5:0.95.

    Unlike the curve plots, this reads the separately-produced test-eval summary
    CSV (which holds final per-variant means/stds across seeds), because test-set
    metrics were not logged to W&B. Bars are ordered by :data:`VARIANT_ORDER`,
    annotated with error bars (std) and value labels. Skips with guidance if the
    summary CSV is not present yet.

    Args:
        summary_csv: Path to ``layer3_summary.csv`` produced from the test eval.
        out_dir: Directory the ``F4_final_map_bars.png`` is written to.
    """
    if not summary_csv.exists():
        print(f"\nF4: skipped (no {summary_csv})")
        print("  Get layer3_summary.csv from the pod and re-run --no-download")
        return
    print("\nF4: Final mAP bar chart")
    df = pd.read_csv(summary_csv)
    # Index by variant and reorder to the canonical ladder, keeping only present ones.
    df = df.set_index("variant").loc[[v for v in VARIANT_ORDER if v in df.index]]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))  # one tick group per variant
    w = 0.35  # bar width; two bars per group sit at x±w/2
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
    # Annotate each mAP@0.5 bar with its numeric height for precise reading.
    for bar in b1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width()/2, h),
                    ha="center", va="bottom", fontsize=9)
    fig.savefig(out_dir / "F4_final_map_bars.png")
    plt.close(fig)
    print("  + F4_final_map_bars.png")


def plot_per_class_bar(summary_csv: Path, out_dir: Path) -> None:
    """Render F4b: grouped bars of per-class test AP@0.5 (MH / PH / WLPH).

    This is where the per-class story missing from F3 is told: for each variant,
    three bars show AP@0.5 for the manhole, pothole and water-logged-pothole
    classes (with std error bars across seeds), revealing which anomaly types each
    variant handles best. Reads the same test-eval summary CSV as F4 and silently
    returns if it is absent.

    Args:
        summary_csv: Path to ``layer3_summary.csv``.
        out_dir: Directory the ``F4b_per_class_bars.png`` is written to.
    """
    if not summary_csv.exists():
        return
    print("\nF4b: Per-class AP50 bar chart")
    df = pd.read_csv(summary_csv)
    df = df.set_index("variant").loc[[v for v in VARIANT_ORDER if v in df.index]]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))  # one tick group per variant
    w = 0.27  # three bars per group sit at x-w, x, x+w
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
    """CLI entry point: obtain run data (download or cache) and emit all figures.

    Parses arguments, ensures the output directory exists, then either downloads
    fresh run histories from W&B or loads the cached CSVs (``--no-download``).
    Finally it calls every figure generator (F1 losses, F2 mAP/precision/recall,
    F3 note, F5 LR schedule, F4 + F4b summary bars) and reports how many PNGs were
    produced. Each generator independently no-ops when its inputs are missing, so
    a partial dataset still yields whatever figures are possible.
    """
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

    # Choose data source: cached CSVs for fast offline re-plotting, else fetch W&B.
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