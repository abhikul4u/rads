#!/usr/bin/env python
"""Stage 05 — evaluate a checkpoint and save metrics JSON next to the run dir.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 pipeline
---------------------------------
This stage scores any trained checkpoint (baseline, an ablation variant, or
the distilled student) on a held-out split — the test split by default — and
records the standard detection metrics for the RADS classes
["MH", "PH", "WLPH"]: overall mAP, per-class AP, precision/recall, and the
confusion matrix. These numbers are what populate the thesis results tables.

Crucially, besides printing the metrics it writes an ``eval.json`` *inside the
checkpoint's run directory*. Stage 07 (aggregate_results) later walks the run
directories looking for exactly those files to compute mean +/- std across the
three seeds, so this side effect is what links evaluation to aggregation.

Example:
    python scripts/05_evaluate.py \
        --weights artifacts/runs/combined_seed42/weights/best.pt \
        --imgsz 768 --split test
"""
from __future__ import annotations
import sys, os
# Make the repo root importable so ``src.*`` resolves when run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import json
from pathlib import Path

from src.data.roboflow_pull import pull
from src.eval.metrics import run_eval, save_results


def main():
    """Parse CLI args, evaluate the checkpoint, and persist the metrics.

    Flow:
      1. Ensure the dataset is present and locate its data.yaml.
      2. Derive a result name (defaults to "<run_dir_name>_<split>").
      3. Run evaluation via ``run_eval`` and pretty-print the metrics.
      4. Write ``eval.json`` into the checkpoint's run directory (so Stage 07
         can aggregate it) and also archive results via ``save_results``.

    No value is returned.
    """
    ap = argparse.ArgumentParser()
    # --weights (required): path to the checkpoint (best.pt) to evaluate.
    ap.add_argument("--weights", required=True, type=Path)
    # --imgsz: square evaluation image size in px. Use 768 for the full-size
    #          models and 640 when evaluating the mobile-size student.
    ap.add_argument("--imgsz", type=int, default=768)
    # --split: which dataset split to score on. "test" is the canonical
    #          reporting split; "val"/"valid" mirror Roboflow's naming and
    #          "train" is available for sanity checks.
    ap.add_argument("--split", default="test", choices=["train", "val", "valid", "test"])
    # --name: optional label for the saved results; defaults to the run
    #         directory name suffixed with the split.
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    # Ensure dataset presence (Stage 00 cache) and locate its data spec.
    dataset = pull()
    data_yaml = dataset / "data.yaml"
    # ``weights`` is .../<run>/weights/best.pt, so parent.parent is the run
    # directory; its name (e.g. "combined_seed42") makes a readable result tag.
    name = args.name or args.weights.parent.parent.name + f"_{args.split}"
    # Run the actual evaluation (mAP, per-class AP, P/R, confusion matrix).
    metrics = run_eval(args.weights, data_yaml, args.imgsz, args.split, name)
    print(json.dumps(metrics, indent=2))

    # Also drop eval.json into the run dir so aggregate_seeds can find it.
    # This is the contract Stage 07 relies on for cross-seed aggregation.
    run_dir = args.weights.parent.parent
    (run_dir / "eval.json").write_text(json.dumps(metrics, indent=2))
    # Archive a copy of the metrics through the central results helper too.
    save_results(metrics, name)


if __name__ == "__main__":
    main()
