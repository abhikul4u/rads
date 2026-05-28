#!/usr/bin/env python
"""Stage 05 — evaluate a checkpoint and save metrics JSON next to the run dir.

Example:
    python scripts/05_evaluate.py \
        --weights artifacts/runs/combined_seed42/weights/best.pt \
        --imgsz 768 --split test
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import json
from pathlib import Path

from src.data.roboflow_pull import pull
from src.eval.metrics import run_eval, save_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--split", default="test", choices=["train", "val", "valid", "test"])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    dataset = pull()
    data_yaml = dataset / "data.yaml"
    name = args.name or args.weights.parent.parent.name + f"_{args.split}"
    metrics = run_eval(args.weights, data_yaml, args.imgsz, args.split, name)
    print(json.dumps(metrics, indent=2))

    # Also drop eval.json into the run dir so aggregate_seeds can find it.
    run_dir = args.weights.parent.parent
    (run_dir / "eval.json").write_text(json.dumps(metrics, indent=2))
    save_results(metrics, name)


if __name__ == "__main__":
    main()
