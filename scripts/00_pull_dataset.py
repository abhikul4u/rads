#!/usr/bin/env python
"""Stage 00 — pull the RADS dataset from Roboflow and verify class order.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 pipeline
---------------------------------
This is the very first stage of the "RADS Layer 3 — Model Development
Pipeline". Before any model can be trained, evaluated, distilled or
quantized, every later stage needs a single, canonical copy of the
Road Anomaly Detection System (RADS) dataset on disk. This script is the
entry point that fetches that dataset from Roboflow and caches it on the
RunPod persistent volume so that the (potentially slow) download only ever
happens once across all experiments and reboots.

The heavy lifting is delegated to ``src.data.roboflow_pull.pull``, which:
  * downloads the dataset export from Roboflow (or reuses the cached copy),
  * lays it out in the YOLO directory format expected by Ultralytics, and
  * verifies that the class order is exactly ["MH", "PH", "WLPH"]
    (manhole, pothole, water-logged pothole) — a strict ordering that every
    downstream training/eval stage depends on for consistent class indices.

Because every other numbered stage calls ``pull()`` itself, running this
script is mainly a convenience for "warming" the cache up front (and for
forcing a clean re-download when the dataset has been re-exported).

Example CLI invocation
----------------------
    # Use the cached dataset if present, otherwise download it once:
    python scripts/00_pull_dataset.py

    # Ignore any cached copy and re-pull a fresh export from Roboflow:
    python scripts/00_pull_dataset.py --force
"""
import sys, os
# Put the repository root (the parent of scripts/) on sys.path so that the
# absolute ``src.*`` imports below resolve no matter what directory the
# script is launched from. This keeps every stage runnable standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.roboflow_pull import pull

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    # --force: when set, bypass the on-disk cache and re-download the dataset
    #          export from Roboflow from scratch. Leave it off (the default,
    #          False) for normal runs so the cached copy on the persistent
    #          volume is reused and we avoid a redundant network pull.
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    # Trigger the download/verification. The verified dataset path returned by
    # pull() is what later stages consume; here we only care about the side
    # effect of populating (or refreshing) the cache.
    pull(force=args.force)
