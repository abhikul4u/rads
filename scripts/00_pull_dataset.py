#!/usr/bin/env python
"""Stage 00 — pull the RADS dataset from Roboflow and verify class order."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.roboflow_pull import pull

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    pull(force=args.force)
