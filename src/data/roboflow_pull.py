"""Pull the RADS dataset from Roboflow and cache it on the persistent volume.

Idempotent: if a valid YOLOv8 export already exists at the target path, we skip
the download. Force a refresh with --force.

Verifies the class ordering matches `CLASS_NAMES` in src/paths.py — a silent
class re-ordering by Roboflow would invalidate every downstream run.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

from src.paths import (
    CLASS_NAMES,
    RADS_DATA,
    ROBOFLOW_API_KEY,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
    ROBOFLOW_WORKSPACE,
)


def _existing_export(target: Path) -> bool:
    """Return True if `target` already holds a valid YOLOv8 export."""
    yaml_path = target / "data.yaml"
    if not yaml_path.exists():
        return False
    for split in ("train", "valid", "test"):
        if not (target / split / "images").exists():
            return False
    return True


def _verify_classes(yaml_path: Path) -> None:
    """Hard fail if class order doesn't match what the code assumes."""
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)
    got = cfg.get("names")
    if isinstance(got, dict):  # Roboflow sometimes exports {0: 'MH', ...}
        got = [got[k] for k in sorted(got)]
    if got != CLASS_NAMES:
        raise RuntimeError(
            f"Class order mismatch.\n"
            f"  expected: {CLASS_NAMES}\n"
            f"  got     : {got}\n"
            f"Fix Roboflow export order or update CLASS_NAMES in src/paths.py."
        )


def pull(force: bool = False) -> Path:
    target = RADS_DATA / f"{ROBOFLOW_PROJECT}-v{ROBOFLOW_VERSION}"

    if _existing_export(target) and not force:
        print(f"[cache] dataset already present at {target}")
        _verify_classes(target / "data.yaml")
        return target

    if force and target.exists():
        shutil.rmtree(target)

    if not ROBOFLOW_API_KEY:
        sys.exit("ROBOFLOW_API_KEY is not set — export it before running.")
    if not (ROBOFLOW_WORKSPACE and ROBOFLOW_PROJECT):
        sys.exit("Set ROBOFLOW_WORKSPACE and ROBOFLOW_PROJECT env vars.")

    # Lazy import — keeps `--help` fast even without roboflow installed.
    from roboflow import Roboflow

    print(f"[pull] {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION}")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download("yolov8", location=str(target), overwrite=True)

    yaml_path = Path(dataset.location) / "data.yaml"
    _verify_classes(yaml_path)
    print(f"[ok] dataset at {dataset.location}")
    return Path(dataset.location)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = ap.parse_args()
    pull(force=args.force)
