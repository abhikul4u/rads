"""Pull the RADS dataset from Roboflow and cache it on the persistent volume.

Author: Rutuja Kulkarni

This is the data-acquisition entry point of the RADS Layer 3 pipeline. It is the
very first stage that must run on a fresh RunPod pod: every later stage —
baseline training, the CBAM / P2 / size-aware-loss ablations, the knowledge
distillation teacher/student runs, and the final evaluation — consumes the
YOLOv8-format dataset that this script materialises under `RADS_DATA`. By
caching the export on the persistent network volume and short-circuiting on a
valid existing copy, repeated pod restarts do not re-download gigabytes of
imagery, while `--force` still allows a deliberate refresh when the Roboflow
version is bumped.

Idempotent: if a valid YOLOv8 export already exists at the target path, we skip
the download. Force a refresh with --force.

Verifies the class ordering matches `CLASS_NAMES` in src/paths.py — a silent
class re-ordering by Roboflow would invalidate every downstream run, because
YOLO label files store the class as a bare integer index into the names list.
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
    """Decide whether a usable YOLOv8 dataset already lives at `target`.

    This is the cache-validity check that makes the pull idempotent. Rather than
    trusting that the directory merely exists, it confirms the export is
    structurally complete so a half-finished/aborted prior download is never
    mistaken for a good cache.

    Args:
        target: directory that should contain the YOLOv8 export.

    Returns:
        True only if `data.yaml` is present AND each of the train/valid/test
        splits has an `images` subdirectory; False otherwise.

    Side effects:
        None (read-only filesystem probing).
    """
    yaml_path = target / "data.yaml"
    # data.yaml is the manifest Ultralytics reads; without it the export is unusable.
    if not yaml_path.exists():
        return False
    # A complete YOLOv8 export always ships all three splits with image folders;
    # a missing split indicates a partial/corrupt download we should re-fetch.
    for split in ("train", "valid", "test"):
        if not (target / split / "images").exists():
            return False
    return True


def _verify_classes(yaml_path: Path) -> None:
    """Assert the dataset's class ordering matches `CLASS_NAMES` exactly.

    This is the class-order lock that protects the entire pipeline. YOLO label
    files identify a class purely by its integer position in the names list, so
    if Roboflow ever exports the classes in a different order (or with renamed
    entries), every annotation would be silently mislabelled and all downstream
    metrics would be meaningless. Failing loudly here is therefore preferable to
    training on quietly corrupted labels.

    Args:
        yaml_path: path to the export's `data.yaml`.

    Returns:
        None.

    Raises:
        RuntimeError: if the parsed class list differs in any way (order or
            contents) from the canonical `CLASS_NAMES`.
    """
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)
    got = cfg.get("names")
    # Roboflow is inconsistent: some exports use a list ['MH', ...], others a
    # dict {0: 'MH', 1: 'PH', ...}. Normalise the dict form to an ordered list
    # by sorting on the integer keys so the comparison below is apples-to-apples.
    if isinstance(got, dict):  # Roboflow sometimes exports {0: 'MH', ...}
        got = [got[k] for k in sorted(got)]
    # Exact equality check: order AND values must match the locked class list.
    if got != CLASS_NAMES:
        raise RuntimeError(
            f"Class order mismatch.\n"
            f"  expected: {CLASS_NAMES}\n"
            f"  got     : {got}\n"
            f"Fix Roboflow export order or update CLASS_NAMES in src/paths.py."
        )


def pull(force: bool = False) -> Path:
    """Ensure the RADS dataset is present locally and return its directory.

    Orchestrates the cache-or-download flow: it computes a version-pinned target
    path, returns immediately if a valid cached export is found (after still
    re-verifying class order, since the lock is cheap and catches a manually
    edited cache), and otherwise downloads a fresh YOLOv8 export from Roboflow.

    Args:
        force: if True, delete any existing export and re-download from Roboflow
            even when a valid cache exists. Used when the dataset version or
            annotations have changed upstream.

    Returns:
        Path to the directory containing the ready-to-use YOLOv8 export.

    Side effects:
        - May delete an existing dataset directory (when `force`).
        - May download gigabytes of imagery into `RADS_DATA`.
        - Prints progress/status lines.
        - Calls `sys.exit` (terminates the process) if required Roboflow
          credentials/coordinates are missing.
    """
    # Target is version-pinned so multiple dataset versions can coexist on disk
    # and so the cache key changes automatically when ROBOFLOW_VERSION is bumped.
    target = RADS_DATA / f"{ROBOFLOW_PROJECT}-v{ROBOFLOW_VERSION}"

    # Fast path: a valid cache exists and the caller did not request a refresh.
    if _existing_export(target) and not force:
        print(f"[cache] dataset already present at {target}")
        # Still verify the lock — guards against a hand-edited/stale cache.
        _verify_classes(target / "data.yaml")
        return target

    # Forced refresh: wipe the stale copy so the download starts from clean state.
    if force and target.exists():
        shutil.rmtree(target)

    # Fail fast with clear guidance if the required credentials are absent,
    # rather than letting the Roboflow SDK raise an opaque error mid-download.
    if not ROBOFLOW_API_KEY:
        sys.exit("ROBOFLOW_API_KEY is not set — export it before running.")
    if not (ROBOFLOW_WORKSPACE and ROBOFLOW_PROJECT):
        sys.exit("Set ROBOFLOW_WORKSPACE and ROBOFLOW_PROJECT env vars.")

    # Lazy import — keeps `--help` fast even without roboflow installed.
    # (Importing roboflow is slow and pulls heavy deps; we only need it on the
    # actual download path, never for the cached path or argument parsing.)
    from roboflow import Roboflow

    print(f"[pull] {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION}")
    # Authenticate and drill down to the exact pinned dataset version.
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    # Download in YOLOv8 layout straight into our version-pinned target dir;
    # overwrite=True lets Roboflow repopulate cleanly after the force-rmtree above.
    dataset = version.download("yolov8", location=str(target), overwrite=True)

    # Verify the freshly downloaded export's class order before declaring success,
    # using the location the SDK actually reports (it may normalise the path).
    yaml_path = Path(dataset.location) / "data.yaml"
    _verify_classes(yaml_path)
    print(f"[ok] dataset at {dataset.location}")
    return Path(dataset.location)


if __name__ == "__main__":
    # CLI wrapper so the data pull can be run standalone (e.g. as the first step
    # of the RunPod setup) or imported and called programmatically via pull().
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = ap.parse_args()
    pull(force=args.force)
