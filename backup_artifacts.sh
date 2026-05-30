#!/bin/bash
# =============================================================================
# RADS Layer 3 — Model Development Pipeline
# backup_artifacts.sh — Snapshot the artifacts/ tree into a single tarball
#
# Author: Rutuja Kulkarni
#
# WHAT THIS SCRIPT DOES
#   Bundles the pipeline's output tree (/workspace/artifacts) into a single
#   timestamped gzip tarball so the trained weights, evaluation results, and
#   exports can be downloaded off the ephemeral RunPod pod for safekeeping and
#   for use in the thesis. The flow is: parse args -> sanity-check paths ->
#   inventory the source -> check free disk -> build the tar -> verify it ->
#   print download/cleanup instructions.
#
#   Creates:  /workspace/rads_full_artifacts_<timestamp>.tar.gz
#   Includes: results/ (JSONs, audits), runs/ (best.pt, csvs, plots,
#             tensorboard, configs), exports/ (if present)
#   Excludes: last.pt (duplicates best.pt), __pycache__, *.tmp, .cache
#
# WHEN TO RUN IT
#   Anytime you want a fresh snapshot. It is read-only with respect to the
#   artifacts (it only reads them to tar), so it is safe to run while training
#   is still in progress.
#
# ARGUMENTS IT HONORS
#   --include-last   also include last.pt files (normally excluded as redundant)
#   --skip-weights   exclude ALL .pt files -> small metadata/plots-only archive
#   --output PATH    directory to write the tarball into (default /workspace)
#   -h | --help      print the usage block (lines 2..17 of this file)
#
# HOW IT FITS THE PIPELINE
#   Run after (or during) 06_run_full_pipeline.sh to preserve results before the
#   pod is torn down. collect_run_figures.sh is the figures-only counterpart.
#
# Usage:
#   bash backup_artifacts.sh                  # default settings
#   bash backup_artifacts.sh --include-last   # also include last.pt files
#   bash backup_artifacts.sh --skip-weights   # exclude all .pt files (small archive)
#   bash backup_artifacts.sh --output PATH    # custom output path
# =============================================================================

# Only -u here: we deliberately do NOT use -e because failures (e.g. a bad tar)
# are detected and reported explicitly below with tailored messages/exit codes.
set -u  # error on undefined vars (no -e — we handle errors)

# === Defaults ===
# Behaviour toggles (flipped by the flags below) and the source/destination dirs.
INCLUDE_LAST=0
SKIP_WEIGHTS=0
OUTPUT_DIR="/workspace"
ARTIFACTS_DIR="/workspace/artifacts"

# === Parse args ===
# Hand-rolled flag parser: walk $@ one token at a time. `shift` consumes a flag;
# `shift 2` consumes a flag plus its value (--output PATH).
while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-last)  INCLUDE_LAST=1; shift ;;
        --skip-weights)  SKIP_WEIGHTS=1; shift ;;
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            # Print the human usage block (this file's lines 2..17) and exit OK.
            sed -n '2,17p' "$0"
            exit 0
            ;;
        *)
            # Anything unrecognised is a usage error (exit 2, message to stderr).
            echo "Unknown option: $1" >&2
            echo "Try: bash backup_artifacts.sh --help" >&2
            exit 2
            ;;
    esac
done

# === Colors ===
# ANSI colours only on an interactive terminal; otherwise blanked so redirected
# output stays clean.
if [ -t 1 ]; then
    BLUE='\033[1;34m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; RESET='\033[0m'
else
    BLUE=''; GREEN=''; YELLOW=''; RED=''; RESET=''
fi

# === Sanity checks ===
# Bail early (with a clear message) if there is nothing to back up or nowhere to
# write — cheaper than discovering it mid-tar.
if [ ! -d "$ARTIFACTS_DIR" ]; then
    echo -e "${RED}ERROR: $ARTIFACTS_DIR does not exist${RESET}" >&2
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${RED}ERROR: output dir $OUTPUT_DIR does not exist${RESET}" >&2
    exit 1
fi

# === Build archive name ===
# Timestamp the filename (YYYYMMDD_HHMMSS) so repeated backups never collide and
# sort chronologically.
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="rads_full_artifacts_${TIMESTAMP}.tar.gz"
ARCHIVE_PATH="${OUTPUT_DIR}/${ARCHIVE_NAME}"

# === Pre-flight summary ===
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BLUE}║         RADS — Artifact Backup Script                            ║${RESET}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${RESET}"
echo "Source:   $ARTIFACTS_DIR"
echo "Output:   $ARCHIVE_PATH"
echo ""

# === Pre-archive disk audit ===
# Show what is about to be archived and how big it is, so the operator can sanity
# check the result and anticipate the archive size / transfer time.
echo -e "${BLUE}─── 1. Source inventory ─────────────────────────────────────────${RESET}"
echo "Subdirectories:"
# Per-subdir sizes, indented two spaces via sed for readability.
du -sh "$ARTIFACTS_DIR"/*/ 2>/dev/null | sed 's/^/  /'
echo ""
TOTAL_RAW=$(du -sh "$ARTIFACTS_DIR" 2>/dev/null | cut -f1)
echo "Total raw size: $TOTAL_RAW"
echo ""

# Sum the byte sizes of all .pt files (find ... -exec stat | awk accumulator),
# then humanise with numfmt (falling back to raw bytes if numfmt is absent), and
# count best.pt vs last.pt so the weights footprint is explicit.
WEIGHTS_SIZE_BYTES=$(find "$ARTIFACTS_DIR" -name "*.pt" -exec stat -c '%s' {} + 2>/dev/null | awk '{s+=$1} END {print s+0}')
WEIGHTS_SIZE_HUMAN=$(numfmt --to=iec --suffix=B "$WEIGHTS_SIZE_BYTES" 2>/dev/null || echo "${WEIGHTS_SIZE_BYTES} bytes")
N_BEST_PT=$(find "$ARTIFACTS_DIR" -name "best.pt" 2>/dev/null | wc -l)
N_LAST_PT=$(find "$ARTIFACTS_DIR" -name "last.pt" 2>/dev/null | wc -l)
echo "Weights count:    ${N_BEST_PT} best.pt + ${N_LAST_PT} last.pt (${WEIGHTS_SIZE_HUMAN} total)"
echo ""

# === Disk space check ===
# Warn if free space on /workspace is tight, since the gzip tarball is written
# alongside the source and could otherwise fail partway through.
FREE_KB=$(df /workspace | tail -1 | awk '{print $4}')
FREE_GB=$((FREE_KB / 1024 / 1024))   # KB -> GB (df reports 1K blocks)
echo -e "${BLUE}─── 2. Disk space ───────────────────────────────────────────────${RESET}"
df -h /workspace | tail -1 | awk '{print "  Free on /workspace: " $4 " of " $2}'
if [ "$FREE_GB" -lt 5 ]; then
    echo -e "  ${RED}WARNING: Less than 5 GB free — archive may fail${RESET}"
fi
echo ""

# === Build exclude flags ===
# Always drop noise that bloats the archive without value (bytecode caches, temp
# and cache files). These become tar --exclude args later.
EXCLUDES=(
    "--exclude=__pycache__"
    "--exclude=*.tmp"
    "--exclude=.cache"
)

# By default exclude last.pt (the final-epoch checkpoint) because best.pt already
# captures the model we keep; --include-last opts back in.
if [ "$INCLUDE_LAST" -eq 0 ]; then
    EXCLUDES+=("--exclude=*/last.pt")
fi

# --skip-weights drops every .pt to produce a tiny metadata/plots-only archive.
if [ "$SKIP_WEIGHTS" -eq 1 ]; then
    EXCLUDES+=("--exclude=*.pt")
fi

# === Identify subdirs to include ===
# Only archive the subdirs that actually exist (avoids tar errors on, e.g., a run
# that never produced exports/). Iterating a fixed list keeps the archive layout
# predictable.
INCLUDE_DIRS=()
for d in results runs exports; do
    if [ -d "$ARTIFACTS_DIR/$d" ]; then
        INCLUDE_DIRS+=("$d")
    fi
done

# Nothing to archive => fail clearly rather than emit an empty tarball.
if [ ${#INCLUDE_DIRS[@]} -eq 0 ]; then
    echo -e "${RED}ERROR: no subdirectories found in $ARTIFACTS_DIR${RESET}" >&2
    exit 1
fi

# === Show config ===
echo -e "${BLUE}─── 3. Archive configuration ────────────────────────────────────${RESET}"
echo "  Including subdirs:  ${INCLUDE_DIRS[*]}"
echo "  Excluding:          last.pt files: $([ $INCLUDE_LAST -eq 1 ] && echo NO || echo YES)"
echo "                      all .pt files: $([ $SKIP_WEIGHTS -eq 1 ] && echo YES || echo NO)"
echo ""

# === Build the archive ===
echo -e "${BLUE}─── 4. Creating archive ─────────────────────────────────────────${RESET}"
echo "  This may take a few minutes for multi-GB archives..."
echo "  Started at $(date +%H:%M:%S)"

START_SEC=$(date +%s)

# Run tar; capture exit status
# Flags: the EXCLUDES array expands to the --exclude patterns; -c create,
# -z gzip, -f write to file; -C changes into ARTIFACTS_DIR first so the archive
# stores relative paths (results/..., runs/...) rather than the absolute tree.
# stderr is redirected to a log so we can show only the tail on failure.
tar "${EXCLUDES[@]}" \
    -czf "$ARCHIVE_PATH" \
    -C "$ARTIFACTS_DIR" \
    "${INCLUDE_DIRS[@]}" \
    2>/tmp/backup_tar_err.log
TAR_EXIT=$?   # capture immediately — any later command would overwrite $?

END_SEC=$(date +%s)
ELAPSED=$((END_SEC - START_SEC))

# tar failed: surface the last few stderr lines and abort with non-zero status.
if [ "$TAR_EXIT" -ne 0 ]; then
    echo -e "${RED}  ✗ tar failed (exit $TAR_EXIT)${RESET}"
    echo "  Last errors:"
    tail -10 /tmp/backup_tar_err.log | sed 's/^/    /'
    exit 1
fi

echo -e "${GREEN}  ✓ Archive created in ${ELAPSED}s${RESET}"
echo ""

# === Verify ===
# Confirm the archive exists, report its size/entry count, preview its contents,
# and finally re-read it end-to-end as an integrity check before we trust it.
echo -e "${BLUE}─── 5. Verification ─────────────────────────────────────────────${RESET}"
if [ ! -f "$ARCHIVE_PATH" ]; then
    echo -e "${RED}  ✗ Archive file not found${RESET}"
    exit 1
fi

ARCHIVE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)
# tar -t lists entries without extracting; counting them confirms it's non-empty.
N_FILES=$(tar -tzf "$ARCHIVE_PATH" 2>/dev/null | wc -l)

echo "  File:        $ARCHIVE_PATH"
echo "  Size:        $ARCHIVE_SIZE"
echo "  File count:  $N_FILES entries"
echo ""

echo "  First 15 entries:"
tar -tzf "$ARCHIVE_PATH" 2>/dev/null | head -15 | sed 's/^/    /'
echo "    ..."
echo ""

# === Quick integrity check ===
# Stream the whole gzip through tar (output discarded) — if it decompresses and
# lists without error, the archive is structurally sound.
echo "  Running integrity check..."
if tar -tzf "$ARCHIVE_PATH" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓ Integrity OK${RESET}"
else
    echo -e "  ${RED}✗ Integrity check FAILED — re-run the script${RESET}"
    exit 1
fi
echo ""

# === Next steps ===
# Operator guidance: how to download the tarball off the pod, verify it locally,
# and reclaim pod disk afterwards.
echo -e "${BLUE}─── 6. Next steps ───────────────────────────────────────────────${RESET}"
echo "  1. Download via Jupyter Lab:"
echo "       RunPod dashboard → Connect → Open Jupyter Lab"
echo "       Navigate to /workspace/"
echo "       Right-click '$ARCHIVE_NAME' → Download"
echo ""
echo "  2. Verify on laptop (PowerShell):"
echo "       cd C:\\D-Drive\\Abhikul\\FinalProject\\backup"
echo "       dir $ARCHIVE_NAME"
echo "       tar -tzf $ARCHIVE_NAME | Select-Object -First 30"
echo ""
echo "  3. After verified download, clean up pod (saves disk):"
echo "       rm $ARCHIVE_PATH"
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}  Backup complete: $ARCHIVE_PATH${RESET}"
echo -e "${GREEN}══════════════════════════════════════════════════════════════════${RESET}"
echo ""
