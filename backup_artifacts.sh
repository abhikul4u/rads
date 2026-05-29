#!/bin/bash
# backup_artifacts.sh — Snapshot RADS Layer 3 artifacts into a single tarball
#
# Creates:  /workspace/rads_full_artifacts_<timestamp>.tar.gz
#
# Includes:  results/ (JSONs, audits), runs/ (best.pt, csvs, plots, tensorboard,
#            configs), exports/ (if present)
# Excludes:  last.pt (duplicates best.pt), __pycache__
#
# Safe to run during training — only reads files; doesn't disturb the pipeline.
# Run anytime you want a fresh snapshot.
#
# Usage:
#   bash backup_artifacts.sh                  # default settings
#   bash backup_artifacts.sh --include-last   # also include last.pt files
#   bash backup_artifacts.sh --skip-weights   # exclude all .pt files (small archive)
#   bash backup_artifacts.sh --output PATH    # custom output path

set -u  # error on undefined vars (no -e — we handle errors)

# === Defaults ===
INCLUDE_LAST=0
SKIP_WEIGHTS=0
OUTPUT_DIR="/workspace"
ARTIFACTS_DIR="/workspace/artifacts"

# === Parse args ===
while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-last)  INCLUDE_LAST=1; shift ;;
        --skip-weights)  SKIP_WEIGHTS=1; shift ;;
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,17p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Try: bash backup_artifacts.sh --help" >&2
            exit 2
            ;;
    esac
done

# === Colors ===
if [ -t 1 ]; then
    BLUE='\033[1;34m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; RESET='\033[0m'
else
    BLUE=''; GREEN=''; YELLOW=''; RED=''; RESET=''
fi

# === Sanity checks ===
if [ ! -d "$ARTIFACTS_DIR" ]; then
    echo -e "${RED}ERROR: $ARTIFACTS_DIR does not exist${RESET}" >&2
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${RED}ERROR: output dir $OUTPUT_DIR does not exist${RESET}" >&2
    exit 1
fi

# === Build archive name ===
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
echo -e "${BLUE}─── 1. Source inventory ─────────────────────────────────────────${RESET}"
echo "Subdirectories:"
du -sh "$ARTIFACTS_DIR"/*/ 2>/dev/null | sed 's/^/  /'
echo ""
TOTAL_RAW=$(du -sh "$ARTIFACTS_DIR" 2>/dev/null | cut -f1)
echo "Total raw size: $TOTAL_RAW"
echo ""

WEIGHTS_SIZE_BYTES=$(find "$ARTIFACTS_DIR" -name "*.pt" -exec stat -c '%s' {} + 2>/dev/null | awk '{s+=$1} END {print s+0}')
WEIGHTS_SIZE_HUMAN=$(numfmt --to=iec --suffix=B "$WEIGHTS_SIZE_BYTES" 2>/dev/null || echo "${WEIGHTS_SIZE_BYTES} bytes")
N_BEST_PT=$(find "$ARTIFACTS_DIR" -name "best.pt" 2>/dev/null | wc -l)
N_LAST_PT=$(find "$ARTIFACTS_DIR" -name "last.pt" 2>/dev/null | wc -l)
echo "Weights count:    ${N_BEST_PT} best.pt + ${N_LAST_PT} last.pt (${WEIGHTS_SIZE_HUMAN} total)"
echo ""

# === Disk space check ===
FREE_KB=$(df /workspace | tail -1 | awk '{print $4}')
FREE_GB=$((FREE_KB / 1024 / 1024))
echo -e "${BLUE}─── 2. Disk space ───────────────────────────────────────────────${RESET}"
df -h /workspace | tail -1 | awk '{print "  Free on /workspace: " $4 " of " $2}'
if [ "$FREE_GB" -lt 5 ]; then
    echo -e "  ${RED}WARNING: Less than 5 GB free — archive may fail${RESET}"
fi
echo ""

# === Build exclude flags ===
EXCLUDES=(
    "--exclude=__pycache__"
    "--exclude=*.tmp"
    "--exclude=.cache"
)

if [ "$INCLUDE_LAST" -eq 0 ]; then
    EXCLUDES+=("--exclude=*/last.pt")
fi

if [ "$SKIP_WEIGHTS" -eq 1 ]; then
    EXCLUDES+=("--exclude=*.pt")
fi

# === Identify subdirs to include ===
INCLUDE_DIRS=()
for d in results runs exports; do
    if [ -d "$ARTIFACTS_DIR/$d" ]; then
        INCLUDE_DIRS+=("$d")
    fi
done

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
tar "${EXCLUDES[@]}" \
    -czf "$ARCHIVE_PATH" \
    -C "$ARTIFACTS_DIR" \
    "${INCLUDE_DIRS[@]}" \
    2>/tmp/backup_tar_err.log
TAR_EXIT=$?

END_SEC=$(date +%s)
ELAPSED=$((END_SEC - START_SEC))

if [ "$TAR_EXIT" -ne 0 ]; then
    echo -e "${RED}  ✗ tar failed (exit $TAR_EXIT)${RESET}"
    echo "  Last errors:"
    tail -10 /tmp/backup_tar_err.log | sed 's/^/    /'
    exit 1
fi

echo -e "${GREEN}  ✓ Archive created in ${ELAPSED}s${RESET}"
echo ""

# === Verify ===
echo -e "${BLUE}─── 5. Verification ─────────────────────────────────────────────${RESET}"
if [ ! -f "$ARCHIVE_PATH" ]; then
    echo -e "${RED}  ✗ Archive file not found${RESET}"
    exit 1
fi

ARCHIVE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)
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
echo "  Running integrity check..."
if tar -tzf "$ARCHIVE_PATH" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓ Integrity OK${RESET}"
else
    echo -e "  ${RED}✗ Integrity check FAILED — re-run the script${RESET}"
    exit 1
fi
echo ""

# === Next steps ===
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
