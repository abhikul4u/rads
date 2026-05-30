#!/bin/bash
# =============================================================================
# RADS Layer 3 — Model Development Pipeline
# status_check.sh — read-only health/progress dashboard for a pipeline run (v2)
#
# Author: Rutuja Kulkarni
#
# WHAT THIS SCRIPT DOES
#   Prints a one-shot, colour-coded snapshot of an in-progress or completed
#   full-pipeline run (see scripts/06_run_full_pipeline.sh). It reports, in
#   seven sections:
#     1. Pipeline activity  — is the tmux session / orchestrator / training alive
#     2. Hardware health    — GPU name/util/memory/temp and /workspace disk usage
#     3. Per-variant completion — how many of the 3 seeds are done per variant
#     4. Validation mAP per run — best mAP per finished run; live epoch+mAP for
#                                 the run that is currently training
#     5. Test-set comparison    — per-variant test metrics from results JSONs
#     6. Recent log activity    — how fresh artifacts/full_run.log is
#     7. Verdict                — overall progress % and a healthy/idle call
#
#   v2 NOTE (preserved from original): v1 mislabelled the actively-training run
#   as "stopped" and showed a stale "best epoch". v2 first detects which run is
#   training and reports its live epoch + work-in-progress validation mAP instead.
#
# WHEN TO RUN IT
#   Anytime during or after a run. It is strictly read-only (no weights, logs,
#   or results are modified), so it is safe to run as often as you like.
#
# ARGUMENTS / ENV
#   None. Paths are hard-coded to the standard /workspace/artifacts layout.
#   It expects the 6 variants × 3 seeds convention used by the orchestrator.
#
# HOW IT FITS THE PIPELINE
#   The operator's monitoring companion to 06_run_full_pipeline.sh — answers
#   "what's trained, what's left, is it healthy?" without touching the run.
# =============================================================================

# Only -u (error on unset vars); intentionally NOT -e, because many checks below
# are best-effort probes (ps/grep/nvidia-smi) whose non-zero exits are expected
# and handled explicitly — we never want a missing tool to abort the dashboard.
set -u

# Enable ANSI colours only when stdout is an interactive terminal ([ -t 1 ]);
# when piped/redirected, blank the codes so logs aren't polluted with escapes.
if [ -t 1 ]; then
    BLUE='\033[1;34m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; RESET='\033[0m'
else
    BLUE=''; GREEN=''; YELLOW=''; RED=''; RESET=''
fi

# The experiment matrix this run is expected to cover: 6 variants × 3 seeds = 18
# total trainings. EXPECTED_TOTAL drives the progress percentage in section 7.
EXPECTED_VARIANTS=("baseline" "cbam" "p2" "sizeaware" "combined" "distill")
EXPECTED_SEEDS=(42 1337 2024)
EXPECTED_TOTAL=$((${#EXPECTED_VARIANTS[@]} * ${#EXPECTED_SEEDS[@]}))
# Standard on-disk locations produced by the pipeline on the persistent volume.
RUNS_DIR="/workspace/artifacts/runs"
RESULTS_DIR="/workspace/artifacts/results"
LOG_FILE="/workspace/artifacts/full_run.log"

# Identify the currently-training run
# Inspect the process table for any running stage-01..04 training script. The
# regex 0[1-4]_ matches scripts/01_..04_; `grep -v grep` drops the grep process
# itself. Results are stashed in a temp file so we can parse them several ways.
CURRENTLY_TRAINING=""
CURRENT_VARIANT=""
CURRENT_SEED=""
if ps aux | grep -E "scripts/0[1-4]_" | grep -v grep > /tmp/active_procs.txt 2>/dev/null; then
    # Pull the --variant and --seed values straight off the live command line.
    # -oE prints only the matched flag+value; head -1 takes the first match;
    # awk '{print $2}' isolates the value after the flag name.
    CURRENT_VARIANT=$(grep -oE "\-\-variant [a-z_]+" /tmp/active_procs.txt | head -1 | awk '{print $2}')
    CURRENT_SEED=$(grep -oE "\-\-seed [0-9]+" /tmp/active_procs.txt | head -1 | awk '{print $2}')
    # baseline (01) and distill (03) scripts don't take a --variant flag, so if
    # none was found, infer the variant from which stage script is running.
    if [ -z "$CURRENT_VARIANT" ]; then
        if grep -q "01_train_baseline" /tmp/active_procs.txt; then
            CURRENT_VARIANT="baseline"
        elif grep -q "03_train_distill" /tmp/active_procs.txt; then
            CURRENT_VARIANT="distill"
        fi
    fi
    # Only declare a run "currently training" when we resolved BOTH the variant
    # and seed, so the run name matches the runs/<variant>_seed<seed> directory.
    if [ -n "$CURRENT_VARIANT" ] && [ -n "$CURRENT_SEED" ]; then
        CURRENTLY_TRAINING="${CURRENT_VARIANT}_seed${CURRENT_SEED}"
    fi
fi

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BLUE}║         RADS Layer 3 — Pipeline Status (v2)                      ║${RESET}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${RESET}"
# Banner: timestamp + pod hostname so a saved/pasted snapshot is self-identifying,
# plus the actively-training run (or a note that nothing is training right now).
echo "Time:     $(date)"
echo "Pod:      $(hostname)"
if [ -n "$CURRENTLY_TRAINING" ]; then
    echo -e "Active:   ${GREEN}$CURRENTLY_TRAINING${RESET}"
else
    echo -e "Active:   ${YELLOW}(none — between variants or pipeline done)${RESET}"
fi
echo ""

# --- 1. Pipeline activity ---
# Liveness check at three levels: the tmux session that hosts the run, the
# orchestrator script itself, and the count of active training subprocesses.
# Together these distinguish "running", "between variants", and "fully stopped".
echo -e "${BLUE}─── 1. Pipeline Activity ────────────────────────────────────────${RESET}"
# A live `rads-full` tmux session is the expected host for an unattended run.
if tmux has-session -t rads-full 2>/dev/null; then
    echo -e "  Tmux session:                    ${GREEN}✓ ALIVE${RESET}"
else
    echo -e "  Tmux session:                    ${YELLOW}○ ENDED${RESET}"
fi
# Count the orchestrator and the training subprocesses separately: the
# orchestrator can be alive while briefly between trainings (0 training procs),
# which is a normal, healthy state we want to distinguish from a crash.
PIPELINE_PROCS=$(ps aux | grep "06_run_full_pipeline.sh" | grep -v grep | wc -l)
TRAINING_PROCS=$(ps aux | grep -E "scripts/0[1-4]_" | grep -v grep | wc -l)
if [ "$PIPELINE_PROCS" -gt 0 ]; then
    echo -e "  Pipeline orchestrator:           ${GREEN}✓ RUNNING${RESET}"
else
    echo -e "  Pipeline orchestrator:           ${YELLOW}○ ENDED${RESET}"
fi
if [ "$TRAINING_PROCS" -gt 0 ]; then
    echo -e "  Training processes:              ${GREEN}$TRAINING_PROCS active${RESET}"
else
    echo -e "  Training processes:              ${YELLOW}○ NONE${RESET}"
fi
echo ""

# --- 2. Hardware ---
# Confirm the GPU is actually working (utilisation/temperature) and that the
# persistent volume isn't about to fill up — both are common silent killers of
# a long unattended run.
echo -e "${BLUE}─── 2. Hardware Health ──────────────────────────────────────────${RESET}"
# Only query the GPU if nvidia-smi exists (guards CPU-only / odd environments).
if command -v nvidia-smi &>/dev/null; then
    # Pull the key fields as a single CSV line (no header/units) for easy parsing;
    # head -1 selects the first GPU on multi-GPU pods.
    GPU_INFO=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$GPU_INFO" ]; then
        # Split the CSV into named fields using comma as the field separator.
        IFS=',' read -r gpu_name gpu_util gpu_mem_used gpu_mem_free gpu_temp <<< "$GPU_INFO"
        # Strip stray spaces so the numeric comparison below is valid.
        gpu_util_int=$(echo $gpu_util | tr -d ' ')
        echo "  GPU:                             $gpu_name"
        # >80% util = green (training hard); >0% = yellow; 0% = idle (suspicious
        # if a run is supposedly active).
        if [ "$gpu_util_int" -gt 80 ]; then
            echo -e "  GPU utilization:                 ${GREEN}${gpu_util}%${RESET}"
        elif [ "$gpu_util_int" -gt 0 ]; then
            echo -e "  GPU utilization:                 ${YELLOW}${gpu_util}%${RESET}"
        else
            echo -e "  GPU utilization:                 ${YELLOW}idle${RESET}"
        fi
        echo "  GPU memory:                      ${gpu_mem_used}/${gpu_mem_free} MiB used/free"
        echo "  GPU temperature:                 ${gpu_temp}°C"
    fi
fi
# Disk usage on the persistent volume: weights/exports accumulate fast, and a
# full /workspace will silently crash trainings. tail -1 drops the df header;
# awk pulls the use%, available, and total columns.
DISK_OUTPUT=$(df -h /workspace 2>/dev/null | tail -1)
if [ -n "$DISK_OUTPUT" ]; then
    DISK_USED_PCT=$(echo "$DISK_OUTPUT" | awk '{print $5}' | tr -d '%')
    DISK_AVAIL=$(echo "$DISK_OUTPUT" | awk '{print $4}')
    DISK_TOTAL=$(echo "$DISK_OUTPUT" | awk '{print $2}')
    # <70% used = green (comfortable); otherwise yellow as a fill-up warning.
    if [ "$DISK_USED_PCT" -lt 70 ]; then
        echo -e "  Disk /workspace:                 ${GREEN}${DISK_USED_PCT}% used${RESET}  ($DISK_AVAIL free of $DISK_TOTAL)"
    else
        echo -e "  Disk /workspace:                 ${YELLOW}${DISK_USED_PCT}% used${RESET}  ($DISK_AVAIL free of $DISK_TOTAL)"
    fi
fi
echo ""

# --- 3. Per-variant completion ---
# For each variant, classify its three seeds as done / active / partial-stale so
# the operator can see at a glance how far the matrix has progressed.
echo -e "${BLUE}─── 3. Per-Variant Completion ───────────────────────────────────${RESET}"
for variant in "${EXPECTED_VARIANTS[@]}"; do
    done_seeds=()
    active_seed=""
    for seed in "${EXPECTED_SEEDS[@]}"; do
        run="${variant}_seed${seed}"
        # Priority of checks matters: the live run is "active"; a saved best.pt
        # means "done"; a bare run dir with no best.pt is a partial/stale leftover
        # (e.g. a crashed or interrupted training).
        if [ "$run" = "$CURRENTLY_TRAINING" ]; then
            active_seed="$seed"
        elif [ -f "$RUNS_DIR/$run/weights/best.pt" ]; then
            done_seeds+=("$seed")
        elif [ -d "$RUNS_DIR/$run" ]; then
            active_seed="$seed (partial/stale)"
        fi
    done
    n_done=${#done_seeds[@]}
    # 3/3 = fully done; some done or one active = in progress; otherwise not started.
    if [ "$n_done" -eq 3 ]; then
        echo -e "  ${variant}:        ${GREEN}3/3 ✓ done${RESET}  (seeds: ${done_seeds[*]})"
    elif [ "$n_done" -gt 0 ] || [ -n "$active_seed" ]; then
        if [ -n "$active_seed" ]; then
            echo -e "  ${variant}:        ${YELLOW}${n_done}/3 done${RESET}, seed $active_seed active"
        else
            echo -e "  ${variant}:        ${YELLOW}${n_done}/3 done${RESET}"
        fi
    else
        echo -e "  ${variant}:        ○ not started"
    fi
done
echo ""

# --- 4. Val mAP per run ---
# Walk every run directory and report its validation mAP. For the live run we
# show the latest in-progress epoch/mAP; for finished runs we parse the full
# results.csv to find the best epoch. This is the section v2 specifically fixed.
echo -e "${BLUE}─── 4. Validation mAP per Run ───────────────────────────────────${RESET}"
for run in $(ls "$RUNS_DIR" 2>/dev/null | sort); do
    csv="$RUNS_DIR/$run/results.csv"
    bestpt="$RUNS_DIR/$run/weights/best.pt"

    # Live run: read the LAST row of results.csv for the current epoch (col 1)
    # and current val mAP50 (col 7) rather than a best-so-far — best.pt may not
    # exist yet and "best epoch" would be misleading mid-training.
    if [ "$run" = "$CURRENTLY_TRAINING" ]; then
        if [ -f "$csv" ]; then
            current_epoch=$(tail -1 "$csv" | awk -F',' '{print $1}' | tr -d ' ')
            current_map=$(tail -1 "$csv" | awk -F',' '{print $7}' | tr -d ' ')
            echo -e "  ${YELLOW}$run${RESET}  IN PROGRESS — epoch $current_epoch, current val mAP50=$current_map"
        fi
        continue
    fi

    # No best.pt on a non-live run => it never finished; flag a bare dir as a
    # likely crash and skip metric parsing.
    if [ ! -f "$bestpt" ]; then
        if [ -d "$RUNS_DIR/$run" ]; then
            echo -e "  ${YELLOW}$run${RESET}  partial (no best.pt — crashed?)"
        fi
        continue
    fi

    # best.pt exists but no metrics log to parse — note it and move on.
    if [ ! -f "$csv" ]; then
        echo "  $run  (no results.csv)"
        continue
    fi

    # Parse the metrics CSV in Python (robust column lookup by header name rather
    # than fragile fixed indices) to report best-epoch mAP50 / mAP50-95 and
    # whether the run early-stopped before the full 100 epochs.
    python3 << PYEOF
import csv as c
try:
    # Read the whole CSV; row 0 is the header, the rest are per-epoch rows.
    rows = list(c.reader(open('$csv')))
    header = [h.strip() for h in rows[0]]   # strip Ultralytics' padding spaces
    data = rows[1:]
    if not data:
        print(f"  $run  (empty csv)")
    else:
        # Resolve column positions by name so this survives column re-ordering.
        epoch_idx = header.index('epoch')
        map50_idx = header.index('metrics/mAP50(B)')
        map5095_idx = header.index('metrics/mAP50-95(B)')

        # Best epoch = the row with the highest mAP50 (treat blank cells as 0).
        best_row = max(data, key=lambda r: float(r[map50_idx]) if r[map50_idx].strip() else 0)
        best_epoch = int(best_row[epoch_idx])
        best_map50 = float(best_row[map50_idx])
        best_map5095 = float(best_row[map5095_idx])
        last_epoch = int(data[-1][epoch_idx])

        # If the final logged epoch is < the 100-epoch budget, the run early-stopped.
        if last_epoch < 100:
            stop_info = f"stopped at {last_epoch} (early-stop)"
        else:
            stop_info = "ran full 100"

        print(f"  $run  best@{best_epoch}: mAP50={best_map50:.4f} mAP50-95={best_map5095:.4f}  [{stop_info}]")
except Exception as e:
    # Never let one malformed CSV abort the whole dashboard — report and continue.
    print(f"  $run  parse error: {e}")
PYEOF
done
echo ""

# --- 5. Test-set comparison ---
# Render a side-by-side table of held-out TEST metrics (written by 05_evaluate.py
# as *_test.json). Columns: overall mAP50/mAP50-95 plus per-class AP for the three
# RADS classes — MH=manhole, PH=pothole, WLPH=water-logged pothole.
echo -e "${BLUE}─── 5. Test Set Comparison ──────────────────────────────────────${RESET}"
if [ -d "$RESULTS_DIR" ]; then
    test_count=$(ls "$RESULTS_DIR"/*_test.json 2>/dev/null | wc -l)
    if [ "$test_count" -gt 0 ]; then
        # Header rows for the fixed-width table (printf %-N left-aligns each column).
        printf "  %-25s %-9s %-12s %-9s %-9s %-9s\n" "Variant" "mAP50" "mAP50-95" "AP_MH" "AP_PH" "AP_WLPH"
        printf "  %-25s %-9s %-12s %-9s %-9s %-9s\n" "-------" "-----" "--------" "-----" "-----" "-------"
        for f in "$RESULTS_DIR"/*_test.json; do
            [ -f "$f" ] || continue           # skip if the glob matched nothing
            # Derive a clean display name: strip the .json and the _test suffix.
            name=$(basename "$f" .json | sed 's/_test//')
            # Pull the metrics out of the JSON and print one aligned table row;
            # .get(...,0) tolerates older JSONs that may lack a per-class key.
            python3 -c "
import json
d = json.load(open('$f'))
print(f'  {\"$name\":<25} {d.get(\"map50\", 0):<9.4f} {d.get(\"map50_95\", 0):<12.4f} {d.get(\"AP50_MH\", 0):<9.4f} {d.get(\"AP50_PH\", 0):<9.4f} {d.get(\"AP50_WLPH\", 0):<9.4f}')
" 2>/dev/null
        done
        echo ""
        echo "  Total test evaluations: $test_count / $EXPECTED_TOTAL"
    else
        echo "  No test results yet"
    fi
fi
echo ""

# --- 6. Log activity ---
# Use the log file's freshness as a heartbeat: how long since full_run.log was
# last written tells us whether the pipeline is actively producing output.
echo -e "${BLUE}─── 6. Recent Log Activity ──────────────────────────────────────${RESET}"
if [ -f "$LOG_FILE" ]; then
    # Seconds since last modification = now (date +%s) minus the file's mtime
    # (stat -c %Y). LOG_SIZE is a human-readable size for context.
    SECONDS_SINCE=$(( $(date +%s) - $(stat -c %Y "$LOG_FILE") ))
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
    # <60s = live; <1800s (30m) = quiet (plausibly between variants); else idle.
    if [ "$SECONDS_SINCE" -lt 60 ]; then
        echo -e "  Log ($LOG_SIZE):  ${GREEN}✓ Live (${SECONDS_SINCE}s ago)${RESET}"
    elif [ "$SECONDS_SINCE" -lt 1800 ]; then
        echo -e "  Log ($LOG_SIZE):  ${YELLOW}Quiet (${SECONDS_SINCE}s — between variants?)${RESET}"
    else
        echo -e "  Log ($LOG_SIZE):  ${YELLOW}Idle ($(($SECONDS_SINCE / 60)) min)${RESET}"
    fi
fi
echo ""

# --- 7. Verdict ---
# Roll everything up into a single progress %/health call. Completion is counted
# by the number of best.pt files on disk (one per finished training).
echo -e "${BLUE}─── 7. Verdict ──────────────────────────────────────────────────${RESET}"
COMPLETED=$(find "$RUNS_DIR" -name "best.pt" 2>/dev/null | wc -l)
# The currently-training run may already have written an interim best.pt, so
# subtract it to avoid counting an unfinished run as complete.
if [ -n "$CURRENTLY_TRAINING" ]; then
    COMPLETED_NOW=$((COMPLETED - 1))
else
    COMPLETED_NOW=$COMPLETED
fi
# Clamp to >=0 (guards the edge case where the subtraction underflows).
[ "$COMPLETED_NOW" -lt 0 ] && COMPLETED_NOW=0
PROGRESS_PCT=$((COMPLETED_NOW * 100 / EXPECTED_TOTAL))

# Three verdicts: all done; still training (healthy); or no active training
# (stalled/finished-with-gaps — worth a manual look).
if [ "$COMPLETED_NOW" -eq "$EXPECTED_TOTAL" ]; then
    echo -e "  ${GREEN}🎉 FULL RUN COMPLETE!${RESET} All $EXPECTED_TOTAL trainings done."
elif [ "$TRAINING_PROCS" -gt 0 ]; then
    echo -e "  ${GREEN}✓ Healthy & progressing${RESET}"
    echo "     $COMPLETED_NOW/$EXPECTED_TOTAL trainings complete ($PROGRESS_PCT%)"
    [ -n "$CURRENTLY_TRAINING" ] && echo "     Currently: $CURRENTLY_TRAINING"
else
    echo -e "  ${YELLOW}⚠ No active training${RESET}"
    echo "     $COMPLETED_NOW/$EXPECTED_TOTAL complete. Check tmux + log."
fi
echo ""
# Handy copy-paste commands for digging deeper than this snapshot allows.
echo "Quick actions:"
echo "  tmux attach -t rads-full   # see live (Ctrl+B D to detach)"
echo "  tail -f $LOG_FILE          # follow log (Ctrl+C to exit)"
echo "  W&B:  https://wandb.ai/abhikul4u-hvpm-college-of-engineering-technology-amravati"
echo ""