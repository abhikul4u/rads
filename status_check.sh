#!/bin/bash
# RADS Layer 3 — Pipeline Status Check (v2)
# Fixes v1 bug: in-progress runs were reported as "stopped" with stale "best epoch"
# Now: detects which run is actively training and reports its current epoch + WIP val mAP
# Safe to run repeatedly — read-only.

set -u

if [ -t 1 ]; then
    BLUE='\033[1;34m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; RESET='\033[0m'
else
    BLUE=''; GREEN=''; YELLOW=''; RED=''; RESET=''
fi

EXPECTED_VARIANTS=("baseline" "cbam" "p2" "sizeaware" "combined" "distill")
EXPECTED_SEEDS=(42 1337 2024)
EXPECTED_TOTAL=$((${#EXPECTED_VARIANTS[@]} * ${#EXPECTED_SEEDS[@]}))
RUNS_DIR="/workspace/artifacts/runs"
RESULTS_DIR="/workspace/artifacts/results"
LOG_FILE="/workspace/artifacts/full_run.log"

# Identify the currently-training run
CURRENTLY_TRAINING=""
CURRENT_VARIANT=""
CURRENT_SEED=""
if ps aux | grep -E "scripts/0[1-4]_" | grep -v grep > /tmp/active_procs.txt 2>/dev/null; then
    CURRENT_VARIANT=$(grep -oE "\-\-variant [a-z_]+" /tmp/active_procs.txt | head -1 | awk '{print $2}')
    CURRENT_SEED=$(grep -oE "\-\-seed [0-9]+" /tmp/active_procs.txt | head -1 | awk '{print $2}')
    if [ -z "$CURRENT_VARIANT" ]; then
        if grep -q "01_train_baseline" /tmp/active_procs.txt; then
            CURRENT_VARIANT="baseline"
        elif grep -q "03_train_distill" /tmp/active_procs.txt; then
            CURRENT_VARIANT="distill"
        fi
    fi
    if [ -n "$CURRENT_VARIANT" ] && [ -n "$CURRENT_SEED" ]; then
        CURRENTLY_TRAINING="${CURRENT_VARIANT}_seed${CURRENT_SEED}"
    fi
fi

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BLUE}║         RADS Layer 3 — Pipeline Status (v2)                      ║${RESET}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${RESET}"
echo "Time:     $(date)"
echo "Pod:      $(hostname)"
if [ -n "$CURRENTLY_TRAINING" ]; then
    echo -e "Active:   ${GREEN}$CURRENTLY_TRAINING${RESET}"
else
    echo -e "Active:   ${YELLOW}(none — between variants or pipeline done)${RESET}"
fi
echo ""

# --- 1. Pipeline activity ---
echo -e "${BLUE}─── 1. Pipeline Activity ────────────────────────────────────────${RESET}"
if tmux has-session -t rads-full 2>/dev/null; then
    echo -e "  Tmux session:                    ${GREEN}✓ ALIVE${RESET}"
else
    echo -e "  Tmux session:                    ${YELLOW}○ ENDED${RESET}"
fi
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
echo -e "${BLUE}─── 2. Hardware Health ──────────────────────────────────────────${RESET}"
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$GPU_INFO" ]; then
        IFS=',' read -r gpu_name gpu_util gpu_mem_used gpu_mem_free gpu_temp <<< "$GPU_INFO"
        gpu_util_int=$(echo $gpu_util | tr -d ' ')
        echo "  GPU:                             $gpu_name"
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
DISK_OUTPUT=$(df -h /workspace 2>/dev/null | tail -1)
if [ -n "$DISK_OUTPUT" ]; then
    DISK_USED_PCT=$(echo "$DISK_OUTPUT" | awk '{print $5}' | tr -d '%')
    DISK_AVAIL=$(echo "$DISK_OUTPUT" | awk '{print $4}')
    DISK_TOTAL=$(echo "$DISK_OUTPUT" | awk '{print $2}')
    if [ "$DISK_USED_PCT" -lt 70 ]; then
        echo -e "  Disk /workspace:                 ${GREEN}${DISK_USED_PCT}% used${RESET}  ($DISK_AVAIL free of $DISK_TOTAL)"
    else
        echo -e "  Disk /workspace:                 ${YELLOW}${DISK_USED_PCT}% used${RESET}  ($DISK_AVAIL free of $DISK_TOTAL)"
    fi
fi
echo ""

# --- 3. Per-variant completion ---
echo -e "${BLUE}─── 3. Per-Variant Completion ───────────────────────────────────${RESET}"
for variant in "${EXPECTED_VARIANTS[@]}"; do
    done_seeds=()
    active_seed=""
    for seed in "${EXPECTED_SEEDS[@]}"; do
        run="${variant}_seed${seed}"
        if [ "$run" = "$CURRENTLY_TRAINING" ]; then
            active_seed="$seed"
        elif [ -f "$RUNS_DIR/$run/weights/best.pt" ]; then
            done_seeds+=("$seed")
        elif [ -d "$RUNS_DIR/$run" ]; then
            active_seed="$seed (partial/stale)"
        fi
    done
    n_done=${#done_seeds[@]}
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
echo -e "${BLUE}─── 4. Validation mAP per Run ───────────────────────────────────${RESET}"
for run in $(ls "$RUNS_DIR" 2>/dev/null | sort); do
    csv="$RUNS_DIR/$run/results.csv"
    bestpt="$RUNS_DIR/$run/weights/best.pt"

    if [ "$run" = "$CURRENTLY_TRAINING" ]; then
        if [ -f "$csv" ]; then
            current_epoch=$(tail -1 "$csv" | awk -F',' '{print $1}' | tr -d ' ')
            current_map=$(tail -1 "$csv" | awk -F',' '{print $7}' | tr -d ' ')
            echo -e "  ${YELLOW}$run${RESET}  IN PROGRESS — epoch $current_epoch, current val mAP50=$current_map"
        fi
        continue
    fi

    if [ ! -f "$bestpt" ]; then
        if [ -d "$RUNS_DIR/$run" ]; then
            echo -e "  ${YELLOW}$run${RESET}  partial (no best.pt — crashed?)"
        fi
        continue
    fi

    if [ ! -f "$csv" ]; then
        echo "  $run  (no results.csv)"
        continue
    fi

    python3 << PYEOF
import csv as c
try:
    rows = list(c.reader(open('$csv')))
    header = [h.strip() for h in rows[0]]
    data = rows[1:]
    if not data:
        print(f"  $run  (empty csv)")
    else:
        epoch_idx = header.index('epoch')
        map50_idx = header.index('metrics/mAP50(B)')
        map5095_idx = header.index('metrics/mAP50-95(B)')

        best_row = max(data, key=lambda r: float(r[map50_idx]) if r[map50_idx].strip() else 0)
        best_epoch = int(best_row[epoch_idx])
        best_map50 = float(best_row[map50_idx])
        best_map5095 = float(best_row[map5095_idx])
        last_epoch = int(data[-1][epoch_idx])

        if last_epoch < 100:
            stop_info = f"stopped at {last_epoch} (early-stop)"
        else:
            stop_info = "ran full 100"

        print(f"  $run  best@{best_epoch}: mAP50={best_map50:.4f} mAP50-95={best_map5095:.4f}  [{stop_info}]")
except Exception as e:
    print(f"  $run  parse error: {e}")
PYEOF
done
echo ""

# --- 5. Test-set comparison ---
echo -e "${BLUE}─── 5. Test Set Comparison ──────────────────────────────────────${RESET}"
if [ -d "$RESULTS_DIR" ]; then
    test_count=$(ls "$RESULTS_DIR"/*_test.json 2>/dev/null | wc -l)
    if [ "$test_count" -gt 0 ]; then
        printf "  %-25s %-9s %-12s %-9s %-9s %-9s\n" "Variant" "mAP50" "mAP50-95" "AP_MH" "AP_PH" "AP_WLPH"
        printf "  %-25s %-9s %-12s %-9s %-9s %-9s\n" "-------" "-----" "--------" "-----" "-----" "-------"
        for f in "$RESULTS_DIR"/*_test.json; do
            [ -f "$f" ] || continue
            name=$(basename "$f" .json | sed 's/_test//')
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
echo -e "${BLUE}─── 6. Recent Log Activity ──────────────────────────────────────${RESET}"
if [ -f "$LOG_FILE" ]; then
    SECONDS_SINCE=$(( $(date +%s) - $(stat -c %Y "$LOG_FILE") ))
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
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
echo -e "${BLUE}─── 7. Verdict ──────────────────────────────────────────────────${RESET}"
COMPLETED=$(find "$RUNS_DIR" -name "best.pt" 2>/dev/null | wc -l)
if [ -n "$CURRENTLY_TRAINING" ]; then
    COMPLETED_NOW=$((COMPLETED - 1))
else
    COMPLETED_NOW=$COMPLETED
fi
[ "$COMPLETED_NOW" -lt 0 ] && COMPLETED_NOW=0
PROGRESS_PCT=$((COMPLETED_NOW * 100 / EXPECTED_TOTAL))

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
echo "Quick actions:"
echo "  tmux attach -t rads-full   # see live (Ctrl+B D to detach)"
echo "  tail -f $LOG_FILE          # follow log (Ctrl+C to exit)"
echo "  W&B:  https://wandb.ai/abhikul4u-hvpm-college-of-engineering-technology-amravati"
echo ""