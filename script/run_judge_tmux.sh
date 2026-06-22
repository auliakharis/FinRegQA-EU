#!/usr/bin/env bash
# Run judge_api.py in a detached tmux session and stream its logs.
# Usage: ./script/run_judge_tmux.sh <split> [extra args passed to judge_api.py]
# Example: ./script/run_judge_tmux.sh train --n_samples 50

set -euo pipefail

SPLIT="${1:-val}"
shift || true

SESSION="judge_${SPLIT}"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/${SESSION}.log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already running. Attaching..."
else
    tmux new-session -d -s "$SESSION" \
        "caffeinate -i python script/judge_api.py --split $SPLIT $* 2>&1 | tee $LOG_FILE"
    echo "Started session '$SESSION', logging to $LOG_FILE"
fi

tmux attach -t "$SESSION"
