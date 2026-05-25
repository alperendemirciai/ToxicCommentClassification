#!/bin/bash
# Chain runner: waits for the currently running zero-shot to finish, then
# launches few-shot, then the extension runs (Qwen zero + few, RoBERTa BERT).
# Uses nohup so the chain survives session exits. Writes a flag file when done.
#
# Usage:
#   bash scripts/chain_llm.sh <zero_shot_pid>
# Pass 0 if no zero-shot run is currently active (will launch fresh).

set -e
cd "$(dirname "$0")/.."

source /home/alperen/miniconda3/etc/profile.d/conda.sh
conda activate ain414

ZERO_PID="${1:-0}"
LOG_DIR="logs"
FLAG_DIR="$LOG_DIR/flags"
mkdir -p "$FLAG_DIR"

log() { echo "[$(date '+%H:%M:%S')] [chain] $*" | tee -a "$LOG_DIR/chain.log"; }

if [ "$ZERO_PID" != "0" ]; then
    log "Waiting for zero-shot PID=$ZERO_PID to finish..."
    while ps -p "$ZERO_PID" > /dev/null 2>&1; do sleep 60; done
    log "Zero-shot done."
else
    log "Launching fresh zero-shot..."
    python scripts/03_run_llm.py --model llama3 --mode zero \
        > "$LOG_DIR/llm_llama3_zero_run.out" 2>&1
    log "Zero-shot done."
fi
touch "$FLAG_DIR/llama3_zero.done"

log "Launching llama3 few-shot..."
python scripts/03_run_llm.py --model llama3 --mode few \
    > "$LOG_DIR/llm_llama3_few_run.out" 2>&1
log "Llama3 few-shot done."
touch "$FLAG_DIR/llama3_few.done"

log "Launching RoBERTa training (extension)..."
python scripts/02_train_bert.py --model roberta --max-len 512 --epochs 4 \
    > "$LOG_DIR/bert_roberta_run.out" 2>&1
log "RoBERTa done."
touch "$FLAG_DIR/roberta.done"

log "Launching qwen zero-shot..."
python scripts/03_run_llm.py --model qwen --mode zero \
    > "$LOG_DIR/llm_qwen_zero_run.out" 2>&1
log "Qwen zero-shot done."
touch "$FLAG_DIR/qwen_zero.done"

log "Launching qwen few-shot..."
python scripts/03_run_llm.py --model qwen --mode few \
    > "$LOG_DIR/llm_qwen_few_run.out" 2>&1
log "Qwen few-shot done."
touch "$FLAG_DIR/qwen_few.done"

log "Aggregating results + generating REPORT.md ..."
python scripts/04_aggregate_results.py >> "$LOG_DIR/aggregate.log" 2>&1
python scripts/05_generate_report.py >> "$LOG_DIR/aggregate.log" 2>&1
touch "$FLAG_DIR/all.done"
log "ALL DONE."
