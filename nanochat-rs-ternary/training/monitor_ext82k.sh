#!/bin/bash
# Watchdog for the 82k continuation run on the 5090.
#
# Every 30 min: log a status line (step/loss/tok/s + GPU temp/util/VRAM).
# Every 60 min: commit the training log to the 5090 branch (same pattern as
# the earlier qwen35 monitors).
# On crash: relaunch from the newest step_* checkpoint with the same schedule.
# Exits on its own once the run reaches total_steps or the deadline passes.
set -u

REPO=/home/galic/nanochat-rs-ternary
DIR=$REPO/nanochat-rs-ternary
LOG=$DIR/training/qwen35_ext82k_train.log
MON=$DIR/training/monitor_ext82k.log
PIDF=$DIR/training/qwen35_ext82k.pid
TOTAL=82000
DEADLINE=$(( $(date +%s) + 9*3600 ))   # hard stop: 9h wall (8h train + margin)

cd "$DIR" || exit 1
say() { echo "[$(date '+%F %T')] $*" >> "$MON"; }

trainer_pid() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1; }

restart() {
    local ckpt
    ckpt=$(ls -d checkpoints/qwen35_hybrid_seq512/step_* 2>/dev/null \
           | sed 's/.*step_//' | sort -n | tail -1)
    [ -z "$ckpt" ] && { say "RESTART FAILED: no checkpoint found"; return 1; }
    say "restarting from step_$ckpt"
    setsid ./target/release/nanochat-train train \
        --config qwen35-hybrid --device cuda \
        --dataset tokens --data-path data/rust_v4_4k/tokens.bin \
        --resume "checkpoints/qwen35_hybrid_seq512/step_$ckpt" \
        --batch-size 4 --seq-len 512 --log-interval 50 \
        --total-steps $TOTAL \
        --checkpoint-dir checkpoints/qwen35_hybrid_seq512 \
        --checkpoint-interval 1000 --keep-last-checkpoints 5 \
        >> "$LOG" 2>&1 < /dev/null &
    disown
    sleep 120
    trainer_pid > "$PIDF"
}

say "monitor started (total=$TOTAL, deadline=$(date -d @$DEADLINE '+%F %T'))"
LAST_PUSH=0
while :; do
    NOW=$(date +%s)
    LINE=$(grep -E "^\[" "$LOG" | tail -1)
    STEP=$(echo "$LINE" | sed 's/^\[ *\([0-9]*\).*/\1/')
    GPU=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader)

    if grep -q "Training complete" "$LOG"; then
        say "COMPLETE: $LINE"
        (cd "$REPO" && git add -f nanochat-rs-ternary/training/qwen35_ext82k_train.log \
            nanochat-rs-ternary/training/monitor_ext82k.log 2>/dev/null \
         && git commit -q -m "qwen35 ext82k: complete — ${LINE:-done}" && git push -q origin 5090) || true
        break
    fi
    if [ "$NOW" -gt "$DEADLINE" ]; then
        say "DEADLINE reached at step ${STEP:-?}; leaving trainer running, monitor exiting"
        break
    fi

    PID=$(trainer_pid)
    if [ -z "$PID" ]; then
        say "TRAINER DOWN. last: $LINE"
        restart || break
    else
        echo "$PID" > "$PIDF"
        say "ok: ${LINE:-starting} | gpu: $GPU"
    fi

    if [ $((NOW - LAST_PUSH)) -ge 3600 ]; then
        (cd "$REPO" && git add -f nanochat-rs-ternary/training/qwen35_ext82k_train.log \
            nanochat-rs-ternary/training/monitor_ext82k.log 2>/dev/null \
         && git commit -q -m "qwen35 ext82k: ${LINE:-starting} (hourly update)" \
         && git push -q origin 5090) || say "git push failed (non-fatal)"
        LAST_PUSH=$NOW
    fi
    sleep 1800
done
say "monitor exiting"
