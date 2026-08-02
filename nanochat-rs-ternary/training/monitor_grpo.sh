#!/bin/bash
# Watchdog for the GRPO compiler-feedback RL run on the 5090.
#
# Every 30 min: log a status line (iteration/reward/compile-rate + GPU stats).
# Every 60 min: commit the RL log to the 5090 branch (same pattern as the
# qwen35 monitors).
# On crash: relaunch from the newest rl-iter-* checkpoint (falls back to the
# pretrained base). Note a restart re-anchors the KL reference model to the
# resumed checkpoint.
# Exits once train_rl finishes all iterations or the deadline passes.
set -u

REPO=/home/galic/nanochat-rs-ternary
DIR=$REPO/nanochat-rs-ternary
LOG=$DIR/training/grpo_train.log
MON=$DIR/training/monitor_grpo.log
PIDF=$DIR/training/grpo_train.pid
CSV=$DIR/rl_training.log
BASE=checkpoints/qwen35_hybrid_seq512/step_82000
# n_samples=8 peaked at 32028/32607 MiB during batched generation (candle
# retains the full activation graph); 6 leaves real headroom for the
# unattended run. ~162s/iter measured at 8 samples → 250 iters ≈ 10h.
ITERATIONS=250
N_SAMPLES=6
BATCH=2
DEADLINE=$(( $(date +%s) + 16*3600 ))   # hard stop: 16h wall

cd "$DIR" || exit 1
say() { echo "[$(date '+%F %T')] $*" >> "$MON"; }

trainer_pid() { pgrep -f "train_rl --checkpoint" | head -1; }

launch() {
    local ckpt="$1"
    say "launching from $ckpt"
    setsid env NANOCHAT_TOKENIZER=data/rust_v4_4k/tokenizer.json \
        ./target/release/examples/train_rl \
        --checkpoint "$ckpt" \
        --iterations $ITERATIONS --n-samples $N_SAMPLES --batch-size $BATCH \
        --device cuda:0 \
        >> "$LOG" 2>&1 < /dev/null &
    disown
    sleep 60
    trainer_pid > "$PIDF"
}

restart() {
    local ckpt
    ckpt=$(ls -d checkpoints/rl-iter-* 2>/dev/null \
           | sed 's/.*rl-iter-//' | sort -n | tail -1)
    if [ -n "$ckpt" ]; then
        launch "checkpoints/rl-iter-$ckpt"
    else
        launch "$BASE"
    fi
}

say "monitor started (iterations=$ITERATIONS, deadline=$(date -d @$DEADLINE '+%F %T'))"
LAST_PUSH=0
while :; do
    NOW=$(date +%s)
    LINE=$(grep -E "^Iteration " "$LOG" | tail -1)
    STATS=$(tail -1 "$CSV" 2>/dev/null)
    GPU=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader)

    if grep -q "Training Complete" "$LOG"; then
        say "COMPLETE: ${LINE:-?} | csv: ${STATS:-none}"
        (cd "$REPO" && git add -f nanochat-rs-ternary/training/grpo_train.log \
            nanochat-rs-ternary/training/monitor_grpo.log \
            nanochat-rs-ternary/rl_training.log 2>/dev/null \
         && git commit -q -m "grpo: complete — ${STATS:-done}" && git push -q origin 5090) || true
        break
    fi
    if [ "$NOW" -gt "$DEADLINE" ]; then
        say "DEADLINE reached at ${LINE:-?}; leaving trainer running, monitor exiting"
        break
    fi

    PID=$(trainer_pid)
    if [ -z "$PID" ]; then
        say "TRAINER DOWN. last: ${LINE:-?} | csv: ${STATS:-none}"
        restart || break
    else
        echo "$PID" > "$PIDF"
        say "ok: ${LINE:-starting} | csv: ${STATS:-none} | gpu: $GPU"
    fi

    if [ $((NOW - LAST_PUSH)) -ge 3600 ]; then
        (cd "$REPO" && git add -f nanochat-rs-ternary/training/grpo_train.log \
            nanochat-rs-ternary/training/monitor_grpo.log \
            nanochat-rs-ternary/rl_training.log 2>/dev/null \
         && git commit -q -m "grpo: ${STATS:-starting} (hourly update)" \
         && git push -q origin 5090) || say "git push failed (non-fatal)"
        LAST_PUSH=$NOW
    fi
    sleep 1800
done
say "monitor exiting"
