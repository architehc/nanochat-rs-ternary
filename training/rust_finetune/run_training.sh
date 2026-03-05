#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "============================================"
echo "RTX 5090 7-Day Training Launcher"
echo "$(date)"
echo "============================================"

# Activate venv
source "$VENV_DIR/bin/activate"
echo "Venv activated"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Download data if needed
DATA_V2="$SCRIPT_DIR/data/rust_code_v2"
if [ ! -d "$DATA_V2" ]; then
    echo "Downloading expanded dataset..."
    python3 "$SCRIPT_DIR/download_more_data.py"
fi

# Launch training
echo "Starting 7-day training..."
mkdir -p "$SCRIPT_DIR/output_5090/logs"
LOGFILE="$SCRIPT_DIR/output_5090/logs/stdout_$(date +%Y%m%d_%H%M%S).log"
nohup python3 -u "$SCRIPT_DIR/train_7day.py" > "$LOGFILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$SCRIPT_DIR/output_5090/train.pid"
echo "PID: $TRAIN_PID | Log: $LOGFILE"

sleep 10
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "Training started successfully"
    tail -5 "$LOGFILE"
else
    echo "ERROR: Training died"
    cat "$LOGFILE"
    exit 1
fi
