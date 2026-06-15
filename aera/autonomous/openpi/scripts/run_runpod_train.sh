#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config_name> [args...]"
    exit 1
fi

CONFIG_NAME=$1
shift

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
EXP_NAME="${CONFIG_NAME}_${TIMESTAMP}"

echo "Running training with:"
echo "  Config: $CONFIG_NAME"
echo "  Experiment Name: $EXP_NAME"
echo "  Checkpoint Dir: /workspace"

# Auto-launch the decoupled eval worker (set RUN_EVAL=0 to disable). It waits for
# training to write the mlflow run id, then evals each new checkpoint and logs the
# granular funnel metrics to the same mlflow run. Because it shares the single GPU
# with training, we also lower training's GPU memory fraction to leave headroom.
# Tune via EVAL_MEM_FRACTION / TRAIN_MEM_FRACTION; pass eval flags via EVAL_ARGS
# (e.g. EVAL_ARGS="--n-substeps 3 --num-episodes 25").
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${RUN_EVAL:-1}" = "1" ]; then
    echo "Launching eval worker in background (set RUN_EVAL=0 to disable)..."
    nohup bash "$SCRIPT_DIR/run_runpod_eval.sh" "$CONFIG_NAME" "$EXP_NAME" \
        ${EVAL_ARGS:-} >/workspace/eval_worker.log 2>&1 &
    echo "  eval worker pid $! -> logs at /workspace/eval_worker.log"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${TRAIN_MEM_FRACTION:-0.8}"
    echo "  lowered training GPU mem fraction to $XLA_PYTHON_CLIENT_MEM_FRACTION to share the GPU"
fi

uv run python aera/autonomous/openpi/scripts/train.py \
    "$CONFIG_NAME" \
    --checkpoint-base-dir "/workspace" \
    --exp-name "$EXP_NAME" \
    "$@"
