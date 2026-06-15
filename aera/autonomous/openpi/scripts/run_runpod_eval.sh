#!/bin/bash
# Launch the decoupled eval worker against a training run.
#
# It resolves the checkpoint dir from <config_name>/<exp_name> (the same path
# train.py writes to under AERA_CHECKPOINT_DIR), waits for training to write the
# mlflow run id, then evals each new checkpoint and logs the granular funnel
# metrics to that same mlflow run.
#
# Usage: run_runpod_eval.sh <config_name> <exp_name> [extra eval_worker args...]
#
# IMPORTANT: --n-substeps MUST match the dataset `--skip` the checkpoint was
# trained on (e.g. a skip=3 dataset needs `--n-substeps 3`). Pass it through as
# an extra arg; the default (20) is only correct for skip=20 datasets.
set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <config_name> <exp_name> [args...]"
    exit 1
fi

CONFIG_NAME=$1
EXP_NAME=$2
shift 2

# Share the single GPU with the training process: don't preallocate, and cap the
# fraction this process may grab so it fits in the headroom training leaves.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${EVAL_MEM_FRACTION:-0.15}"

echo "Starting eval worker:"
echo "  Config:          $CONFIG_NAME"
echo "  Experiment Name: $EXP_NAME"
echo "  GPU mem fraction: $XLA_PYTHON_CLIENT_MEM_FRACTION (preallocate=$XLA_PYTHON_CLIENT_PREALLOCATE)"
echo "  NOTE: ensure --n-substeps matches the dataset skip the checkpoint used."

uv run python aera/autonomous/openpi/scripts/eval_worker.py \
    --config "$CONFIG_NAME" \
    --exp-name "$EXP_NAME" \
    "$@"
