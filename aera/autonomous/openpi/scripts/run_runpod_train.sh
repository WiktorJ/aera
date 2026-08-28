#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <config_name> [args...]"
  echo "  RESUME=<exp_name>|latest   resume that run instead of starting a new one"
  echo "  AERA_CHECKPOINT_DIR=<dir>  checkpoint base dir (default /workspace)"
  exit 1
fi

CONFIG_NAME=$1
shift

# train.py and eval_worker.py both resolve <base>/<config>/<exp> from
# AERA_CHECKPOINT_DIR, so export it here to keep the two on the same path.
export AERA_CHECKPOINT_DIR="${AERA_CHECKPOINT_DIR:-/workspace}"
CONFIG_CKPT_DIR="$AERA_CHECKPOINT_DIR/$CONFIG_NAME"

TRAIN_ARGS=()
if [ -n "${RESUME:-}" ]; then
  # Resume restores step, optimizer state and EMA, and re-attaches to the run's
  # existing mlflow run id, so metrics continue on the same curve.
  if [ "$RESUME" = "latest" ]; then
    # exp names are <config>_<timestamp>, so lexicographic order is chronological.
    EXP_NAME=$(ls -1 "$CONFIG_CKPT_DIR" 2>/dev/null | sort | tail -1)
    if [ -z "$EXP_NAME" ]; then
      echo "Error: no runs found under $CONFIG_CKPT_DIR to resume."
      exit 1
    fi
  else
    EXP_NAME="$RESUME"
  fi

  CKPT_DIR="$CONFIG_CKPT_DIR/$EXP_NAME"
  if [ ! -d "$CKPT_DIR" ]; then
    echo "Error: $CKPT_DIR does not exist. Available runs:"
    ls -1 "$CONFIG_CKPT_DIR" 2>/dev/null || echo "  (none under $CONFIG_CKPT_DIR)"
    exit 1
  fi
  # Without a saved step, train.py aborts the resume and silently restarts from
  # the base checkpoint, so fail here instead.
  if ! ls -d "$CKPT_DIR"/[0-9]* >/dev/null 2>&1; then
    echo "Error: $CKPT_DIR holds no step checkpoints to resume from."
    exit 1
  fi
  TRAIN_ARGS+=(--base-config.resume)
else
  TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
  EXP_NAME="${CONFIG_NAME}_${TIMESTAMP}"
fi

echo "Running training with:"
echo "  Config: $CONFIG_NAME"
echo "  Experiment Name: $EXP_NAME"
echo "  Checkpoint Dir: $CONFIG_CKPT_DIR/$EXP_NAME"
if [ -n "${RESUME:-}" ]; then
  echo "  Mode: resume (steps found: $(ls -d "$CKPT_DIR"/[0-9]* 2>/dev/null | xargs -n1 basename | sort -n | tr '\n' ' '))"
  echo "        note: no-ops if the run already reached num_train_steps;"
  echo "        pass --base-config.num-train-steps <N> to extend it."
fi

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
  --base-config.exp-name "$EXP_NAME" \
  "${TRAIN_ARGS[@]}" \
  "$@"
