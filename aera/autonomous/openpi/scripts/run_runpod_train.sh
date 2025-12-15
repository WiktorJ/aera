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

uv run python aera/autonomous/openpi/scripts/train.py \
    "$CONFIG_NAME" \
    --checkpoint-base-dir "/workspace" \
    --exp-name "$EXP_NAME" \
    "$@"
