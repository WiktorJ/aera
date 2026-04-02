#!/usr/bin/env bash
# Collect pick-and-place trajectories with a mixed perturbation strategy:
#   30% base (domain randomization only)
#   20% offset approach (1 waypoint)
#   50% IK noise (0.3 default fraction)
#
# Usage:
#   ./collect_mixed.sh [total_trajectories] [save_dir]
#
# Examples:
#   ./collect_mixed.sh 100
#   ./collect_mixed.sh 500 /data/my_run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT="$SCRIPT_DIR/collect_trajectories.py"

TOTAL="${1:-100}"
SAVE_DIR="${2:-rl_training_data}"

BASE_N=$((TOTAL * 30 / 100))
OFFSET_N=$((TOTAL * 20 / 100))
NOISE_N=$((TOTAL - BASE_N - OFFSET_N)) # remainder goes to ik_noise (~50%)

echo "============================================"
echo "  Mixed trajectory collection"
echo "  Total:         $TOTAL"
echo "  Base (none):   $BASE_N  (30%)"
echo "  Offset:        $OFFSET_N  (20%)"
echo "  IK noise:      $NOISE_N  (~50%)"
echo "  Save dir:      $SAVE_DIR"
echo "============================================"

echo ""
echo "[1/3] Base trajectories ($BASE_N)..."
python3 "$COLLECT" \
  --num-trajectories "$BASE_N" \
  --save-dir "$SAVE_DIR" \
  --perturbation.mode none

echo ""
echo "[2/3] Offset-approach trajectories ($OFFSET_N)..."
python3 "$COLLECT" \
  --num-trajectories "$OFFSET_N" \
  --save-dir "$SAVE_DIR" \
  --perturbation.mode offset_approach \
  --perturbation.num-approach-waypoints 1

echo ""
echo "[3/3] IK-noise trajectories ($NOISE_N)..."
python3 "$COLLECT" \
  --num-trajectories "$NOISE_N" \
  --save-dir "$SAVE_DIR" \
  --perturbation.mode ik_noise \
  --perturbation.ik-noise.default-fraction 0.3

echo ""
echo "Done. All trajectories saved to: $SAVE_DIR"
