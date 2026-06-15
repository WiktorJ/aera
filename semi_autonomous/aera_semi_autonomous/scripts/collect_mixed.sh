#!/usr/bin/env bash
# Collect pick-and-place trajectories with a mixed perturbation strategy.
#
# Every segment runs the full non-visual lever stack (camera + arm-dynamics +
# object-yaw randomization, plus home / hover / speed / actuation perturbation);
# the segments differ only in the trajectory MODE and whether recovery data is
# injected:
#
#   70%  all levers, ik_noise mode, NO recovery   (the clean working bulk)
#   20%  all levers, ik_noise mode, WITH recovery (grasp-time failure + recovery)
#   10%  all levers, offset_approach mode         (varied approach paths)
#
# Visual/material/light DR is always on inside collect_trajectories and is not
# toggled here.
#
# Usage:
#   ./collect_mixed.sh [total_trajectories] [save_dir] [seed]
#
# Examples:
#   ./collect_mixed.sh 100
#   ./collect_mixed.sh 500 /data/my_run
#   ./collect_mixed.sh 500 /data/my_run 42

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT="$SCRIPT_DIR/collect_trajectories.py"

TOTAL="${1:-100}"
SAVE_DIR="${2:-rl_training_data}"
SEED="${3:--1}"

# Multiplicative IK-noise level for the two ik_noise segments.
IK_NOISE_FRACTION="${IK_NOISE_FRACTION:-0.1}"

NOISE_N=$((TOTAL * 70 / 100))
RECOVERY_N=$((TOTAL * 20 / 100))
OFFSET_N=$((TOTAL - NOISE_N - RECOVERY_N)) # remainder (~10%)

# Distinct seeds per segment so segments don't draw the same scenes. Only
# meaningful when a fixed seed is given; with the -1 default each run is
# unseeded/independent.
if [ "$SEED" -eq -1 ]; then
  NOISE_SEED=-1; RECOVERY_SEED=-1; OFFSET_SEED=-1
else
  NOISE_SEED="$SEED"
  RECOVERY_SEED=$((SEED + 1000000))
  OFFSET_SEED=$((SEED + 2000000))
fi

# Full non-visual lever stack, shared by every segment.
COMMON=(
  --save-dir "$SAVE_DIR"
  --randomize-cameras
  --randomize-arm-dynamics
  --randomize-object-yaw
  --perturbation.perturb-home
  --perturbation.perturb-hover-height
  --perturbation.perturb-speed
  --perturbation.perturb-actuation
)

echo "============================================"
echo "  Mixed trajectory collection (full-stack)"
echo "  Total:            $TOTAL"
echo "  ik_noise:         $NOISE_N  (70%, no recovery)"
echo "  ik_noise+recovery:$RECOVERY_N  (20%)"
echo "  offset_approach:  $OFFSET_N  (10%, no recovery)"
echo "  ik_noise frac:    $IK_NOISE_FRACTION"
echo "  Save dir:         $SAVE_DIR"
echo "  Seed:             $SEED"
echo "============================================"

echo ""
echo "[1/3] ik_noise, no recovery ($NOISE_N)..."
python3 "$COLLECT" \
  "${COMMON[@]}" \
  --num-trajectories "$NOISE_N" \
  --seed "$NOISE_SEED" \
  --perturbation.mode ik_noise \
  --perturbation.ik-noise.default-fraction "$IK_NOISE_FRACTION" \
  --perturbation.no-perturb-recovery

echo ""
echo "[2/3] ik_noise + recovery ($RECOVERY_N)..."
python3 "$COLLECT" \
  "${COMMON[@]}" \
  --num-trajectories "$RECOVERY_N" \
  --seed "$RECOVERY_SEED" \
  --perturbation.mode ik_noise \
  --perturbation.ik-noise.default-fraction "$IK_NOISE_FRACTION" \
  --perturbation.perturb-recovery

echo ""
echo "[3/3] offset_approach, no recovery ($OFFSET_N)..."
python3 "$COLLECT" \
  "${COMMON[@]}" \
  --num-trajectories "$OFFSET_N" \
  --seed "$OFFSET_SEED" \
  --perturbation.mode offset_approach \
  --perturbation.num-approach-waypoints 1 \
  --perturbation.no-perturb-recovery

echo ""
echo "Done. All trajectories saved to: $SAVE_DIR"
