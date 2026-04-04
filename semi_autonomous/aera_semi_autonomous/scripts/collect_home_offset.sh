#!/usr/bin/env bash
# Collect pick-and-place trajectories with home offset perturbation:
#   3 home offset magnitudes (0.1, 0.2, 0.3) × 2 IK noise settings (none, 0.3)
#   = 6 buckets, each receiving an equal share of total trajectories.
#
# Usage:
#   ./collect_home_offset.sh [total_trajectories] [save_dir] [seed]
#
# Examples:
#   ./collect_home_offset.sh 600
#   ./collect_home_offset.sh 600 /data/my_run
#   ./collect_home_offset.sh 600 /data/my_run 42

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT="$SCRIPT_DIR/collect_trajectories.py"

TOTAL="${1:-600}"
SAVE_DIR="${2:-rl_training_data}"
SEED="${3:--1}"

BUCKET=$((TOTAL / 6))
# Last bucket absorbs any remainder from integer division
LAST_BUCKET=$((TOTAL - BUCKET * 5))

echo "============================================"
echo "  Home-offset trajectory collection"
echo "  Total:        $TOTAL"
echo "  Buckets:      6 x ~$BUCKET each"
echo "  Save dir:     $SAVE_DIR"
echo "  Seed:         $SEED"
echo "============================================"

echo ""
echo "[1/6] Home offset 0.1, no IK noise ($BUCKET)..."
python3 "$COLLECT" \
  --num-trajectories "$BUCKET" \
  --save-dir "$SAVE_DIR" \
  --seed "$SEED" \
  --perturbation.mode none \
  --perturbation.perturb-home \
  --perturbation.home-offset.default-max-offset 0.1

echo ""
echo "[2/6] Home offset 0.2, no IK noise ($BUCKET)..."
python3 "$COLLECT" \
  --num-trajectories "$BUCKET" \
  --save-dir "$SAVE_DIR" \
  --seed "$SEED" \
  --perturbation.mode none \
  --perturbation.perturb-home \
  --perturbation.home-offset.default-max-offset 0.2

echo ""
echo "[3/6] Home offset 0.3, no IK noise ($BUCKET)..."
python3 "$COLLECT" \
  --num-trajectories "$BUCKET" \
  --save-dir "$SAVE_DIR" \
  --seed "$SEED" \
  --perturbation.mode none \
  --perturbation.perturb-home \
  --perturbation.home-offset.default-max-offset 0.3

echo ""
echo "[4/6] Home offset 0.1, IK noise 0.3 ($BUCKET)..."
python3 "$COLLECT" \
  --num-trajectories "$BUCKET" \
  --save-dir "$SAVE_DIR" \
  --seed "$SEED" \
  --perturbation.mode ik_noise \
  --perturbation.ik-noise.default-fraction 0.3 \
  --perturbation.perturb-home \
  --perturbation.home-offset.default-max-offset 0.1

echo ""
echo "[5/6] Home offset 0.2, IK noise 0.3 ($BUCKET)..."
python3 "$COLLECT" \
  --num-trajectories "$BUCKET" \
  --save-dir "$SAVE_DIR" \
  --seed "$SEED" \
  --perturbation.mode ik_noise \
  --perturbation.ik-noise.default-fraction 0.3 \
  --perturbation.perturb-home \
  --perturbation.home-offset.default-max-offset 0.2

echo ""
echo "[6/6] Home offset 0.3, IK noise 0.3 ($LAST_BUCKET)..."
python3 "$COLLECT" \
  --num-trajectories "$LAST_BUCKET" \
  --save-dir "$SAVE_DIR" \
  --seed "$SEED" \
  --perturbation.mode ik_noise \
  --perturbation.ik-noise.default-fraction 0.3 \
  --perturbation.perturb-home \
  --perturbation.home-offset.default-max-offset 0.3

echo ""
echo "Done. All trajectories saved to: $SAVE_DIR"
