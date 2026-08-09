#!/usr/bin/env bash
# Run collect_mixed.sh as N parallel shards, then merge them into one dataset
# directory that convert_data_to_lerobot can read. Sharding is at the
# collect_mixed.sh level so every shard keeps the 90/10 ik_noise /
# offset_approach split.
#
# Three things this handles that a bare `for ... &` loop does not:
#   1. MUJOCO_GL=egl. Unset, MuJoCo can pick a software/integrated-GPU
#      rasterizer, and renders are ~99% of a recorded frame's cost. Same
#      frames either way — verified bit-identical at a fixed seed.
#   2. One save dir per shard. Episode ids are `episode_{unix_ms}` and
#      _create_episode_directory passes exist_ok=True, so same-millisecond
#      finishes across shards would silently merge into one directory.
#   3. Disjoint seed ranges, including collect_mixed.sh's SEED + 2000000
#      offset_approach band.
#
# PROCS is bounded by GPU render throughput (and per-process VRAM), not core
# count — measure it on the machine that runs the collect.
#
# Usage:
#   ./collect_parallel.sh [total_attempts] [save_dir] [seed] [procs]
#
# Examples:
#   ./collect_parallel.sh 32 data/smoke 1000 4
#   ./collect_parallel.sh 3100 data/aera_semi_pnp_dr_09082026 1000 8
#
# Env overrides:
#   PYTHON_BIN        bin/ of the interpreter to use (needs cv2 + cv_bridge)
#   MUJOCO_GL         render backend (default egl)
#   SEED_STRIDE       seed spacing between shards (default 10000)
#   IK_NOISE_FRACTION passed through to collect_mixed.sh
#   MERGE             1 (default) to merge shards into $SAVE_DIR/episodes
#   MB_PER_ATTEMPT    disk preflight estimate (default 31)
#   FORCE             1 to proceed despite the disk preflight

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIXED="$SCRIPT_DIR/collect_mixed.sh"

TOTAL="${1:-100}"
SAVE_DIR="${2:-rl_training_data}"
SEED="${3:-1000}"
PROCS="${4:-8}"

PYTHON_BIN="${PYTHON_BIN:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
SEED_STRIDE="${SEED_STRIDE:-10000}"
MERGE="${MERGE:-1}"
MB_PER_ATTEMPT="${MB_PER_ATTEMPT:-31}"
FORCE="${FORCE:-0}"

if [ "$SEED" -eq -1 ]; then
  echo "ERROR: seed -1 (unseeded) defeats the point of sharding — shards would" >&2
  echo "       draw overlapping scenes and the batch would not be reproducible." >&2
  exit 1
fi

# Keep every shard's ik_noise and offset_approach seed bands clear of every
# other shard's.
LAST_SEED=$((SEED + SEED_STRIDE * (PROCS - 1)))
PER_SHARD=$(((TOTAL + PROCS - 1) / PROCS))
if [ "$PER_SHARD" -ge "$SEED_STRIDE" ]; then
  echo "ERROR: SEED_STRIDE ($SEED_STRIDE) <= episodes per shard ($PER_SHARD):" >&2
  echo "       shard seed ranges would overlap. Raise SEED_STRIDE." >&2
  exit 1
fi
if [ "$LAST_SEED" -ge 2000000 ]; then
  echo "ERROR: highest shard seed ($LAST_SEED) collides with collect_mixed.sh's" >&2
  echo "       offset_approach band (SEED + 2000000). Lower SEED or SEED_STRIDE." >&2
  exit 1
fi

# collect_mixed.sh calls bare `python3`; fail now, not N tracebacks deep.
if [ -n "$PYTHON_BIN" ]; then
  export PATH="$PYTHON_BIN:$PATH"
fi
if ! python3 -c "import cv2, cv_bridge, mujoco" 2>/dev/null; then
  echo "ERROR: $(command -v python3) cannot import cv2 / cv_bridge / mujoco." >&2
  echo "       Set PYTHON_BIN to the bin/ of an environment that can." >&2
  exit 1
fi

mkdir -p "$SAVE_DIR"
LOG_DIR="$SAVE_DIR/logs"
mkdir -p "$LOG_DIR"

# Preflight covers only the RAW episode dirs; the LeRobot datasets built from
# them are several times larger again.
NEED_MB=$((TOTAL * MB_PER_ATTEMPT))
AVAIL_MB=$(df -Pm "$SAVE_DIR" | awk 'NR==2 {print $4}')
if [ "$AVAIL_MB" -lt "$NEED_MB" ] && [ "$FORCE" != "1" ]; then
  echo "ERROR: need ~$((NEED_MB / 1024)) GB for raw episodes, $((AVAIL_MB / 1024)) GB free on $SAVE_DIR." >&2
  echo "       Free space, or set FORCE=1 to proceed anyway." >&2
  exit 1
fi

# Spread the remainder over the first shards rather than dumping it on the last.
BASE=$((TOTAL / PROCS))
REM=$((TOTAL % PROCS))

echo "================================================================"
echo "  Parallel collection"
echo "  Total attempts:   $TOTAL"
echo "  Processes:        $PROCS"
echo "  Save dir:         $SAVE_DIR"
echo "  Base seed:        $SEED  (stride $SEED_STRIDE)"
echo "  MUJOCO_GL:        $MUJOCO_GL"
echo "  python3:          $(command -v python3)"
echo "  Raw disk (est.):  ~$((NEED_MB / 1024)) GB of $((AVAIL_MB / 1024)) GB free"
echo "================================================================"

PIDS=()
SHARD_DIRS=()
cleanup() {
  echo ""
  echo "Interrupted — stopping shards..."
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  exit 130
}
trap cleanup INT TERM

START=$(date +%s)
for k in $(seq 0 $((PROCS - 1))); do
  N=$BASE
  [ "$k" -lt "$REM" ] && N=$((BASE + 1))
  [ "$N" -eq 0 ] && continue

  SHARD_SEED=$((SEED + SEED_STRIDE * k))
  SHARD_DIR="$SAVE_DIR/shard$k"
  SHARD_DIRS+=("$SHARD_DIR")

  echo "  shard $k: $N attempts, seed $SHARD_SEED -> $SHARD_DIR"
  "$MIXED" "$N" "$SHARD_DIR" "$SHARD_SEED" > "$LOG_DIR/shard$k.log" 2>&1 &
  PIDS+=($!)
done

FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=$((FAILED + 1))
done
trap - INT TERM
END=$(date +%s)
ELAPSED=$((END - START))

echo ""
echo "All shards finished in $((ELAPSED / 60))m $((ELAPSED % 60))s"
[ "$FAILED" -gt 0 ] && echo "WARNING: $FAILED shard process(es) exited non-zero — read $LOG_DIR/"

# --- Merge -------------------------------------------------------------------
# convert_data_to_lerobot takes one --data-dir and iterates its subdirectories.
EPISODE_DIR="$SAVE_DIR/episodes"
if [ "$MERGE" = "1" ]; then
  mkdir -p "$EPISODE_DIR"
  COLLISIONS=0
  for d in "${SHARD_DIRS[@]}"; do
    [ -d "$d" ] || continue
    for ep in "$d"/episode_*; do
      [ -d "$ep" ] || continue
      name="$(basename "$ep")"
      if [ -e "$EPISODE_DIR/$name" ]; then
        # Same-millisecond ids across shards: rename, never clobber.
        COLLISIONS=$((COLLISIONS + 1))
        name="${name}_$(basename "$d")"
      fi
      mv "$ep" "$EPISODE_DIR/$name"
    done
    rmdir "$d" 2>/dev/null || true
  done
  [ "$COLLISIONS" -gt 0 ] && echo "NOTE: $COLLISIONS episode-id collision(s) across shards, renamed (no data lost)"
  DATA_DIR="$EPISODE_DIR"
else
  DATA_DIR="$SAVE_DIR"
fi

# --- Stage 1 log checks, across every shard ---------------------------------
echo ""
echo "=== Stage 1 checks (verification_gate.md) ==="
NOT_LOCKED=$(cat "$LOG_DIR"/shard*.log | grep -c "Grasp not locked" || true)
IK_ABORT=$(cat "$LOG_DIR"/shard*.log | grep -cE "Max steps .* reached|could not move" || true)
EMPTY_SYNC=$(cat "$LOG_DIR"/shard*.log | grep -c "Synchronized 0 " || true)
COLLECTED=$(ls -d "$DATA_DIR"/episode_* 2>/dev/null | wc -l)

echo "  Grasp not locked:        $NOT_LOCKED        (want 0 — check 8)"
echo "  IK aborts:               $IK_ABORT        (want <=1 in 30 — check 1)"
echo "  Empty episodes:          $EMPTY_SYNC        (want 0 — check 0)"
echo "  Episodes on disk:        $COLLECTED / $TOTAL attempts"
if [ "$TOTAL" -gt 0 ]; then
  echo "  Yield:                   $(awk -v c="$COLLECTED" -v t="$TOTAL" 'BEGIN {printf "%.1f%%", 100 * c / t}')"
  echo "  Wall clock per attempt:  $(awk -v e="$ELAPSED" -v t="$TOTAL" 'BEGIN {printf "%.2f s", e / t}')"
fi
[ "$COLLECTED" -gt 0 ] && echo "  Disk per episode:        $(du -sm "$DATA_DIR" | awk -v c="$COLLECTED" '{printf "%.1f MB", $1 / c}')"

echo ""
echo "Episodes in: $DATA_DIR"
echo "Next: convert_data_to_lerobot.py --data-dir $DATA_DIR --output-dir <dataset_name>"

if [ "$NOT_LOCKED" -ne 0 ] || [ "$EMPTY_SYNC" -ne 0 ]; then
  echo ""
  echo "GATE FAILED: re-collect before converting (see verification_gate.md Stage 1)." >&2
  exit 1
fi
