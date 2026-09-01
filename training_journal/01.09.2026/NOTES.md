# Training journal — 2026-09-01

Diagnosis of the remaining failure mode on the 08_08 slow-arm run. The changes
from [06.07.2026](../06.07.2026/implementation_plan.md) all landed and worked:
`grasped` is at 90–95%, and when a grasp is clean the arm finishes the task.

## Run under test

- Checkpoint: `pi05_ar4_mk3_2026-08-27_09-47-05/50000`, dataset
  `Purple69/aera_semi_pnp_dr_08_08_2026_skip10_delta`
- Baseline eval: `eval_results/pi05_ar4_mk3_2026-08-27_09-47-05/50000/summary.json`
  — 60% success, **30% `timeout_holding`**, 7.5% `grasp_missed`

`timeout_holding` is the whole remaining problem: the arm grabs the block and
then stops moving. The rest (~10%) is genuinely hard scenes (DR colour clash,
extreme lighting) and is acceptable.

## Finding 1 — the data contains a "hold still while holding the block" mode

Measured on 177 sampled episodes of the training dataset (parquet, state/actions
only). **Every episode** has a 9–11 frame block immediately after the gripper
close where the joint action is 1e-6–2e-4 rad, against an episode median of
~14 mrad — four orders of magnitude down, i.e. numerically zero.

Cause is known and intended: `_interpolate_gripper` parks the arm
(`data.ctrl[:6] = data.qpos[...]`) for the whole close ramp. At
`record_every=5 × skip=2` the ~100 mj-step ramp becomes ~10 dataset frames.

The defect is not the frame count (3% of an episode) but that those frames are
**not distinguishable from the frame where the lift starts**:

| across the dwell + the resume frame | measured |
|---|---|
| arm joint state travel, total | 5.9e-4 rad (4% of one normal step) |
| jaw state travel | 2.8 mm, stalled for the last ~6 frames |
| frames sharing that observation labelled ~zero | median 6, max 8 |
| resume action size | 0.63 x the episode median — full size, no ramp |

So at that observation the data says "hold still" ~6 times for every 1 "lift".
Emitting zero leaves the observation unchanged, so the policy can re-sample the
same 6:1 mixture indefinitely. Note the previous gate reported this as a healthy
"3–7% parked mass" — the fraction was fine, the *contiguity and ambiguity* is
what matters and was never measured.

## Finding 2 — every demonstrated grasp is perfect, so a crooked one is unseen

`collect_trajectories.py:103,134` grasps at `get_object_pose(env)` exactly.
`ik_noise`, `offset_approach` and `hover` vary the approach *path* only; the
grasp target itself is never jittered. Measured on the scripted expert over 12
seeds with object yaw spanning −44.8°…+31.5°:

| axis | expert (n=12) | eval success | eval `timeout_holding` |
|---|---|---|---|
| yaw misalignment | **0.00°** (max 0.01°) | 3.70° | 9.56° |
| \|finger\| offset | **0.13 mm** (max 0.29 mm) | 4.11 mm | 7.76 mm |
| grip above block top | **+0.90 mm** (spread 0.17 mm) | +1.14 mm | +4.66 mm |
| grip→object distance | — | 13.05 mm | 17.90 mm |

The demonstrations are a *point*, not a distribution. Meanwhile the engage gate
admits `pinch_tol=7 mm`, `finger_tol=20 mm`, `height_tol=27 mm` and any yaw — a
large region where the lock says "grasped" and no demonstration exists.

## The two findings are one mechanism

Alignment predicts the *stall*, not merely the failure. Pooled over two
40-episode runs (n=74 engaged grasps):

| grasp offset tercile | failure rate | mean fraction of held time frozen |
|---|---|---|
| best (6.5–12.1 mm) | 16% | 0.21 |
| middle (12.2–15.0 mm) | 25% | 0.35 |
| worst (15.2–30.6 mm) | **64%** | **0.72** |

Same shape on yaw: 0.1–2.0° → 16% fail / 0.24 frozen; 5.1–35.7° → **64%** /
**0.76**. Pooled correlations with the frozen fraction: distance +0.50,
finger +0.43, yaw +0.41.

Reading: a clean grasp is in-distribution and the policy lifts. A crooked one is
outside anything demonstrated, so the nearest-looking behaviour — the parked
dwell from Finding 1 — wins, and it is self-reinforcing.

Per-episode confirmation: `timeout_holding` episodes are frozen **83–91%** of
their held time (median longest run 154–318 steps). Successes sit at **15–23%**,
median longest run **9–10 steps** — exactly the demonstrated dwell length.

## Finding 3 — `replan_steps=4` discards the escape action

`action_horizon=10`, `DEPLOY_REPLAN_STEPS=4` (`task_env_factory.py:72`). The
dwell is 6–9 frames, so "start lifting" sits at chunk positions 7–10 and is
**never executed**. The policy runs 4 zeros, re-plans from an identical
observation, and repeats.

Tested at `replan_steps=10` (same 20 seeds x 2 repeats, same server):

| replan | success | `timeout_holding` | frozen ≥100 steps | mean frozen fraction |
|---|---|---|---|---|
| 4 (run 1) | 65% | 30% | 17/38 | 0.446 |
| 4 (run 2) | 55% | 35% | 11/36 | 0.413 |
| **10** | 62% | **22%** | **3/36** | **0.335** |

Freezing drops ~38% → 8% of grasps with no retraining. Success barely moves:
unfrozen episodes now fail as `grasp_missed` / `dropped_early` instead, which is
what the two-cause model predicts — this releases the arm, it does not fix the
bad grasp that caused the stall.

## Code change

`aera/autonomous/openpi/eval/metrics.py` only, so the failure is visible in
future evals without watching videos:

- Per engaged grasp: `yaw` (reduced into [0°,45°] for the blocks' 90° symmetry)
  and `grip_above_top`. The latter matters — raw tool-frame `height` is
  grip-to-block-*centre*, so it carries the block's half-height and is not
  comparable across the 19/22/24/27 mm presets; subtracting it gives an axis on
  which the expert sits at +0.9 mm on every episode.
- Per episode: `held_steps`, `stall_steps`, `longest_stall_run`, `stalled`
  (≥100 steps under 1 mrad of joint motion, counted only while holding).
- Aggregated as `eval/grasp/*` and `eval/stall/*`; `flatten_for_mlflow` passes
  them through unchanged.

Results: `eval_results/pi05_ar4_mk3_2026-08-27_09-47-05/{50000_alignment,
50000_alignment_v2,50000_replan10}`.

## Caveats

- 40 episodes per run; two identical configurations gave 55% and 65% success.
  Treat the stall numbers as solid and the success rates as noisy.
- The expert alignment baseline was measured on **nominal** arm dynamics (no DR)
  — the same trap recorded in the 06.07 journal for Stage 0. The grasp *target*
  is exact by construction either way, but the achieved spread under DR could be
  wider than 0.13 mm.
- All three evals were driven through an already-running `serve_policy` on port
  8000 rather than loading a second copy onto the GPU. Same checkpoint, same
  `run_suite`, only the policy handle differs.
- The alignment↔stall link is a strong association (n=74, monotonic in both
  axes) plus a plausible mechanism, not a proven direction.

## Next, in cost order

1. **Raise `replan_steps`** (free). Sweep 4–10 properly; 10 is one run.
2. **Drop the parked ramp frames in the transform** — re-transform only, no
   re-collect. Removes the zero-action block that the freeze imitates. Check it
   against check 5's grasp-window floor, which counts those same frames.
3. **Jitter the grasp pose during collection** (yaw and lateral offset, inside
   the engage gate's bounds) so the data shows the arm lifting from an imperfect
   grasp. Needs a re-collect: ~17.7 h single process, ~2.2 h at 8 shards.

1 and 2 attack the freeze; only 3 attacks its cause.
