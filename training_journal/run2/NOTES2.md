# Training journal — 2026-09-04

Grasp-pose DR — item 3 of [01.09](../01.09.2026/NOTES.md). Jitter the grasp
target so demos show successful lifts from *imperfect* grasps, unlocking
"proceed when the grasp isn't perfect" without teaching sloppiness.

## What

`GraspPoseJitter` + `apply_grasp_pose_jitter` (`trajectory_perturbation.py`),
exposed as the composable `perturb_grasp_pose` flag on `PerturbationConfig` and
wired into `collect_mixed.sh`'s lever stack. Jitters the grasp **target only**
(the place height and any recovery injection keep the true object pose).
Zero-centred, in the gripper tool frame. Validation subcommand
`measure_scripted_arm grasp-jitter` reports the achieved offset + lock yield.

## Axes — measured what the close actually does to each (dt=0.009, full close)

- **finger ±7 mm** (along the jaws): the only axis that *persists* — the jaws
  don't recentre along their length. Reproduces the eval stall band; this is the
  one that attacks the post-grasp stall.
- **yaw ±12°** and **pinch ±1.5 mm**: *self-correct* on close (free block
  settles: ±12° → <2° held; ±5 mm pinch → ±2.7 mm). Not a defect — it's what
  real hardware does. The signal is the jittered *approach* ("commit to the
  close, the jaws settle it"). Corrects item 3's "yaw and lateral offset"
  premise: target-yaw jitter cannot produce a held-yaw grasp.
- **height +0–2 mm** (higher only): a shallower grasp; lower would hit the table.

## Magnitudes vs block size

Graspable presets 19/22/24 mm; 23 mm pad; 28.8 mm open jaw gap. Ranges are sized
for the worst preset so no per-episode clamping is needed:

- finger 7 mm → ≥60% of the pad stays on the smallest (19 mm) block.
- pinch 1.5 mm → inside the widest (24 mm) block's 2.4 mm open-jaw descent
  clearance, so a jaw never comes down on the block top.

## Safety against sloppiness

Zero-centred (mean grasp stays perfect); the self-correcting axes end at a clean
grasp; `stop_episode(success)` drops any grasp too crooked to complete — the
jitter can only lower yield, never inject a failure. Validation: 10/10 locked at
the defaults on the 24 mm block.

## Next

Lands with the item-2 static-frame filter in the same re-collect (shared
`integration_dt` / skip / filter params). Residual dwell after that filter sets
the next policy's `replan` (~5–6), unchanged by this.
