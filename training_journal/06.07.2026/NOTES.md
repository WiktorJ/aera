# Training journal — 2026-07-06

**Status: OPEN — waiting for the 50k checkpoint eval before final conclusion.**

## Run

- Run name: `pi05_ar4_mk3_2026-07-04_16-51-56` (mlflow run `f8cd833f3a4a434fb323f2012113016c`, snapshot in `mlflow.db` alongside this note)
- Config: `pi05_ar4_mk3` — pi0.5 full finetune, action_horizon 10, batch 32, effectively flat LR 5e-5 (`peak_lr == decay_lr`), EMA 0.999, planned 300k steps
- Dataset: `Purple69/aera_semi_pnp_dr_16_06_2026_skip3_delta_no_go_home_no_static_smoothed_v2`
- Eval: decoupled eval worker, 20 episodes per checkpoint, kinematic grasp lock on

## Eval funnel by checkpoint

| step | reached | grasped | transported | success | grasp_drop_rate |
|------|---------|---------|-------------|---------|-----------------|
| 10k  | 0.90 | 0.75 | 0.15 | 0.25 | 0.60 |
| 20k  | 0.90 | 0.65 | 0.30 | 0.35 | 0.39 |
| 30k  | 0.95 | 0.80 | 0.30 | 0.40 | 0.38 |
| 40k  | 0.80 | 0.75 | 0.30 | 0.20 | 0.67 |

Improved through 30k (peak success 0.40), regressed at 40k (success 0.20, drop rate back to 0.67).

## Visual observations (manual rollouts of this policy)

- Object identification and approach are reliable; positioning around the block is good.
- The grasp itself is the weak phase: jaws don't close fully, visible gaps between jaws and block, block wobbles, drops mid-transport.
- When a grasp does complete fully/strongly, transport to target almost always succeeds — outcome is nearly binary.

## Diagnosis (working hypothesis)

The failure signature matches the **eval grasp-engagement gate rejecting marginal closes**, not a physics mismatch per se. Collection and eval share `KinematicGraspLock`, so welded transport is identical in both; what differs is who decides engagement:

- Collection engages after a scripted close that physically completed, with the close command computed from the block's true geometry: `-(half_width - 0.5mm)`.
- Eval infers engagement from the policy's gripper command, gated by alignment (7/20/27 mm tool-frame tolerances) **and** a close-depth gate: command must reach `-(half_width + 0.5mm)`.

Net effect: the policy must reproduce the per-block close depth to within **~1 mm of jaw travel** or it gets no weld and falls into raw MuJoCo contact grasping — a regime the codebase documents as near-hopeless for these blocks and of which the training data contains **zero** frames (every demo grasp is welded). This maps 1:1 onto the observed symptoms:

- Visible jaw–block gap = commanded close stopped 1–2 mm short of the surface.
- Wobble = lock never engaged (a welded object cannot wobble; `enforce()` re-pins every substep).
- Mid-transport drops = friction-only hold slipping.
- Binary outcomes = the gate is binary.

Why the policy plausibly misses by ~1 mm:

1. **Regression to the mean across block widths.** Close depth varies with sampled block size (≈ -0.009 for 19 mm to ≈ -0.0115 for 24 mm). A flow-matching policy blurring across sizes outputs an averaged depth; on smaller-than-average blocks that command physically stops short of the surface and fails the gate. Predicts failures biased toward smaller blocks — testable.
2. **Savitzky-Golay smoothing (`smoothed_v2`) is applied to ALL action dims including the gripper** (`dataset_transforms.compute_smoothed_arrays`, no dim mask), softening the open→close transition frames — exactly where the gate demands precision.
3. Under quantile normalization a 1 mm gripper error is negligible to the training loss, while eval demands sub-mm accuracy on that channel.

The ≥40k regression is likely a **second, separate issue**: flat LR 5e-5 full-finetune on a modest dataset → classic overfit territory. Note 40k also lost `reached` (0.95→0.80), so it's not purely a grasp-channel problem.

## Diagnostics to run before deciding

- [ ] Eval the 50k checkpoint (in progress — training continues to 50k).
- [ ] Log engage rejection reason + margin in `KinematicGraspLock.engage()` (distance vs alignment vs close-depth, and by how much). Histogram over one eval run is decisive.
- [ ] A/B eval with `close_depth_tol` raised to 2–3 mm (or `require_alignment=False`) to quantify how much of the grasp funnel the gate explains.
- [ ] Check whether grasp failures correlate with small block widths (regression-to-mean prediction).
- [ ] Manual runs: confirm `--n-substeps 3` was passed (`run_policy_on_env.py` defaults to 20; skip3 checkpoint needs 3).

## Provisional recommendation for the next run (pending 50k eval)

Change **collection**, not the eval gate: command a *fully closed* gripper target (0.0) and let the jaws stall on the block (lock pins them at settle via `maybe_pin_jaws`). The demonstrated gripper channel becomes effectively binary — no per-block depth to regress on, sub-mm precision requirement disappears — and this is also what transfers to the real robot (position command past the surface, squeeze force set by the actuator's force/current limit, not by the setpoint). Requires re-collecting or re-deriving the dataset and simplifying the eval close-depth gate to a plain threshold. Caveats for hardware: gripper must tolerate sustained stall (current limit / compliant fingertips), and compliance pads also prevent stiff-jaw block ejection.

Also worth considering for the next run regardless: exclude the gripper dim from Savitzky-Golay smoothing, and add LR decay (the flat 5e-5 schedule is a plausible contributor to the post-30k regression).

## Final conclusion

_To be filled in after evaluating the 50k checkpoint._


## checkpoint 50k
Never works in manual tests. Arm approaches fine but then it tries to grasp a bit off (like in front and next to the block) and obviously keeps failing to grasp and lift. Sometimes it manages an awkward grasp (not nicely enclosed, but half levitating with some gaps between object and jaws) but usually drops it. This won't be fixed changing the fully closed to 0, the position of gripper is not precise enough. Even a few it did grasp, it didn't drop it off.

Works better with DR off. Sometimes it grasp okish but then randomly drops it, it look a bit like an artifact of the recovery/partial grasp (policy seems to learn to just drop???), have to revisit how this is implemented/perhaps do not use this feature. Maybe it's similar story with wrong approach perturbation? It learns to go approach a bit off?
I'm thinking, since the training just sees random subset of steps in trajectory (actually how many subsequent steps does it see during training?), it doesn't actually learn the intended "fix your approach"/"fix your grasp", it learns to approach awkwardly and randomly drop.


