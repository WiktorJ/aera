# Sim2Real DR backlog — physics / control / motion

Temporary working doc. We have strong visual + shape DR; these axes close the
remaining *dynamics / behaviour* gap so the real AR4 lands inside the training
distribution. Item #3 (observation/sensor noise) is deliberately deferred.

Tackle one at a time. Mark items done as we land them.

---

## [x] #1 — Robot actuator & joint physics DR  ← DONE

The arm is currently a near-perfect position tracker (kp 20000/5000) with fixed
joint damping and zero friction/armature variation. Randomize per episode:

- Actuator gains: `actuator_gainprm[:,0]` (kp) + `actuator_biasprm[:,1:3]`
  (= -kp, -kv) for act1..act6.
- Joint friction / damping / inertia: `dof_damping`, `dof_armature`,
  `dof_frictionloss` on joint_1..joint_6.
- Torque saturation: `actuator_forcerange` (reduce only).
- (later / stretch) backlash / deadband approximation.

Plumbing: `ArmDynamicsConfig` dataclass → `DomainRandConfig.arm_dynamics` →
sampler in `domain_rand_config_generator.py` → applied in
`Ar4Mk3Env._apply_domain_randomization`. Ranges must stay conservative enough
that the scripted pick/place demo collector still succeeds.

## [x] #2 — Control-loop realism (latency, lag, jitter)  ← DONE (collection path)

Implemented in the demo-collection stepping path (the robot interface), since
that's the only path that changes the trained VLA under pure offline imitation.
Per-episode `ActuationConfig` (latency_steps / command_lag_alpha /
step_jitter_prob) applied to the arm actuators (act1..act6; gripper left crisp)
right before each `mj_step` in `_step_simulation` and the in-place IK loop.

- `ActuationConfig` on `Ar4Mk3InterfaceConfig` (identity defaults = no-op).
- `ActuationPerturbation` ranges + `perturb_actuation` flag on
  `PerturbationConfig`; `sample_actuation_config()` resolves per episode.
- Enable via `--perturbation.perturb-actuation` on collect_trajectories.py /
  demo_pick_and_place.py.

Deferred from this item (overlaps #3 / needs the env path):
- Encoder quantization + per-joint observed-qpos bias → folded into #3.
- Env-path latency for honest eval / RL-in-the-loop (run_policy_on_env) → only
  matters once we do in-sim rollouts that feed back into training.

## [~] #4 — Contact & environment physics  ← SKIPPED (low value here)

Decided not to pursue. Blocks sit stably in the workspace (no meaningful
lift-friction issue), and post-release settling doesn't matter because the
deterministic kinematic controller takes over once the policy releases. The one
genuinely high-value contact axis — friction-based grasp slip — is removed by
the kinematic grasp lock, and the rest (table friction, solref/solimp, COM
offset, gravity tilt) is low-value given that. Revisit only if we ever do
contact-tuning to retire the grasp lock.

Instead, addressed the grasp lock's real downside (see below).

## [x] Grasp lock mirrored into eval  ← DONE (came out of the #4 discussion)

The kinematic grasp lock existed only in the data-collection interface; eval
(`run_policy_on_env` → `env.step`) grasped by raw friction physics — the same
unstable contacts that made physical-grasp collection unusable — so eval success
was confounded and under-reported policy quality. Fix: share one lock
implementation and apply it in eval too.

- `KinematicGraspLock` extracted to `aera/autonomous/envs/kinematic_grasp.py`;
  the collection interface now delegates to it (no behavior change).
- `Ar4Mk3Env` uses it behind the `kinematic_grasp` config flag: engage on
  gripper-close command + 5 cm proximity, enforce per-substep (n_substeps=20).
- Enabled by default in `run_policy_on_env` (`--no-kinematic-grasp` to eval
  under raw friction instead).

Future (separate, optional): contact-tuning to make physical grasping stable
enough to retire the lock — gives genuine grasp-robustness + slip-recovery in
the data. Big effort, deferred.

## [x] #5 — Motion / trajectory dynamics  ← DONE (core)

Two composable, opt-in perturbations on `PerturbationConfig` (orthogonal to
`mode`), sampled per episode and baked into the interface config like
`ik_noise` / `actuation`:

- `perturb_speed` — one per-episode tempo factor (U(0.7,1.4)) scales IK step
  size (integration_dt, max_update_norm) up, interpolation step counts
  (go_home, gripper) down, and IK max_steps up (slow-episode convergence
  headroom). Varies recorded action-delta scale + frame cadence.
- `perturb_hover_height` — per-episode `above_target_offset` (U(0.04,0.10))
  varying the pre-grasp / pre-place hover geometry.
- Enable via `--perturbation.perturb-speed` / `--perturbation.perturb-hover-height`
  on collect_trajectories.py / demo_pick_and_place.py.

Deliberately skipped (see discussion):
- Overshoot/settle — already emerges from #1's soft-gain episodes.
- Approach-angle / tilted grasp — covered by offset_approach's XY disk;
  tilting the final grasp risks the jaws not seating squarely.
- Mid-trajectory dwell — low value, injects near-no-op frames.
- Cable-harness mass — not worth the effort; cables stay visual-only.

---

## [x] #3 — Observation / sensor noise (images + proprioception)  ← DONE

Key findings that shaped it:
- The VLA consumes only {image, gripper_image, state(7-dim qpos), prompt}. The
  25-dim goal-env observation (object pose) is NOT fed to the policy → object-
  pose noise is N/A, skipped.
- openpi already does train-only resampled RandomCrop/Rotate/ColorJitter; this
  layer is the SENSOR gap (noise/blur/vignette/WB/jpeg/grayscale/frozen frame)
  + proprioception, which openpi does not touch.
- Lives in the offline dataset-prep stage (not collection, not the openpi
  train transform): train-only, per-episode context, and decoupled from
  expensive collection so magnitudes can be retuned by re-running the cheap
  offline pass.

Implementation:
- Shared module `aera/autonomous/obs_augmentation.py`: per-episode
  `CameraProfile` + `augment_image` (full sensor set); per-episode
  `StateNoiseProfile` + `apply_state_noise` (bias + jitter on STATE input only,
  never the action target — self-consistent with delta actions).
- Offline: `transform_skip_dataset.py` flags `--image-aug` / `--state-aug` /
  `--obs-aug-strength` / `--obs-aug-seed`; per-episode profiles applied in the
  transform loop (incl. frozen-frame reuse). Guard: `--state-aug` rejects
  `--smooth-state` (would low-pass the jitter).
- Eval (behind flag): `Ar4Mk3EnvConfig.obs_image_aug` augments env-render obs
  with the same module (per-episode profile on reset); exposed as
  `--obs-image-aug` on run_policy_on_env.
- Verification: `aera/autonomous/openpi/scripts/preview_obs_augmentation.py`
  renders an original|aug|aug… grid (from --from-sim, --image, or synthetic).

Caveats / not done:
- Offline transform not run end-to-end here (needs a real LeRobot dataset);
  shared funcs + eval path are tested, integration is wiring on top.
- Object-pose noise: N/A (not consumed by the VLA).
