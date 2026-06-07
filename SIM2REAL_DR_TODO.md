# Sim2Real DR backlog — physics / control / motion

Temporary working doc. We have strong visual + shape DR; these axes close the
remaining *dynamics / behaviour* gap so the real AR4 lands inside the training
distribution. Item #3 (observation/sensor noise) is deliberately deferred.

Tackle one at a time. Mark items done as we land them.

---

## [ ] #1 — Robot actuator & joint physics DR  ← IN PROGRESS

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

## [ ] #4 — Contact & environment physics

- Surface friction: `geom_friction` on table top (`floor`), room floor, and
  gripper pads (currently only the block's friction is randomized).
- Contact softness / restitution: `geom_solref` / `geom_solimp` on object +
  table.
- COM / inertia offset on the block (not just mass) → in-hand rotation.
- Gripper jaw dynamics: jaw damping/frictionloss/solimplimit, close force (kp on
  act8/act9), close speed.
- Base/table level: tilt `model.opt.gravity` 1-3 deg; tiny `opt.timestep` jitter.

## [ ] #5 — Motion / trajectory dynamics (beyond pre-grasp waypoints)

Extends `trajectory_perturbation.py` (today: pre-grasp disk waypoints + IK-param
noise + home offset).

- Speed / velocity-profile variation per episode/segment.
- Overshoot & settle variation (partly emergent from #1).
- Mid-trajectory pauses / jerk.
- Approach-angle + grasp-height variation; release-height variation at place.
- Give distal cable-harness arcs a little mass/inertia (currently visual-only).

---

## Deferred

## [ ] #3 — Observation / sensor noise (images + proprioception)

Image aug (brightness/contrast/gamma, hue/WB, gauss+shot noise, motion blur,
vignette, JPEG, dropped/frozen frame) + proprioception noise/bias + perception-
grade noise/dropout on any object-pose obs. Revisit after #1, #2, #4, #5.
