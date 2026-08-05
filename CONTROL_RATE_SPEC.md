# Control-rate & action-format spec (sim → real parity)

Purpose: pin down the effective control rate and the action representation the
VLA policy is trained on, so the real AR4 MK3 driver applies actions at the same
rate and in the same units. A mismatch here looks like a "bad policy" but is
actually a units/rate bug, so this is worth checking before a hardware run.

All file:line references are to the state of the repo when this was written;
re-verify if the pipeline changes.

## Timing

**Current triple: `record_every 5` × `--skip 2` / `n_substeps = 10` ⇒ 50 Hz.**
These are defined once in `aera/autonomous/envs/task_env_factory.py`
(`COLLECTION_RECORD_EVERY`, `DATASET_SKIP`, `DEPLOY_N_SUBSTEPS`,
`DEPLOY_REPLAN_STEPS`, `DEPLOY_MAX_EPISODE_STEPS`) and consumed by
`Ar4Mk3InterfaceConfig`, `SuiteConfig` and `run_policy_on_env.Args` — the last
two previously kept separate copies and drifted (`n_substeps` 3 vs 20).

| quantity | value | source |
|---|---|---|
| sim timestep | `0.002 s` | `aera/.../ar4_mk3/ar4_mk3.xml` (`<option timestep="0.002">`) |
| collection `record_every` | `5` | `COLLECTION_RECORD_EVERY` → `Ar4Mk3InterfaceConfig.record_every` |
| dataset `--skip` | `2` | `DATASET_SKIP`; the transform invocation |
| `n_substeps` | `10` at deploy | `DEPLOY_N_SUBSTEPS`; env default is still 20, always set it |
| `replan_steps` | `4` (= 80 ms between inferences) | `DEPLOY_REPLAN_STEPS` |
| **env step** | **`n_substeps × 0.002 s`** = `0.02 s` = **50 Hz** | one action is held for all `n_substeps` substeps |

> **The invariant is `n_substeps = record_every × skip`, not `n_substeps =
> skip`.** Decimation happens twice — once at collection (`record_every`
> mj-steps per recorded frame) and once at transform (`--skip` recorded frames
> per dataset frame) — and only the product is a rate. `--skip` alone has not
> been the deploy rate since `record_every` landed.
>
> The two are exactly equivalent as data: the transform pairs `state[t]` with
> `state[t + skip]` in *recorded* frames (`offset = skip - 1`, and the
> collector's action is the next recorded state), so a delta spans
> `record_every × skip` raw steps and the frame stride matches. `record_every=5,
> skip=2` and `record_every=1, skip=10` produce the same dataset. The split is a
> cost choice, not a learning choice: recording is ~99% of a collected frame's
> wall clock (measured 13.7 ms of camera renders against 0.09 ms of physics, and
> the renders are per-call, not per-pixel), so a larger `record_every` collects
> proportionally faster, while a smaller one leaves room to re-transform at a
> finer rate without re-collecting.
>
> `record_every` is written into each episode's metadata;
> `convert_data_to_lerobot` echoes it and **refuses to build a dataset from
> episodes recorded at mixed values**, since their deltas would span different
> amounts of sim time with nothing left to distinguish them. Recordings from
> before it landed have no key and are read as `1`.

- **Eval / deploy** (`run_policy_on_env.py`) applies **one policy action per
  `env.step`** (action chunk, `replan_steps` applied one-per-step). `n_substeps`
  is exposed as `--n_substeps`, so the decision interval is whatever you set.
- **Collection** (`Ar4Mk3RobotInterface`) records via `_record_step()` on
  **every `record_every`-th mj-step**, timestamped with wall-clock
  `time.time()`. One counter serves all four call sites (IK loop, gripper ramp,
  `_settle`, `go_to_qpos`) so the stride stays uniform across an episode.
- The source LeRobot dataset `fps` is computed as
  `round(total_frames / total_wall_clock_duration)`
  (`convert_data_to_lerobot.py`) — **machine-dependent, not a clean sim
  rate**. Do not use it to time the real control loop.
- `transform_skip_dataset.py` subsamples by **recorded** frame count (`--skip`),
  so an action's true timescale is **`record_every × skip × 0.002 s` of sim
  time**. It takes `--record-every` purely to print that rate; it transforms
  nothing with it.

### ⇒ skip is a learning choice; matching is a deploy choice

`--skip` is a **data/learning** hyperparameter: it sets how far apart the paired
frames are so the per-step delta carries signal (recording is per mj-step, so a
too-small skip yields near-zero deltas and the policy can learn to sit still).
Pick it for learnability. It is **not** a control-rate setting and is **not**
enforced against any `n_substeps` at the transform layer.

The faithful-deploy invariant is satisfied **downstream**, by configuring the
applier to one decision per training delta:

```
deploy decision interval  ==  record_every × skip × 0.002 s
   sim:   set env n_substeps = record_every × skip  (run_policy_on_env --n_substeps)
   real:  run the driver loop at 1 / (record_every × skip × 0.002 s) Hz
          e.g. product=20 → 25 Hz, product=10 → 50 Hz, product=3 → 167 Hz
```

Note skip=3 (the old datasets) implies a **167 Hz** faithful deploy, which is
not achievable on hardware — one of the reasons the current pair is skip=10.

So train with whatever skip learns best, then set sim `n_substeps` (and the real
loop rate) to match it. **Record the skip alongside the dataset** (e.g. in the
repo name) so the deploy side knows what to match. If you're targeting hardware,
keep the achievable real control rate in mind when choosing skip — but that's a
planning consideration, not something the transform enforces.

## Action vector (per frame)

Built in `trajectory_data_collector.py` (`action` = the *next* recorded state);
re-paired and optionally delta-converted in `transform_skip_dataset.py`.

With `--delta-actions` (`num_joint_dims=6`):

| index | meaning | representation |
|---|---|---|
| `0..5` | arm joints `joint_1 … joint_6` | **delta** `state[t+skip] − state[t]` (radians) |
| `6`    | gripper | **absolute** (not delta) |

- **Joint order**: `joint_1 … joint_6` (the env's `arm_joint_names` order).
- **Gripper convention**: policy raw output is **−0.014 = open, 0 = closed**;
  `run_policy_on_env._denormalize_gripper` maps it to the env's normalized
  `[-1 = closed, +1 = open]`. Constants live in
  `aera/autonomous/envs/jaw_geometry.py`.
- **Gripper is binarized** (`transform_skip_dataset --binarize-gripper`): the
  action takes **exactly two values, `-0.014` and `0`**, nothing between, so a
  deployed policy commands full-open or full-close and never a partial close.
  The **state** channel stays continuous (holding a block reads ≈ −0.0088 /
  −0.0102 / −0.0112 for the 19/22/24 mm blocks) and is input only.
- **The jaw actuators are force-limited to ±10 N** (`act8`/`act9`
  `forcerange`), which is what makes "just close all the way" safe. A full
  close leaves a ~10 mm position error against `kp=10000`, so unlimited it
  applies ~102 N per jaw — measured, that pressed the pads 0.7 mm into a 24 mm
  block and made them buzz against it. **The real gripper must be
  current/torque-limited to roughly this level**, or the same command will
  stall a position-controlled servo at max current against a rigid block.
  This is now a sim/real parity requirement, not just a caveat.
- **Episodes start with the jaws OPEN** (both `qpos` and the `act8`/`act9`
  `ctrl` seed). A deploy loop must not begin by commanding them shut.
- **Env interpretation** (`ar4_mk3_base._set_action`, `absolute_state_actions=
  False`): `arm_target = current_qpos + action[:6] * relative_action_scale`;
  gripper `ctrl = -0.014 * (action[6] + 1) / 2` (so `+1 → -0.014` open,
  `-1 → 0` closed).
- **`relative_action_scale`** (`Ar4Mk3EnvConfig`): the dataset stores joint
  deltas in *radians* (Unnormalize restores physical units), so a delta policy
  is applied at **`1.0`** — `run_policy_on_env` defaults to it. Use `0.05` only
  for a policy whose arm output is normalized to ~[-1, 1]; applying `0.05` to a
  radian-delta checkpoint moves the arm ~20× short.

## Verify against the real driver / trained checkpoint

1. **`n_substeps == record_every × skip`** at deploy (sim) and **real loop rate
   == 1/(record_every × skip × 0.002 s)** — match the rate to the *product* the
   dataset was built with. `convert_data_to_lerobot` prints the `record_every`
   it found; `transform_skip_dataset` prints the resulting rate.
2. **Arm delta units / scale.** Dataset deltas are *raw radians over `skip`
   steps*; apply them with `relative_action_scale = 1.0` (NOT `0.05`). Spot-check
   end-to-end against the checkpoint's norm stats.
3. **Gripper dims.** The collector records **2** jaw joints but the action is
   **1** gripper command (`convert_data_to_lerobot` action_dim = `6 arm + 1/2
   gripper`). Confirm how the gripper is collapsed in the trained dataset.

## Real-robot deploy spec (honor this)

- Run the VLA at the deploy rate **`1 / (skip × 0.002 s)`** for whatever skip the
  dataset used (e.g. 25 Hz for skip=20, 50 Hz for skip=10).
- Each inference → **6 relative joint-delta targets** (radians, `joint_1…6`
  order) applied as a relative move from current joint positions at scale `1.0`,
  **+ 1 absolute gripper** command.
- Gripper: open = `-0.014`, closed = `0` (or normalized `+1 = open`,
  `-1 = closed`, depending on where you denormalize — match
  `run_policy_on_env`).
