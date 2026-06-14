# Control-rate & action-format spec (sim → real parity)

Purpose: pin down the effective control rate and the action representation the
VLA policy is trained on, so the real AR4 MK3 driver applies actions at the same
rate and in the same units. A mismatch here looks like a "bad policy" but is
actually a units/rate bug, so this is worth checking before a hardware run.

All file:line references are to the state of the repo when this was written;
re-verify if the pipeline changes.

## Timing

| quantity | value | source |
|---|---|---|
| sim timestep | `0.002 s` | `aera/.../ar4_mk3/ar4_mk3.xml` (`<option timestep="0.002">`) |
| `n_substeps` | configurable (default `20`) | `ar4_mk3_config.py` (`n_substeps: int = 20`); overridable per run |
| **env step** | **`n_substeps × 0.002 s`** (default `0.04 s` = 25 Hz) | one action is held for all `n_substeps` substeps |

- **Eval / deploy** (`run_policy_on_env.py`) applies **one policy action per
  `env.step`** (action chunk, `replan_steps` applied one-per-step). `n_substeps`
  is exposed as `--n_substeps`, so the decision interval is whatever you set.
- **Collection** (`Ar4Mk3RobotInterface`) records via `_record_step()` on
  **every IK mj-step (~0.002 s sim), not throttled**, timestamped with
  wall-clock `time.time()`.
- The source LeRobot dataset `fps` is computed as
  `round(total_frames / total_wall_clock_duration)`
  (`convert_data_to_lerobot.py`) — **machine-dependent, not a clean sim
  rate**. Do not use it to time the real control loop.
- `transform_skip_dataset.py` subsamples by **frame count** (`--skip`), so a
  recorded action's true timescale is **`skip × 0.002 s` of sim time**.

### ⇒ skip is a learning choice; matching is a deploy choice

`--skip` is a **data/learning** hyperparameter: it sets how far apart the paired
frames are so the per-step delta carries signal (recording is per mj-step, so a
too-small skip yields near-zero deltas and the policy can learn to sit still).
Pick it for learnability. It is **not** a control-rate setting and is **not**
enforced against any `n_substeps` at the transform layer.

The faithful-deploy invariant is satisfied **downstream**, by configuring the
applier to one decision per training delta:

```
deploy decision interval  ==  skip × 0.002 s
   sim:   set env n_substeps = skip   (run_policy_on_env --n_substeps)
   real:  run the driver loop at 1 / (skip × 0.002 s) Hz
          e.g. skip=20 → 25 Hz, skip=10 → 50 Hz, skip=3 → 167 Hz
```

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
  `[-1 = closed, +1 = open]` (`GRIPPER_CLOSED_ACTION = -1.0`).
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

1. **`n_substeps == skip`** at deploy (sim) and **real loop rate == 1/(skip ×
   0.002 s)** — match the rate to the skip the dataset was built with.
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
