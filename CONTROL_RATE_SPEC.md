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
| `n_substeps` | `20` | `aera/autonomous/envs/ar4_mk3_config.py` (`n_substeps: int = 20`) |
| **env step** | **`0.04 s` = 25 Hz** | `n_substeps × timestep`; one action is held for all 20 substeps |

- **Eval / deploy** (`aera/autonomous/openpi/scripts/run_policy_on_env.py`)
  applies **one policy action per `env.step`** (action chunk, `replan_steps`
  applied one-per-step) → **25 Hz, one delta-joint action per 0.04 s**.
- **Collection** (`Ar4Mk3RobotInterface`) records via `_record_step()` on
  **every IK mj-step (~0.002 s sim), not throttled**, timestamped with
  wall-clock `time.time()`.
- The source LeRobot dataset `fps` is computed as
  `round(total_frames / total_wall_clock_duration)`
  (`convert_data_to_lerobot.py:131`) — **machine-dependent, not a clean sim
  rate**. Do not use it to time the real control loop.
- `transform_skip_dataset.py` subsamples by **frame count** (`--skip`), so a
  recorded action's true timescale is **`skip × 0.002 s` of sim time**.

### ⇒ Parity requirement (load-bearing)

For the training action timescale to match the 25 Hz the eval loop (and the real
driver) apply actions at:

```
skip = n_substeps = 20      ( 20 × 0.002 s = 0.04 s = 25 Hz )
```

The `transform_skip_dataset.py` docstring examples use `skip=5` / `skip=10`,
which give 0.01 s / 0.02 s action deltas — **4× / 2× faster** than the deploy
loop applies them. If the trained dataset used `skip ≠ 20`, the arm moves at the
wrong speed on hardware even with a perfect policy. **Verify the skip used for
the trained dataset.**

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
  False`): `arm_target = current_qpos + action[:6] * 0.05` (relative); gripper
  `ctrl = -0.014 * (action[6] + 1) / 2` (so `+1 → -0.014` open, `-1 → 0` closed).

## Verify against the real driver / trained checkpoint

1. **`skip = 20`** for the trained dataset (see parity requirement above).
2. **Arm delta units.** Dataset deltas are *raw radians over `skip` steps*, but
   the env applies `action[:6] * 0.05`, with openpi action normalization in
   between. Confirm the **end-to-end arm scale** using the checkpoint's
   norm stats — a stray `0.05` factor would move the arm ~20× wrong.
3. **Gripper dims.** The collector records **2** jaw joints but the action is
   **1** gripper command (`convert_data_to_lerobot` action_dim = `6 arm + 1/2
   gripper`). Confirm how the gripper is collapsed in the trained dataset.

## Real-robot deploy spec (honor this)

- Run the VLA at the deploy rate: **25 Hz** if `skip = 20` (else `1 / (skip ×
  0.002 s)`).
- Each inference → **6 relative joint-delta targets** (radians, `joint_1…6`
  order) applied as a relative move from current joint positions, **+ 1 absolute
  gripper** command.
- Gripper: open = `-0.014`, closed = `0` (or normalized `+1 = open`,
  `-1 = closed`, depending on where you denormalize — match
  `run_policy_on_env`).
