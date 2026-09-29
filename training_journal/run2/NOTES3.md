# Training journal — 2026-09-05

Grasp-pose jitter sampling: **uniform → truncated Gaussian**. Continues
[04.09](../04.09.2026/NOTES.md) (item 3 of [01.09](../01.09.2026/NOTES.md)).
Same axes and bounds; only the *shape* of the per-episode draw changes. Old
uniform path removed — no flag to keep it.

## Why

The reservation against jittering every grasp is teaching the policy to grasp
*sloppily*. Uniform ±max makes a moderately-off grasp as common as a perfect
one, and pi05's flow head can reproduce that spread as habitual sloppiness (it
models the marginal, it doesn't collapse to the mean). Zero-peaking fixes this:
the **mode is a perfect grasp**, the spread is a covered tail, not the norm.

## The test — when zero-peaked beats uniform

All three must hold:

1. **Terminal target the policy imitates** (endpoint or committed approach), not
   world/dynamics it must be robust to, not an intermediate waypoint.
2. **Observation-independent** — the offset can't be predicted from the image,
   so the only thing an expressive head can do with it is reproduce it as
   deployed scatter.
3. **Nominal is the desired-typical** — the clean value is what you want to stay
   dominant; the spread is a tail worth covering, not a behaviour worth adopting.

When any fails, uniform is right — and for the sim2real DR levers it is
*actively* right: `ik_noise`, `actuation`, arm-dynamics, cameras randomize a
range whose nominal you don't trust, so flat coverage is the goal and peaking at
nominal would concentrate mass on the sim-ideal reality isn't. `hover_height`,
`offset_approach`, `home`, `speed` are approach/observation diversity — breadth
is the point. **None of these change.**

## Per-axis (the four grasp-target axes all pass the test)

| axis | held behaviour | σ | role of the shape |
|---|---|---|---|
| **finger** | **persists** (jaws don't recentre along their length) | **3.5 mm** (bound 7) | mode at perfect keeps aim true **and** the tail supplies held-crooked recovery demos — the one axis whose *job* is the tail, so it runs the widest σ |
| **yaw** | self-corrects on close | 4.0° (bound 12) | held endpoint is aligned regardless; shape only keeps the *approach* aligned so the policy commits to the close when slightly off without adopting a habitually-yawed approach |
| **pinch** | self-corrects on close | 0.6 mm (bound 1.5) | same as yaw (approach diversity) |
| **height** | persists, one-sided up | 0.8 mm (bound 2) | half-normal peaked at the nominal grasp depth |

Finger is the split: it's persistent, so its Gaussian both preserves precision
and carries the recovery signal — σ=3.5 keeps ~11% in the >5 mm stall band
(~340 demos at 3000 eps), vs ~4% at σ=2.5 (too tail-starved) or ~29% uniform
(sloppiness pull). The self-correcting axes need no tail, so they run tight.

Bonus of the shape: with the bound at ~2–3σ it becomes a **soft tail**, so the
exact `*_max` is no longer load-bearing — which is the honest answer to "we
can't measure where 'low yaw' ends, ±12° just feels right from the real thing."

## Why no friction sweep (yaw self-correction is representative)

Yaw *could* fit the persistent-axis case if real friction stopped the free block
rotating to align before the lock engages. It doesn't, in our regime: gripper and
blocks are **PLA-on-PLA** — hard, low-friction, non-deformable — so the aligning
torque beats rotational friction at the small offsets we operate at, and the
sim's self-correction matches the real thing (confirmed by hand). The held-yaw
stall the eval does show (01.09, up to 35.7°) comes from the *policy's* non-clean
closes, which the scripted expert can't reproduce anyway — so jittering the
scripted approach yaw can't manufacture those recovery demos, whatever the shape.
Recovery for held-yaw belongs to post-training DAgger, not this lever.

## Mechanism note (why finger ≠ yaw under the same jitter)

The grasp is a `KinematicGraspLock` that captures the object pose *at engage*.
Finger offset is still present at engage → the lock captures it → recovery demo.
Yaw/pinch have settled by engage → the lock captures an aligned grasp → approach
diversity only. So the jitter *shape* controls the held finger distribution
directly, but only the *approach* distribution for yaw/pinch.

## Validation

Samplers (200k draws): truncate cleanly at the bound, means ~0 (height +0.62 mm),
std = σ, reproducible under `np.random.seed` (scipy truncnorm on the numpy global
RNG, so collection seeding is unchanged).

Sim probe `measure_scripted_arm grasp-jitter`, 10 seeds, dt=0.009, shipped shape:

- **locked 10/10** — yield intact.
- **no overshoot** — finger max 5.39/7, pinch 0.40/1.5, yaw 0.21/12, `over=0/10`.
- **mechanism confirmed in the held offsets:** finger persists (held up to
  5.39 mm, tracking the command), yaw self-corrects to **0.01–0.21°**, pinch to
  ≤0.4 mm. Exactly the finger-carries-recovery / yaw-carries-approach split.

The subcommand now defaults to the shipped shape (all four axes on), takes per-
axis σ flags, and prints per-axis max/p50/p90/p99 + an overshoot count.

## Runbook — the combined filter + grasp-jitter re-collect

Builds `05_09_2026` from raw. Flags pinned to the 08_08 build (`record_every=5`,
`--skip 2` → net 10, `--delta-actions`, `--binarize-gripper`, `--squeeze-gripper`
default, `--frame-skip 1` default, no smoothing, no go-home exclusion; the jaw
action stats on `08_08…skip10_delta` are exactly {−0.014, 0}, which is how
`--binarize-gripper` was confirmed). **Only two changes vs 08_08:** grasp-jitter
in the raw (Gaussian, this entry) + the static filter at transform. So the
retrain is a clean A/B.

```bash
# 1. Collect (~2.2 h; grasp-jitter is already in collect_mixed.sh's lever stack)
cd /home/wiktor/Projects/aera
PYTHON_BIN=/home/wiktor/Projects/aera/.venv/bin \
  ./semi_autonomous/aera_semi_autonomous/scripts/collect_parallel.sh \
  3100 data/aera_semi_pnp_dr_05_09_2026 1000 8
# 3100 attempts → ~2959 saved at 08_08's 95.5% yield (jitter costs little; probe locked 10/10)

# 2. Convert raw -> lerobot  (--output-dir supplies the name only; writes to the
#    lerobot cache via repo_id, no on-disk duplicate)
uv run python semi_autonomous/aera_semi_autonomous/scripts/convert_data_to_lerobot.py \
  --data-dir data/aera_semi_pnp_dr_05_09_2026/episodes \
  --output-dir aera_semi_pnp_dr_05_09_2026 \
  --push-to-hub

# 3. Transform: skip2 (net 10) + delta + binarize + go-home exclusion + static filter.
#    --exclude-prompts does NOT change the auto-name, and the transform SKIPS
#    (silently reloads stale data) if the output cache dir already exists — so
#    name the output explicitly with the exclusion. --output-repo-suffix replaces
#    the WHOLE tag, so it must carry skip10_delta too, not just no_go_home.
uv run python aera/autonomous/openpi/scripts/transform_skip_dataset.py \
  --repo-id Purple69/aera_semi_pnp_dr_05_09_2026 \
  --skip 2 --delta-actions --binarize-gripper \
  --exclude-prompts "go home" \
  --output-repo-suffix "skip10_delta_no_go_home" \
  --min-action-delta 0.0005 --gripper-eps 0.0002 \
  --push-to-hub
# -> Purple69/aera_semi_pnp_dr_05_09_2026_skip10_delta_no_go_home

# 4. Health check the output (check 5: grasp window intact, residual dwell)
uv run python aera/autonomous/openpi/scripts/check_dataset_health.py \
  --repo-id Purple69/aera_semi_pnp_dr_05_09_2026_skip10_delta_no_go_home
```

Confirm at step 4 that the filter left the grasp window ≥3 frames and the
residual near-static run is p90≈5 — that p90 (a touch above) is what sets the
next policy's `replan`, not the full horizon.

### Health check (10.09) — at 08_08 parity, go for retrain

`_skip10_delta_no_go_home`: **check 4 (the true gate) PASSES 87.8%** (08_08:
88.2%). Checks 2 and 3 "fail" but are identical to the 08_08 baseline that
trained the working policy, and both now trace to the same accepted slow-arm
timescale, not to go-home or the jitter:

- check 2: the *tail* ratios all pass (p99:med 1.98≤2.5, max:med 2.99≤3.0 —
  cleaner than 08_08's 3.25, below_0.25x 0.031). The lone fail is `eef_median
  2.28 mm < 2.4` — per-step arm speed (collection integration_dt), 08_08 is 2.27.
- check 3: descent max 7.49 mm — 08_08 is the identical 7.49.

Leave the timescale alone (do not drop to skip 5–6 as the check suggests):
08_08 shipped a working policy at exactly it, and changing skip would break the
A/B and the deploy `n_substeps=10` invariant. Grasp window median 4 / min 3, 0
close-less — the filter behaved as validated.

### Go-home: excluded, and the name now says so

Decision: **exclude go-home from training; script it at deploy.** Rationale:

- **Deploy has no go-home phase.** `run_policy_on_env._run_episode` runs only
  PHASE_PICK / PHASE_PLACE and ends on `_is_success` (object at goal). The arm
  never returns home in a rollout, and the pick→place switch is already an
  external trigger (`_stdin_pressed`) — sequencing is orchestrated, not learned.
- **Go-home is control, not perception** — a fixed joint interpolation to a known
  home config, which the collector already scripts. Learning it would spend model
  capacity (and the quantile-normalization range) on the one perceptually
  contentless part of the task.
- **Learning it buys zero autonomy.** Using a learned go-home still needs
  placement detection + a prompt switch — the same machinery the scripted
  fallback uses (`_is_success` already computes the trigger). So scripting wins.
- **Any inclusion damages the descent.** Normalization is over all action deltas,
  prompt-agnostic, so the go-home sprints (joint Δ to 0.15 rad vs the task's
  ~0.035) compress the descent's share of [−1,1] — measured 21% vs 88%. Even
  go-home-under-its-own-prompt would poison it; not a labeling problem.
- **Aliasing** (return-empty vs approach-empty gripper) is a real one-sided cost
  the prompt could only fix by driving it — which again needs the detection.

Naming: the go-home-excluded set carries `no_go_home` in its name from now on.
Not doing so on 08_08 (which is *also* go-home-excluded, just named plain
`skip10_delta`) is what made an earlier flag-pinning infer the opposite from the
name. Verify from frames, not names.


## Next

Retrain on `05_09_2026_skip10_delta_no_go_home`; set deploy `replan` from step 4's residual
dwell (~5–6). Deployed-grasp precision is falsifiable via the `eval/grasp/*`
metrics: mean offset should stay ~on-object (no drift = the Gaussian mode held),
scatter within the ~7 mm band where recovery is now demonstrated.
