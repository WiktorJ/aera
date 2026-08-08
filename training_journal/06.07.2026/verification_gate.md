# Verification gate — run this before launching the next training

Branched from [next_run_changes.md](./next_run_changes.md) (the "Verify on the first new data" section) and [implementation_plan.md](./implementation_plan.md) (item V3).

**Why:** training is the expensive step, and every load-bearing claim the next run rests on is measurable on a small batch of data. The previous run spent its whole budget on demonstrations whose scripted close never touched the block — a failure that was visible in a 3-seed headless run and in the collection log, and that no amount of eval analysis could have diagnosed after the fact.

**Rule:** nothing proceeds to the next stage until the current one passes. Each failure below routes to either **transform** (cheap — re-run the transform) or **re-collect** (expensive — fix collection first).

Thresholds throughout are the slow-arm / net-decimation-10 targets (`record_every=5` × `--skip 2`; see [CONTROL_RATE_SPEC.md](../../CONTROL_RATE_SPEC.md)). If the batch sends you to a different rate, pass the new expectations via `check_dataset_health`'s threshold flags rather than editing the tool.

---

## Stage 0 — pre-collect sanity (no data needed, ~2 min)

Catches the "0% locked grasps" class of failure *before* spending a collection run on it.

```bash
# timing: does the slow arm land where it should, and does the lock engage?
python semi_autonomous/aera_semi_autonomous/scripts/measure_scripted_arm.py \
    timing --dt 0.009 --max-steps 3000 --seeds 0 1 2

# close depth: does the scripted close actually reach the block?
python semi_autonomous/aera_semi_autonomous/scripts/measure_scripted_arm.py \
    close-sweep --dt 0.009 --max-steps 3000 --seeds 0

# camera parameterization: does eval draw from collection's distribution?  (needs F2)
python semi_autonomous/aera_semi_autonomous/scripts/check_camera_parameterization.py

# dead time: how much of the episode is the arm parked with nothing happening?
python semi_autonomous/aera_semi_autonomous/scripts/measure_scripted_arm.py \
    dwell --dt 0.009 --max-steps 3000 --seeds 0 1 2
```

| expect | if not |
|---|---|
| `sim_time_s` 3.0–4.0, `avg_speed_cm_s` 20–24, `raw_frames` 1450–2000. **This is the NOMINAL arm and it is meant to read faster than the 12–15 cm/s target** — that target applies to the arm as COLLECTED (DR on), measured 13.1 cm/s. Compare harness-to-harness only. | C1 — retune `integration_dt` / `max_steps` |
| **`lock_engaged=True`** on every seed | C4/C5/F1 — the close never reaches the block. **This is the failure that produced the last run.** |
| `close-sweep`: `pad_contacts > 0`, `pinching=True`, `lock=True` at the scripted target | F1 — the depth model still disagrees with the pad geometry |
| collection and eval camera rows agree under DR offsets | F2/E2 — eval is sampling an OOD camera distribution |
| `dwell`: `parked` ≤0.15, `parked_gripper_idle` ≤0.08 (measured 0.102–0.135 / 0.051–0.068 at `dt=0.009`) | see the dead-time note below. **Check the dataset number first** — Stage 3's `parked_below_0.05x_median` is what reaches training, and it reads 5.2%. Only if *that* is high is `gripper_action_steps` the remedy. |

### Gripper-close dead time — measure it, it doubled with C5 (added 04.08.2026)

Noticed in a manual demo: after the jaws close there is a visible freeze before
the arm lifts. It is expected, but it is worth watching because **C5 doubled it
and nothing downstream flags it**.

`_interpolate_gripper` parks the arm for the whole close (`data.ctrl[:6] =
data.qpos[...]`, set once before the loop) and runs `gripper_action_steps * 2`
= 100 mj-steps. It exits early only once the ramp finishes *and*
`‖target − current‖ < gripper_pos_tolerance` — and with a full-close target of
`0.0` and a block in the jaws that error is ~0.016, so it can never drop below
the 1e-4 tolerance. The loop therefore always burns its full budget. That
non-convergence is deliberate (it is what keeps driving the jaws in, see the
comment at the end of `_interpolate_gripper`) — the dead time is its price.

| | loop steps | sim time |
|---|---|---|
| before (target −11.5 mm, tol 1e-3) | 50 — converged right at ramp end | 0.10 s |
| **now** (target 0.0, tol 1e-4) | **100 — never converges** | **0.20 s** |

Measured on a 24 mm block, the jaws land within 50 µm of their resting position
by **step 14** and travel only another 88 µm after the ramp ends at step 50 —
so **86 of the 100 steps are visually static**, arm frozen and jaws already
clamped. `dwell` on a full episode: `parked=0.06`, of which half
(`parked_gripper_idle=0.03`, ~101 frames) is arm-parked *and* jaws-not-moving,
i.e. ~2 full-budget calls per episode.

Note the slow arm did **not** slow this: `gripper_action_steps` is a fixed
mj-step count, unaffected by `integration_dt`. In sim-time *proportion* the
close shrank (the intended dilution, ~20% of frames → ~3%), but in absolute
terms it doubled, and it is now a hard freeze surrounded by smooth slow motion.

**This is why the `dwell` thresholds are dt-dependent and were rescaled on
07.08.2026.** Because the count is fixed in mj-steps while the arm around it
speeds up, the parked *share* rises with `integration_dt`: measured `parked`
0.06–0.081 at `dt=0.005` and 0.102–0.135 at 0.009. Any future change to
`integration_dt` moves these thresholds with it — they are not absolute
properties of the arm. The dataset-level figure is the one that gates training,
and it stayed benign across the same change (2.8% → 5.2% below 0.05 × median,
against check 2's 0.10 limit on the 0.25 × term).

**If it needs trimming, the lever is `gripper_action_steps` (or the `* 2` budget
multiplier), NOT `gripper_pos_tolerance`.** Loosening the tolerance would
restore the early exit that C5 removed on purpose, and the jaws would stop short
of the block again — the exact failure that produced the last run. Dropping the
multiplier from `2 ×` to ~`1.3 ×` cuts the dead half while still delivering the
full ramp. Do not tune it before Stage 3 says it matters.

### Stage 0 measured at the landed defaults (07.08.2026, `dt=0.009`)

| check | measured | |
|---|---|---|
| `timing` | 2.98–3.96 s, 20.2–24.0 cm/s, `lock_engaged` **3/3** | PASS |
| `close-sweep` | full close → 8 pad contacts, `pinching=True`, `lock=True` | PASS |
| `check_camera_parameterization` | **4/4 poses agree** | PASS |
| `dwell` | `parked` 0.102–0.135, `parked_gripper_idle` 0.051–0.068 | PASS (rescaled) |

Baseline for comparison, measured 02.08.2026 on unfixed code: `dt=0.15` → 0.69–0.75 s, 103–107 cm/s, **`lock_engaged=False`**; `dt=0.005` → 4.81–6.61 s, 12.1–14.9 cm/s, **`lock_engaged=False`**.

---

## Stage 1 — collect 30 episodes

```bash
./semi_autonomous/aera_semi_autonomous/scripts/collect_mixed.sh 30 data/verify_batch 42 2>&1 | tee /tmp/collect_verify.log
```

**30, not 10.** Several checks are extreme-value statistics — check 3's descent
`max`, check 2's `max:median`, check 5's `min` frames — so they are set by the
single most extreme episode, and each episode contributes exactly one draw of
the per-episode speed factor. With `s ~ U(0.85, 1.15)`, `E[max of n]` is 1.123
at n=10 against 1.140 at n=30 (true max 1.15), so ten episodes systematically
under-samples the tail. Measured cost of that: two 10-episode batches of the
same configuration reported `max:median` 2.4 and 3.8. Thirty also samples the
90/10 `collect_mixed` split properly rather than giving `offset_approach` a
single episode. At ~20.5 s/attempt this is ~10 minutes.

Then read the log — these are warnings, not errors, so they scroll past silently:

```bash
grep -c "Grasp not locked"                     /tmp/collect_verify.log   # want 0
grep -cE "Max steps .* reached|could not move" /tmp/collect_verify.log   # want 0
grep    "Successfully collected"               /tmp/collect_verify.log   # want 27/27 + 3/3
grep    "Synchronized"                         /tmp/collect_verify.log   # want NON-ZERO on every line
```

| check | expect | failure routes to |
|---|---|---|
| 8 — lock engaged | ≥95% of episodes, i.e. 0 `Grasp not locked` | **re-collect** (C4/C5/F1) |
| 1 — IK budget | ≤1 `Max steps` / `could not move` in 30; measured 3.3% | **re-collect** (C1 `max_steps`) |
| 0 — data actually recorded | every `Synchronized N` line has `N > 0` | **fix the collector, then re-collect** |

### The `Synchronized 0` check is not redundant — it is the one that caught a real failure (added 05.08.2026)

`Successfully collected N/N` counts the *physical* task (did the block reach
the target), **not** whether a single training frame was written. When
`record_depth` first went off (C7), `_synchronize_all_data` still required a
depth match for every frame, so every episode wrote `trajectory_data: []` — and
**the whole gate stayed green**: 0 `Grasp not locked`, all collected, and
Stage 2 exiting 0 while building `total_frames: 0`. The failure only surfaced at
Stage 3, as a `RepositoryNotFoundError` 404 (LeRobot finds no local parquet and
falls back to the hub), which reads like an auth problem and sends you chasing
the wrong thing entirely.

Whenever a recording knob changes, check the frame count, not the success count.

Also record **wall-clock per attempt** and **disk per collected episode** (`du -sh data/verify_batch`), and sanity-check the recorded frame count: at `record_every=5` an episode should hold **~650** frames, not ~3300. If it holds ~3300 the decimation is not in effect and collection will run ~5× slower than it needs to.

Measured 07.08.2026 at the landed defaults (`dt=0.009`, `record_every=5`, one process): **20.5 s per attempt**, 96.7% yield, **26.1 MB per collected episode**, ~660 frames/episode. If wall-clock comes in much higher, the profile to check first is the two camera renders in `_record_step`, which measured 13.7 ms against 0.09 ms of physics.

---

## Stage 2 — convert + transform

```bash
# The dataset name comes from --output-dir's BASENAME: convert_data_to_lerobot
# builds repo_id = f"Purple69/{output_path.name}" and writes to the default HF
# cache path for it, so --output-dir names the dataset rather than placing it.
python semi_autonomous/aera_semi_autonomous/scripts/convert_data_to_lerobot.py \
    --data-dir data/verify_batch \
    --output-dir aera_semi_pnp_dr_<DDMMYYYY>_verify
# -> Purple69/aera_semi_pnp_dr_<DDMMYYYY>_verify

python -m aera.autonomous.openpi.scripts.transform_skip_dataset \
    --repo-id Purple69/aera_semi_pnp_dr_<DDMMYYYY>_verify \
    --skip 2 --delta-actions --binarize-gripper \
    --exclude-prompts "go home" \
    --output-repo-suffix skip10_delta_verify
# -> Purple69/aera_semi_pnp_dr_<DDMMYYYY>_verify_skip10_delta_verify
```

Four notes:
* **`--exclude-prompts "go home"` is load-bearing for check 4, not cosmetic.**
  `go_home` is an interpolated joint-space move, not IK: it sprints, and its
  deltas alone set the q01/q99 span every other frame is normalized against.
  Measured 05.08.2026 on the same 7-episode batch, dropping **93 frames
  (2.5% of the dataset)**:

  | | without the flag | with it |
  |---|---|---|
  | check 2 p99 | 36.76 mm | 3.68 mm |
  | check 2 max | 53.72 mm | 4.35 mm |
  | check 2 max:median | 32.4 | 2.6 |
  | **check 4 descent** | **14.6% — FAIL** | **99.2% — PASS** |

  Every earlier dataset carries `_no_go_home` in its name for this reason; the
  flag was simply missing from this runbook. Without it the headline gate reads
  as a catastrophic failure when the data is fine.
* **`--skip 2`, not 10.** Collection records every 5th mj-step
  (`COLLECTION_RECORD_EVERY`), so the net decimation is `5 × 2 = 10` and the
  deploy rate is unchanged at `n_substeps=10` / 50 Hz. The transform prints the
  resulting rate; check that line reads `= 10 substeps = 20 ms = 50 Hz`. If
  check 3 sends you to a finer rate, `--skip 1` gives net 5 without a
  re-collect — that headroom is why `record_every` is 5 rather than 10.
* **No `--min-action-delta`** (implementation plan T3 — with the arm parked during the ramp and a binarized gripper action, the idle filter deletes the grasp window).
* `--binarize-gripper` landed with **T1**. It snaps only the **action** gripper dims to `{-0.014, 0}` (`--gripper-binarize-threshold`, default `-0.013`); the state channel stays continuous, which is what check 6 asserts. There is also `--drop-leading-closed`, which is **not** needed here — C6 removed the leading closed frames at the source. Use it only when re-processing a dataset collected before that landed.

---

## Stage 3 — dataset checks

```bash
python -m aera.autonomous.openpi.scripts.check_dataset_health \
    --repo-id Purple69/<name>_skip10_delta_verify
```

Exit code is 1 if any check fails, so this can gate a pipeline directly. For the per-joint quantile detail behind check 4:

```bash
python -m aera.autonomous.openpi.scripts.measure_action_dynrange \
    --repo-id Purple69/<name>_skip10_delta_verify
```

| check | expect | `16_06` was | failure routes to |
|---|---|---|---|
| 2 delta distribution | median 2.4–3.0 mm (timescale); below-0.25×-median ≤10%, p99:median ≤2.5, max:median ≤3 | 3.28 mm / 9.2 max:median | median → **re-collect** (`integration_dt`); ratios → **transform** (skip) |
| 3 descent resolution | median ≤3.5 mm, max ≤7 mm | 0.78 / 11.4 | **transform** — drop to skip 5–6 (deploy `n_substeps` follows) |
| 4 **normalized output range** | descent ≥50% of span | **5.3%** | **transform** (skip) or **re-collect** (dt), depending which term is off |
| 5 grasp window | ≥3 frames/episode spanning open→closed, expect 5–8 | 4–5 | **transform** — duplicate / weight grasp-window frames |
| 6 binarization | 2 action values, 0 leading-closed frames, state continuous | 251 levels, 8–57 leading | **transform** (flags) or **re-collect** (C6 jaws-open) |
| 7 no recovery | 1 grasp cycle per episode | 3/3 with extra cycles | **re-collect** (C3) |

### Check 2 was measuring arm speed with a fixed ruler (rewritten 06.08.2026)

Check 2 used to gate on `near_static = fraction of frames moving < 2 mm`, with a
≤10% threshold, glossed as "near-static mass is the imitation loss learning to
sit still". **That term was an artefact and has been replaced.**

`integration_dt` is very close to a *pure rescaling* of the delta distribution.
Over a 0.005 / 0.007 / 0.009 / 0.012 sweep (12 perturbed episodes each, seeded
so the scenes are identical):

| | 0.005 | 0.007 | 0.009 | 0.012 |
|---|---|---|---|---|
| median (mm) | 1.45 | 1.87 | 2.37 | 3.10 |
| **fraction < 2 mm** | **0.822** | **0.550** | **0.325** | **0.184** |
| fraction < 0.25 × median | 0.034 | 0.056 | 0.070 | 0.089 |
| p99 : median | 1.97 | 2.13 | 2.15 | 2.19 |
| max : median | 2.4 | 2.5 | 2.5 | 2.4 |

Every ratio is flat; only the absolute scale moves. And pi0.5
quantile-normalizes actions by q01/q99, so that scale is divided straight back
out — check 4 reads 92 / 87 / 88% of span across the same sweep. The old term
swung 82% → 18% purely because 2 mm is a fixed ruler against a stretching
distribution.

The distribution is also **unimodal and tight** (at dt=0.010: q10 = 0.53×,
q25 = 0.79×, q75 = 1.33×, q99 = 2.17× the median), so there is no "do nothing"
mode for a policy to alias onto — which is what would actually have justified
the original gloss. The genuinely parked mass is the **gripper close**: 3–7% of
frames sit below 0.05 × median, matching `measure_scripted_arm dwell`'s
independent attribution (`parked` 0.06–0.081). That number is now **reported but
not gated**, since its remedy is `gripper_action_steps`, not skip or dt.

### The ratio terms are measured in JOINT space, the median in EEF

A second measurement-space error, same family as the ruler. The ratios ask
whether the action distribution has a tail that eats the normalized output
range — and openpi's q01/q99 are **per action dimension**, which here are joint
deltas. EEF millimetres are that same motion through a configuration-dependent
Jacobian, so ill-conditioned poses inflate a typical joint step into a large
Cartesian one. Measured on the same 29-episode batch:

| | EEF | joint |
|---|---|---|
| p99 : median | 2.52 | **2.16** |
| max : median | 4.1 | **2.76** |
| below 0.25 × median | 0.077 | **0.060** |

The EEF tail is real but it is a *kinematics* fact, not an action-distribution
fact. It stays **reported** (`eef_max_mm`, `eef_max_over_median`) because a
10 mm step in 20 ms is ~50 cm/s of instantaneous tool speed and that matters for
hardware. Where Cartesian overshoot actually costs a grasp is already gated in
EEF by **check 3**, against `pinch_tol`.

Consequences for reading this check:
* **The EEF median is a timescale calibration, not a learning gate.** It
  answers "does the arm we collected move at the intended physical speed".
  Route a median failure to `integration_dt`, never to skip. Note the band
  `[2.4, 3.0] mm` only encodes "12–15 cm/s" *at a 20 ms decision interval* — at
  a different net rate it must be rescaled, or read `avg EEF speed` instead.
* **The joint-space ratios are the learning-relevant terms** — they are what
  survives normalization. A fat tail there really does spend the normalized
  range on sprint steps the descent never uses; that is the `16_06` pathology.
* `16_06`'s "40% near-static" was largely the ruler artefact. But it does have a
  genuine do-nothing mode, which only the scale-invariant metric shows:
  **30% below 0.25 × median, against 6% in the new data.** The old absolute term
  ranked the new data (82%) as *worse* than `16_06` (40%) — an inverted
  ordering the rewrite corrects.

### `integration_dt` and `SpeedPerturbation.factor_range` are one constraint, not two

The speed perturbation multiplies `integration_dt` per episode, and
`perturb_ik_config` adds a further ±10%, so the *effective* dt an episode runs
at is `base_dt × s × (1 ± 0.1)`. The usable window is narrow (below), so the two
have to be chosen together:

| base dt | `factor_range` | effective dt |
|---|---|---|
| 0.005 | (0.7, 1.4) | 0.0032 – 0.0077 |
| 0.010 | (0.7, 1.4) | 0.0063 – **0.0154** ← past the ceiling |
| 0.010 | (0.85, 1.15) | 0.0077 – 0.0127 |
| **0.009** | **(0.85, 1.15)** | **0.0069 – 0.0114** |

At `dt=0.005` the whole perturbed range sat inside the window, so this coupling
never showed. Raising the base dt without narrowing `factor_range` pushed
episodes to an effective 0.012–0.015, and a 30-episode batch then failed check 3
on descent `max` (7.17 mm against the ±7 mm `pinch_tol`) — a physical limit, not
a tunable threshold.

### Calibrate `integration_dt` against the arm you COLLECT

C1 set `integration_dt=0.005` from `measure_scripted_arm`, which runs **nominal
dynamics**. Collection always runs with arm-dynamics DR on, and that
randomization is biased toward a *slower* arm — `_ARM_FORCE_SCALE` is
reduce-only `(0.6, 1.0)`, `frictionloss` is additive from 0, `damping` adds
0–1.5 on top of a 0.7–1.5 scale, `armature` reaches 1.8×, all compounding.

Measured: the DR'd arm at `dt=0.005` runs at **7.5 cm/s**, half the 12–15 cm/s
C1 was chosen to deliver. Stage 0's harness cannot see this, because it measures
a configuration that never appears in the dataset.

`integration_dt=0.009` puts the collected arm at **13.1 cm/s** and drops IK
aborts to ~3%. `0.012` is the hard ceiling — it breaks check 3 (descent median
3.73 against a 3.5 limit). `0.010` looked fine on a 12-episode ik_noise-only
batch but failed check 3 on a 30-episode `collect_mixed` batch (descent max
7.36 mm against the ±7 mm `pinch_tol`), because the speed perturbation stacks
on top; 0.009 leaves the margin that absorbs it.

### Measured at the calibrated defaults (07.08.2026, `dt=0.009`, `record_every=5 × --skip 2`)

29 episodes / 9268 frames, collected via `collect_mixed.sh 30`. **All six checks
pass; `check_dataset_health` exits 0.**

| check | measured | verdict |
|---|---|---|
| 2 delta distribution | EEF median 2.41 mm; joint below-0.25× 0.060, p99:median 2.16, max:median 2.76 | PASS |
| 3 descent resolution | median 3.14 mm, max 6.73 | PASS |
| 4 normalized output range | **descent 88.2%**, all-frames 92.8 | PASS |
| 5 grasp window | min 3, median 4, max 6; 0 episodes without a close | PASS |
| 6 binarization | 2 levels `{-0.014, 0}`, 0 leading-closed, state 279 values | PASS |
| 7 no recovery | 0 extra cycles | PASS |

Stage 1 alongside: 29/30 collected, 1 IK abort (3.3%), 0 `Grasp not locked`,
0 empty episodes. Reported diagnostics: avg EEF speed 13.1 cm/s, parked-frame
fraction 5.2%, `eef_max_mm` 9.92.

Check 4 going 5.3% → 86.7% is the timescale change delivering what it was for.
Checks 6 and 7 confirm C6 and C3 at the dataset level. Reported alongside:
parked-frame fraction 5.7%, avg EEF speed 13.5 cm/s.

**Check 4 is the headline gate.** It is the entire reason for the timescale change: if the descent still commands a small fraction of the normalized output span, the re-collect did not buy what it was for and the training should not start.

Checks 2 and 3 pull in opposite directions — a coarser net rate raises the median toward the target but coarsens the descent. If they cannot be satisfied together, that is a real finding about the velocity profile, not a threshold to tune away. Measured at `record_every=5 × skip=2` over 29-episode batches: `integration_dt=0.009` gives median 2.41 mm with descent 3.14 / 6.73 mm; 0.010 gives median 2.65 but descent 3.60 / 7.36, over the limit. The usable window is `dt ∈ [0.009, 0.010)` once `factor_range` is applied on top.

Note the two are reached by *different* levers. `--skip` and `integration_dt` both scale the median, but only `integration_dt` changes the physical arm; skip changes only how the same motion is sampled. Prefer `integration_dt` when the median is off and the arm's measured speed is off with it, and skip when the arm is right but the sampling is not.

### Where the gripper-close dead time lands in the data

Stage 0's `dwell` measures this on the raw sim stream; this is how to read it in
the **built dataset**, which is what actually reaches training. There is no
dedicated check for it — it shows up split across two existing ones, so neither
on its own makes it obvious:

* **Check 2's `parked_below_0.05x_median`** is where these frames land. It is
  **reported, not gated** — deliberately, because its remedy is
  `gripper_action_steps` and nothing else. Adjusting skip cannot remove them:
  the parking is long in *time*, so decimation keeps it proportionally. Raising
  `integration_dt` makes it *relatively worse* (measured 2.8% → 5.0% → 5.7%
  across dt 0.005 / 0.009 / 0.010), because `gripper_action_steps` is a fixed
  mj-step count while the arm around it speeds up. Cross-check the number
  against `dwell` rather than against a threshold.
* **Check 5's grasp-window count** is the same frames from the other side. A
  window at the high end of the expected 5–8 is partly this: ~86 of the 100
  close steps are near-identical parked images, all labelled "closed" once
  binarized. Duplicated-looking signal rather than a bug — but worth knowing
  before concluding the grasp phase is well represented.

Record both on the 10-episode batch **even when they pass**, so the full collect
has a baseline to compare against.

---

## Stage 4 — full collect

Repeat Stage 1's log greps at scale, then re-run Stage 3 on the **full** dataset:

```bash
python -m aera.autonomous.openpi.scripts.check_dataset_health --repo-id Purple69/<full>
```

Ten episodes will not surface a rare IK abort or a block-preset-dependent problem, and this is the last cheap check before the expensive step. Check 4 needs the finalized `meta/stats.json`, so it can only be read once the dataset is built.

---

## Stage 5 — eval config, before reading any eval number

Not a data check, but the same class of silent error: an eval whose rate or scene distribution does not match the data reports a number about the wrong thing.

- `n_substeps == skip` (10). A mismatch here silently rescales every commanded motion.
- Eval env config == collection env config (F2/E2): cameras + geometric lookat + object yaw + prompt string.
- `replan_steps=4` is a decision, not an inheritance — confirm with a 1–5 sweep on the first checkpoint (`eval_variance.py --replan-steps`).
- The old `eval/dr/*` and `eval/nodr/*` mlflow series stop at this run (E1); `eval/...` is now the in-distribution number.

---

## Only then: train.
