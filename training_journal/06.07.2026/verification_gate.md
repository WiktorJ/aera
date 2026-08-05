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
    timing --dt 0.005 --max-steps 3000 --seeds 0 1 2

# close depth: does the scripted close actually reach the block?
python semi_autonomous/aera_semi_autonomous/scripts/measure_scripted_arm.py \
    close-sweep --dt 0.005 --max-steps 3000 --seeds 0

# camera parameterization: does eval draw from collection's distribution?  (needs F2)
python semi_autonomous/aera_semi_autonomous/scripts/check_camera_parameterization.py

# dead time: how much of the episode is the arm parked with nothing happening?
python semi_autonomous/aera_semi_autonomous/scripts/measure_scripted_arm.py \
    dwell --dt 0.005 --max-steps 3000 --seeds 0 1 2
```

| expect | if not |
|---|---|
| `sim_time_s` 4.8–6.6, `avg_speed_cm_s` 12–15, `raw_frames` 2400–3300 | C1 — retune `integration_dt` / `max_steps` |
| **`lock_engaged=True`** on every seed | C4/C5/F1 — the close never reaches the block. **This is the failure that produced the last run.** |
| `close-sweep`: `pad_contacts > 0`, `pinching=True`, `lock=True` at the scripted target | F1 — the depth model still disagrees with the pad geometry |
| collection and eval camera rows agree under DR offsets | F2/E2 — eval is sampling an OOD camera distribution |
| `dwell`: `parked` ≤0.08, `parked_gripper_idle` ≤0.04 | see the dead-time note below — remedy is `gripper_action_steps` |

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

**If it needs trimming, the lever is `gripper_action_steps` (or the `* 2` budget
multiplier), NOT `gripper_pos_tolerance`.** Loosening the tolerance would
restore the early exit that C5 removed on purpose, and the jaws would stop short
of the block again — the exact failure that produced the last run. Dropping the
multiplier from `2 ×` to ~`1.3 ×` cuts the dead half while still delivering the
full ramp. Do not tune it before Stage 3 says it matters.

Baseline for comparison, measured 02.08.2026 on unfixed code: `dt=0.15` → 0.69–0.75 s, 103–107 cm/s, **`lock_engaged=False`**; `dt=0.005` → 4.81–6.61 s, 12.1–14.9 cm/s, **`lock_engaged=False`**.

---

## Stage 1 — collect 10 episodes

```bash
./semi_autonomous/aera_semi_autonomous/scripts/collect_mixed.sh 10 data/verify_batch 42 2>&1 | tee /tmp/collect_verify.log
```

Then read the log — these are warnings, not errors, so they scroll past silently:

```bash
grep -c "Grasp not locked"                     /tmp/collect_verify.log   # want 0
grep -cE "Max steps .* reached|could not move" /tmp/collect_verify.log   # want 0
grep    "Successfully collected"               /tmp/collect_verify.log   # want 10/10
```

| check | expect | failure routes to |
|---|---|---|
| 8 — lock engaged | ≥95% of episodes, i.e. 0 `Grasp not locked` | **re-collect** (C4/C5/F1) |
| 1 — IK budget | 0 `Max steps` / `could not move`, 10/10 collected | **re-collect** (C1 `max_steps`) |

Also record **wall-clock per episode** and **disk per episode** (`du -sh data/verify_batch`), and sanity-check the recorded frame count: at `record_every=5` an episode should hold **~500** frames, not ~2500. If it holds ~2500 the decimation is not in effect and collection will run ~5× slower than it needs to.

Expect ~8 s/episode and ~1000 jpeg sidecars per episode. That is the point of `record_every=5` — see the X1 note in [implementation_plan.md](./implementation_plan.md). If it comes in much higher, the profile to check first is the two camera renders in `_record_step`, which measured 13.7 ms against 0.09 ms of physics.

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
    --output-repo-suffix skip10_delta_verify
# -> Purple69/aera_semi_pnp_dr_<DDMMYYYY>_verify_skip10_delta_verify
```

Three notes:
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
| 2 delta distribution | median 2.4–3.0 mm, p99 ≤5, near-static ≤10%, max:median ≤2 | 3.28 / 25.6 / 40% / 9.2 | **transform** — adjust skip |
| 3 descent resolution | median ≤3.5 mm, max ≤7 mm | 0.78 / 11.4 | **transform** — drop to skip 5–6 (deploy `n_substeps` follows) |
| 4 **normalized output range** | descent ≥50% of span | **5.3%** | **transform** (skip) or **re-collect** (dt), depending which term is off |
| 5 grasp window | ≥3 frames/episode spanning open→closed, expect 5–8 | 4–5 | **transform** — duplicate / weight grasp-window frames |
| 6 binarization | 2 action values, 0 leading-closed frames, state continuous | 251 levels, 8–57 leading | **transform** (flags) or **re-collect** (C6 jaws-open) |
| 7 no recovery | 1 grasp cycle per episode | 3/3 with extra cycles | **re-collect** (C3) |

**Check 4 is the headline gate.** It is the entire reason for the timescale change: if the descent still commands a small fraction of the normalized output span, the re-collect did not buy what it was for and the training should not start.

Checks 2 and 3 pull in opposite directions — coarser skip raises the median toward the target but coarsens the descent. If they cannot be satisfied together, that is a real finding about the velocity profile, not a threshold to tune away.

### Where the gripper-close dead time lands in the data

Stage 0's `dwell` measures this on the raw sim stream; this is how to read it in
the **built dataset**, which is what actually reaches training. There is no
dedicated check for it — it shows up split across two existing ones, so neither
on its own makes it obvious:

* **Check 2's near-static fraction** is where the parked frames land (threshold
  ≤10%). **If check 2 fails on near-static specifically, get the attribution
  before touching skip** — `dwell` says whether it is gripper parking (remedy:
  `gripper_action_steps`) or genuine arm creep (remedy: skip). Adjusting skip
  cannot remove parked frames: the parking is long in *time*, so decimation
  keeps it proportionally. This is the same trap as the fast arm's 40%
  near-static, which was also gripper ramps rather than timescale.
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
