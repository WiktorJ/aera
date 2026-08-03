# Implementation plan for the next run (derived from [next_run_changes.md](./next_run_changes.md))

Enumerates every code change required to reach the target state: **slow arm, skip=10, gripper full-close + binarized, no recovery, in-distribution-only eval**, plus the tooling needed to run the pre-training verification gate.

Item IDs: **F** = shared foundations, **C** = collection, **T** = transform, **E** = eval, **V** = verification tooling, **X** = decisions that need a call before coding. Each item lists the anchor files, the change, and what "done" means.

Ordering: F → C → (collect 10 eps) → V → T → E. F1/F2 first because several C/E items are just consumers of them.

---

## X. Decisions — RESOLVED 02.08.2026

| # | Question | Decision |
|---|---|---|
| X1 | Recording cost at dt=0.005 | **Drop depth recording; accept the 8× raw-frame growth otherwise.** `record_every` decimation stays a deferred option, re-considered only if the 10-episode batch shows unacceptable wall-clock/disk. |
| X2 | Where the slow-arm IK values live | **Change the `IKConfig` defaults** — the slow arm is the definition of the demonstrated behaviour, so every consumer (demo, sweeps, DAgger) gets it. |
| X3 | Seed budget after dropping the noDR arm | **40 episodes** — `n_dr_seeds=20`, `k_repeats=2`. Partial reallocation of the freed noDR budget: 5 more scenarios than today's DR arm, still 20% cheaper than the old 50-episode suite. |
| X4 | `replan_steps` / `max_episode_steps` | **`replan_steps=4`.** `max_episode_steps` still derived from the measured demo length (below). The 1–5 sweep drops from blocking to confirmatory. |
| C1 | IK step budget | **`max_steps=3000`** — measured sufficient, see below. |
| C6 | Jaws-closed-at-start / eval warmup gap | **Close it** (env-level jaws-open + the paired warmup change). |
| E4 | `close_depth_tol` semantics flip | **Rename** the field so a stale config fails loudly. |
| V1 | Measurement harness | **Pulled into the repo** — see V1 below; all figures re-verified locally. |

### Measured while landing V1 (this repo, `measure_scripted_arm`)
| dt | max_steps | sim time | avg EEF speed | raw frames | EEF path | lock engaged |
|---|---|---|---|---|---|---|
| 0.15 | 700 | 0.69–0.75 s | 103–107 cm/s | 347–375 | 0.72–0.80 m | **False** |
| 0.005 | 3000 | 4.81–6.61 s | 12.1–14.9 cm/s | 2407–3306 | 0.72–0.80 m | **False** |

⇒ dt=0.005 lands inside the 5–7 s / 12–15 cm/s target, **`max_steps=3000` is enough** (3/3 seeds, no aborts), and path length is unchanged, so only the timing shifts. `lock_engaged=False` in **both** rows independently confirms the doc's central claim: with the current preload target the kinematic lock never engages, regardless of arm speed. The close sweep reproduces the doc's table exactly (cmd −11.5 mm → jaw −11.745, 0 pad contacts, no lock; −11.0 → −11.429, 8 contacts, lock; 0.0 → −10.466, 8 contacts, lock).

---

## X-archive. Rationale behind the resolved decisions

### X1. Recording cost at dt=0.005 — the plan doc does not cost this, and it is the biggest hidden change
`_record_step` (`ar4_mk3_robot_interface.py:223`) runs **once per mj-step** and renders **both RGB cameras + both depth cameras** every time. Slowing the arm 8× multiplies raw frames 347–375 → 2434–3306 per episode, so it also multiplies:
* **render time** ~8× (4 renders/frame) — the dominant cost of collection wall-clock;
* **RAM per episode**: JPEG bytes + **raw float32 depth arrays** are held in memory for the whole episode and flushed at `stop_episode` (`trajectory_data_collector.py:394-404`). Depth alone is 224×224×4 B × 2 cams ≈ 400 kB/frame → ~0.15 GB/ep today, **~1.2 GB/ep** at 3000 frames;
* **disk**: 2 jpg + 2 depth png per raw frame, ~8× the `16_06` footprint, ×2855 episodes.

Options (not mutually exclusive):
* **(a) Drop depth recording** (recommended, near-free): depth is already unused downstream — `convert_data_to_lerobot.py:213-215` has the depth branch commented out. Add `record_depth: bool = False` to the interface config and gate the depth block in `_record_step`. Halves render calls, removes the RAM spike and half the disk.
* **(b) Record every Nth mj-step** (recommended if (a) alone is not enough): a `record_every: int = 1` counter in `_record_step`. Keeps raw frames/ep at today's order while the sim runs slow. **Invariant becomes `n_substeps_deploy = record_every × skip`**, and the transform's `--skip` must be divided accordingly (e.g. `record_every=2, skip=5` ≡ today's plan of `skip=10` at 2 ms). Needs the invariant documented in `CONTROL_RATE_SPEC.md` and asserted in the transform log line.
* **(c) Accept the 8×.** Only viable if collection wall-clock and ~TB-scale disk are actually fine.

**Decided: (a) + (c).** Depth goes off now; the raw stream stays at native 2 ms so the verification measurements are unambiguous. (b) is held in reserve for the full collect if the 10-episode batch shows the wall-clock or disk is untenable.

### X2. Where the slow-arm IK values live
`IKConfig.integration_dt` / `max_steps` (`ar4_mk3_interface_config.py:16,19`) are shared by every consumer of the scripted expert (collection, `demo_pick_and_place`, `sweep_ik_params`, future DAgger). **Decided: change the dataclass defaults** — the slow arm is now the definition of the demonstrated behaviour, and a collection-only override would silently leave `demo_pick_and_place` (the thing used to eyeball demos) on the old timescale.

### X3. Seed budget after dropping the noDR arm
Doc item 10 says "move the whole seed budget to DR seeds". **Decided: 40 episodes — `n_dr_seeds=20`, `k_repeats=2`.** Five more scenarios than today's DR arm (tightening the between-seed estimate, which ran ~0.30–0.45 std), at 20% less total cost than the old 50-episode suite. `k_repeats=2` is kept so the within-seed / policy-variance decomposition survives; that is what 50×1 would have traded away.

### X4. `replan_steps` / `max_episode_steps` are explicitly deferred to a sweep
Doc item 12 forbids inheriting the old values. **Decided: `replan_steps=4`** (at `n_substeps=10` that is inference every 80 ms, the rate the doc argues for). `max_episode_steps` still comes from the measured demo length — at `n_substeps=10` a demo is ~250–330 env steps, so **1000** (≈3× demo length) is the value to land. The 1–5 sweep on the first checkpoint becomes confirmatory rather than blocking; `eval_variance.py` already exposes both flags.

---

## F. Shared foundations (do these first)

### F1. ✅ LANDED (`2a6c621`) — one jaw-geometry module, the single source of truth for close depth
**Problem** (doc items 4 + 13): three places independently model "where do the pads touch", and all three are wrong by the same +0.4 mm:
* `get_object_grasp_gripper_pos` — `pick_and_place_helpers.py:233` (`target = -(pinch_half_width - preload)`)
* the lock's close-depth gate — `kinematic_grasp.py:293-298` (`close_ctrl_target >= -(half_width + close_depth_tol)`)
* `metrics._classify_miss`'s `close_shallow` — `metrics.py:504-507` (same expression)

**Change**: new module (e.g. `aera/autonomous/envs/jaw_geometry.py`) exporting:
```python
pad_inner_offset(model) -> float          # |geom_pos_x| - geom_size_x per jaw pad
first_contact_qpos(model, half_width)     # = pad_inner_offset - half_width
engage_qpos(model, half_width, squeeze)   # = first_contact_qpos + squeeze
```
Derive `pad_inner_offset` **from the model**, not a constant: `gripper_jaw{1,2}_contact` are `size="0.001 …" pos="-0.0014 …"` (`ar4_mk3.xml:322,336`) ⇒ 0.0014 − 0.001 = **0.0004 m**, which reproduces the measured `gap(mm) = 0.8 − 2·q` exactly. A geometry change in the XML then cannot silently desync the three call sites again.
* Default `squeeze` = 0.0006 m — the measured margin that makes a close actually engage (cmd ≥ −11.0 mm for a 24 mm block; first contact −11.6 mm).
* Unit test asserting `first_contact_qpos` for the 19/22/24/27 mm presets, and that the 30 mm preset lands outside `[-0.014, 0]` (i.e. is unpinchable by construction — distractor-only, consistent with `_GRASPABLE_BLOCK_PRESETS`, `domain_rand_config_generator.py:36-39`).

**Done when**: all three call sites import from this module and no expression of the form `-(half_width ± tol)` remains.

### F2. ✅ LANDED (`95099fb`) — shared env-config factory consumed by collection and eval
**Problem** (doc item 11, "Rule going forward: eval env config == collection env config"): `collect_trajectories.py:213-224` and `run_policy_on_env._build_env` (`run_policy_on_env.py:170-184`) build `Ar4Mk3EnvConfig` independently. Four mismatches were found by hand (cameras, geometric lookat, object yaw, prompt); nothing prevents a fifth.

**Change**: new factory (e.g. `aera/autonomous/envs/task_env_factory.py`):
```python
def build_task_env_config(model_path, domain_rand, *, eval_overrides=None) -> Ar4Mk3EnvConfig
```
holding the *collection* values as the single definition: `reward_type="sparse"`, `use_eef_control=False`, `translation=T`/`quaterion=Q` (→ `T_GEOMETRIC`/`Q_GEOMETRIC` via `Ar4Mk3EnvConfig.__post_init__`, `ar4_mk3_config.py:366-374`), `distance_multiplier=1.2`, `z_offset=0.3`, `use_geometric_lookat=True`, `randomize_object_yaw=True`. Eval layers only its own knobs on top: `n_substeps`, `kinematic_grasp`, `relative_action_scale`, `include_images_in_obs`, `absolute_state_actions`, `obs_image_aug`.
* Also centralize the **prompt template** here or next to it (`f"pick the {obj} block and place it on the {tgt} target"`, lowercase, no period) so eval can't drift again (`run_policy_on_env.py:160-163`).
* Also centralize the **DR generator call** — `generate_random_domain_rand_config(randomize_cameras=True, randomize_arm_dynamics=True)` — because eval currently calls it with defaults (`run_policy_on_env.py:148-150`), giving `randomize_cameras=False`.

**Done when**: both `collect_trajectories.main` and `_build_env` call the factory, and a test asserts the two produced configs differ only in the documented eval-only fields.

---

## C. Collection

### C1. ✅ LANDED (`59f4577`) — slow the scripted arm (doc item 1)
* `ar4_mk3_interface_config.py:16` — `integration_dt: 0.15 → 0.005` (default change, per X2)
* `ar4_mk3_interface_config.py:19` — `max_steps: 700 → 3000` (measured sufficient: 3/3 seeds complete, 4.81–6.61 s, 12.1–14.9 cm/s)

**Budget note (not in the doc, from the perturbation code)**: the effective dt is perturbed twice and only one of the two scales the step budget.
* `apply_speed_perturbation` (`trajectory_perturbation.py:530-535`) scales `integration_dt × s`, `max_steps / s`, `s ∈ [0.7, 1.4]` — self-correcting.
* `perturb_ik_config` (`trajectory_perturbation.py:451-472`) scales `integration_dt` by ±10% but passes `max_steps=base.max_steps` through **unscaled**.
⇒ worst case is `s=0.7` × `−10%` ⇒ dt ≈ 0.00315 with only `max_steps/0.7` of budget. **3000 confirmed** on the unperturbed path; the perturbed worst case is still only checked by the 10-episode batch.

**Done when**: ✅ headless episodes report 4.8–6.6 s / 12–15 cm/s (verified), and a 10-episode *perturbed* collection log has zero `Max steps ... reached` / `could not move above target`.

### C2. ✅ DECIDED — no code change (leave `gripper_action_steps=50` for run 1)
`gripper_action_steps=50` (`ar4_mk3_interface_config.py:64`) is what produces the doc's "grasp window dilutes to ~3% of the episode". Note it is *already* speed-perturbed (`trajectory_perturbation.py:543-545`, `/s` ⇒ 36–71 mj-steps). Leave as-is for run 1 — verification check V-5 measures whether the resulting grasp window is degenerate, and the doc's chosen remedy is a transform-time fix (T2), not a collection change.

### C3. ✅ LANDED (`dc06e77`) — drop recovery injection (doc item 2)
* `collect_mixed.sh` — delete segment `[2/3]` (`ik_noise + recovery`) and re-weight to 90% `ik_noise` / 10% `offset_approach` (update the header comment and the echo block).
* Keep `inject_partial_grasp` / `inject_wrong_approach` (`pick_and_place_helpers.py`) and `perturb_recovery` (default already `False`, `trajectory_perturbation.py:308`) — they are the DAgger substrate for the round after this one.
* Optional guard: log a loud warning in `collect_trajectories.main` when `perturb_recovery` is on, so a stale script can't silently reintroduce it.

**Done when**: `grep -c perturb-recovery collect_mixed.sh` returns only the `--perturbation.no-perturb-recovery` occurrences.

### C4. ✅ LANDED (`5fd58fd`) — full-close command at collection (doc item 3 — load-bearing)
`collect_trajectories.py:115` computes `grasp_gripper_pos` from the (broken) formula and passes it to `robot.grasp_at` at line 132; line 128 recomputes it after `inject_partial_grasp`.
* Command `0.0` for the real grasp. Implementation: a `GRIPPER_FULL_CLOSE = 0.0` constant + a `CollectConfig.full_close_grasp: bool = True` flag so the old behaviour is still reachable for A/B, rather than deleting the code path.
* Mirror the same change in `demo_pick_and_place.py` so the demo tool shows what collection now does.
* The recovery path (`inject_partial_grasp`, `grasp_and_slip`) keeps using the *computed* target — a marginal grip is its whole point — which is why F1 still matters even with C4 landed.

**Done when**: 3/3 seeds log `Grasp locked: object0` (`ar4_mk3_robot_interface.py:164`) and never `Grasp not locked`.

### C5. ✅ LANDED (`5fd58fd`) — fix the depth model + convergence tolerance (doc item 4)
* `get_object_grasp_gripper_pos` (`pick_and_place_helpers.py:212-245`) — replace `target = -(pinch_half_width - preload)` with `jaw_geometry.engage_qpos(model, pinch_half_width, squeeze=preload)` from F1. Update `DEFAULT_GRASP_PRELOAD`'s docstring: it is now a **squeeze past first contact**, not a fudge against half-width.
* `gripper_pos_tolerance` (`ar4_mk3_interface_config.py:67`) — `1e-3 → 1e-4` (below the preload it is supposed to deliver). Two call sites in `_interpolate_gripper` (`ar4_mk3_robot_interface.py:352-358` early exit, `:368` final-error log).
* **Do not "fix" the non-convergence TODO at `ar4_mk3_robot_interface.py:373`** — with a full-close target and a block in the jaws the loop is *supposed* to run its full `2 × gripper_action_steps` budget. That is what drives the jaws into the block. Add a comment saying so.
* Side effect to expect: with C4 + tighter tolerance, every close now runs the full ramp (≈100 mj-steps) instead of exiting early — this is the grasp-window frame count V-5 measures.

### C6. ✅ LANDED (`5105bf1`) — jaws start closed, kill the spurious leading "closed" frames (doc verify item 6)
`m.qpos0` for both jaw joints is `[0, 0]` and `_reset_sim` restores `self.initial_qpos` verbatim (`ar4_mk3_base.py:1150`), so every episode opens with a 0 → −14 mm ramp that binarizes to "commanded closed" while nothing is held (measured: 8 leading frames normal, 57 in recovery episodes).
* **Decided: the env-level fix** — open the jaws in the initial state — set `gripper_jaw{1,2}_joint` to `-0.014` in `_env_setup` / the captured `initial_qpos`, and seed `data.ctrl` for `act8/act9` to `-0.014` so the first step doesn't command them shut again.
* **This is shared with eval**, which is the point — but it forces a paired change: `run_policy_on_env._build_warmup_action` (`run_policy_on_env.py:236-241`, `GRIPPER_CLOSED_ACTION = -1.0` at `:49`) currently holds the gripper **closed** through the settle window. Change the warmup to command **open**, or the eval env immediately re-closes jaws the training data now shows open. Land both together.
* The transform workaround (drop leading frames before the first open, T1 sub-flag) is **not** the primary fix — it would leave eval and collection inconsistent at t=0, which is exactly the class of mismatch F2/E2 exist to eliminate. Keep the flag anyway for re-processing datasets collected before this lands.

### C7. ✅ LANDED (`59f4577`) — depth recording off (per X1)
* `record_depth: bool = False` in `Ar4Mk3InterfaceConfig`, gating `ar4_mk3_robot_interface.py:241-245`. Halves the per-frame render calls and removes the ~1 GB/episode RAM spike; downstream is unaffected (`convert_data_to_lerobot.py:213-215` already ignores depth).
* `record_every` decimation is **not** implemented now. If the 10-episode batch shows the wall-clock or disk is untenable, add `record_every: int = 1` to the same config plus a counter in `_record_step`, record it in the episode metadata, and propagate the `n_substeps = record_every × skip` invariant to the transform log and `CONTROL_RATE_SPEC.md`.

---

## T. Transform (`transform_skip_dataset.py`)

### T1. ✅ LANDED (`39725ea`) — `--binarize-gripper` (doc item 8 — load-bearing)
New flag mapping the **action** gripper dims to `{-0.014 open, 0 closed}` with `--gripper-binarize-threshold` (default `-0.013`: `> threshold → 0`, else `-0.014`). Constants should come from one place shared with `_denormalize_gripper` (`run_policy_on_env.py:271-276`), which already hardcodes `-0.014`.
* **State gripper channel untouched** (doc: it is a useful proprioceptive cue, and width-invariance is only wanted on the command side).
* **Ordering inside `transform_dataset`**: binarize `action_future` immediately after it is read (`transform_skip_dataset.py:490-494`), i.e. before `last_action_for_episode` is updated. ~~Then the `--gripper-eps` guard sees the 0.014 jump at transitions and nothing in between — strictly better behaved than today.~~ **That was wrong** (corrected 03.08.2026): with the static filter active, binarizing before it *deletes* the whole close ramp, whereas binarizing after it merely decimates. The ordering is now moot because T3 both drops the filter for this run and moves its guard onto the state channel — but the reasoning is recorded so it isn't re-derived incorrectly.
* Delta conversion (`:518-525`) only touches `[:num_joint_dims]`, so the binarized gripper passes through absolute. No change needed, but assert `num_joint_dims == 6`.
* Optional sub-flag `--drop-leading-closed` (fallback for C6): drop each episode's leading frames up to the first "open" action.
* Unit tests in `test_transform_skip_dataset.py`: threshold behaviour, exactly two output values, state untouched, ordering vs. the static filter.

### T3. **Drop `--min-action-delta` — it deletes the grasp window, and binarization makes that total.** (found 03.08.2026 while calibrating V2)
Not in the plan doc; found by running the new checker on `16_06`. Doc item 9's claim that "the gripper guard deliberately preserves exactly those frames" is **wrong**, and the error gets much worse under T1.

The mechanism:
1. `_interpolate_gripper` parks the arm for the whole close ramp (`data.ctrl[:6] = data.qpos[...]`, `ar4_mk3_robot_interface.py:323`) ⇒ every ramp frame has joint-Δ ≈ 0, i.e. below `min_action_delta` by construction.
2. The static filter (`transform_skip_dataset.py:504-516`) then keeps such a frame only if the gripper moved more than `--gripper-eps` (**1 mm**) since the last *written* frame.
3. Per-frame jaw travel during a close is **~0.15 mm at skip3** and ~0.35 mm at skip10 — far below that 1 mm guard. So the guard fires only once the drift accumulates, roughly every 7th frame, and the ramp is **decimated rather than preserved**.

Measured on `16_06` (episode 0, hand-traced): the close ramp survives as jaw qpos −13.68 → −11.76 mm across frames 51–55, i.e. **4–5 dataset frames** where the 50 mj-step ramp should give ~17 at skip3.

**Under `--binarize-gripper` this becomes total deletion.** The binarized action is *constant* across the ramp (already at `0`), so `gripper_moved` is False for every ramp frame except the single transition — and the arm is parked — so all of them are dropped and the grasp window collapses to ~1 frame. That fails check 5's floor of 3 and destroys precisely the signal T1 exists to sharpen.

**Two fixes, both agreed:**

**(a) Drop `--min-action-delta` for the new dataset** (transform invocation, T2). The doc's own measurement says it is nearly inert on the slow arm (joint-Δ median 0.011 rad ≈ 20× the 0.0005 threshold), so its only material effect is deleting parked gripper-ramp frames. Not worth keeping a knob whose sole live behaviour is harmful.

**(b) ✅ LANDED — the guard now reads the jaw STATE, not the action** (`transform_skip_dataset._gripper_moved`). The filter's question is "did anything happen in the world"; the action is only a *label* for it, and binarization deliberately makes that label constant across the ramp. Judging staticness on the label was the actual defect — dropping the filter for one run would have left the trap armed for the next person who switches it back on. Also lowers `--gripper-eps` `0.001 → 0.0002`, since the old default sat 3–7× above the per-frame ramp travel it was supposed to protect. Unit tests in `test_transform_skip_dataset.py` cover mid-ramp (binarized action constant, state moving ⇒ keep), parked jaw (⇒ drop), and the no-state fallback.

Note the sizing is genuinely tight: a close ramp travels ~0.15 mm/frame at skip3 and ~0.35 mm at skip10, while an open jaw jitters ~0.1–0.2 mm/frame against its limit stop — signal and noise floor nearly overlap. Another reason (a) is the right call for this run, with (b) as the correct behaviour whenever the filter *is* used.

### T2. ✅ PINNED in the runbook (no code change)
`--skip 10`, `--binarize-gripper`, **no `--min-action-delta`** (see T3), no `--image-aug` / `--state-aug` (doc item 6). Grasp-window duplication/weighting is **contingent on check 5 failing** — do not build it up front.

---

## E. Eval

### E1. ✅ LANDED (`6dabd29`) — delete the noDR arm from the suite (doc item 10)
`suite.py`: remove `n_seeds` / `seed_start` (`:66,70`), `_nodr_seeds` (`:200`), the `(False, _nodr_seeds(cfg))` leg of `run_suite` (`:183-186`), the `no_domain_rand` group in `summarize` (`:265-266`), the `nodr` prefix in `flatten_for_mlflow` (`:277`), and the noDR branch of `log_summary` (`:307-310`). `EpisodeRecord.domain_rand` (`:88`) and the `dr`/`nodr` video tag (`:149`) become dead — drop them.
* Consumers to update in the same commit: `eval_worker.py:85,88` (`n_seeds`, `seed_start` flags) and its summary log line `:212-217` (`eval/nodr/success_rate`); `eval_variance.py:59,62,111`.
* `n_dr_seeds: 15 → 20` (`suite.py:65`), `k_repeats` stays **2** ⇒ 40 episodes per checkpoint (X3).
* **Backwards-compat note**: `eval/nodr/*` and `eval/dr/*` mlflow series stop here. Since `eval/...` (overall) now *is* the DR number, the headline curves stay continuous; note the break in `NOTES.md`.

### E2. ✅ LANDED (`95099fb`) — make the DR-on path genuinely in-distribution (doc item 11)
Delivered by **F2** — but check each of the four explicitly, since they are the regression tests for the factory:
1. `randomize_cameras=True` **and** `use_geometric_lookat=True` **together** (either alone is worse than neither: `_apply_camera_offset` adds collection-calibrated offsets to a differently-parameterized base — measured, the camera lands on the other side of the scene).
2. `randomize_object_yaw=True`.
3. Prompt string exactly `"pick the {obj} block and place it on the {tgt} target"`.
4. `randomize_arm_dynamics=True` (already matching; pin it with the test so it stays).

### E3. ✅ LANDED (`6dabd29`) — deploy at the faithful rate (doc item 12)
* `SuiteConfig.n_substeps: 3 → 10` (`suite.py:78`) — must equal the dataset skip (× `record_every` per X1).
* `replan_steps: 10 → 4` (`suite.py:77`, X4) — 80 ms between inferences at `n_substeps=10`. Confirm (don't discover) with a 1–5 sweep via `eval_variance.py --replan-steps` on the first checkpoint. Note the coincidence with the old ablation's `replan=4` is not inheritance: that value was found at `n_substeps=20` on skip3 data, i.e. a 4× rate mismatch, so it means something different here.
* `max_episode_steps: 1000` (`suite.py:75`) — keep; ≈3× the demo length (250–330 env steps at `n_substeps=10`, from the measured 2407–3306 raw frames/episode). Re-check against V-1 on the perturbed batch.
* `run_policy_on_env.Args` defaults (`:66,68,85`) drift from `SuiteConfig` (`replan_steps=5`, `max_episode_steps=400`, `n_substeps=20`). Point them at `SuiteConfig` or delete the duplication.
* Update `CONTROL_RATE_SPEC.md` with the new skip/n_substeps pair (and the `record_every` invariant if X1(b) is taken).

### E4. ✅ LANDED (`2a6c621`, folded into F1) — recalibrate the close-depth thresholds (doc item 13)
Consumers of F1:
* `kinematic_grasp.engage` (`:293-298`) — gate becomes `close_ctrl_target >= engage_qpos(model, half_width)`.
* `metrics._classify_miss` (`:504-507`) — `close_shallow` fires below the same threshold. This closes the measured **[−12.5, −11.0) mm dead band** where an attempt physically cannot pinch but is reported as `no_pinch`/`unknown`.
* `GraspEngageConfig.close_depth_tol` (`:85`) changes meaning from *slack below the surface* to *required squeeze past first contact*. **Rename to `close_squeeze`** (decided) so a stale config carrying the old field fails loudly instead of being silently misread, and rewrite the docstring at `:56-60`. Default `0.0006` — the measured margin between first contact (−11.6 mm) and reliable engagement (−11.0 mm) on a 24 mm block.
* Note the gate is **eval-only**: collection passes `close_ctrl_target=None` (`ar4_mk3_robot_interface.py:157` → `engage(max_distance)`), so this cannot regress collection.
* With T1 landed, a binarized policy commands `0` ⇒ the gate always passes ⇒ `close_shallow` should collapse to ~0 by construction. That is the doc's predicted, testable effect — make sure the metric is still *computed* so its collapse is visible rather than absent.

### E5. ✅ LANDED (`95099fb`) — keep `--domain_rand=False` as a dev tool (doc item 14)
No removal. Just confirm `_resolve_prompts`'s non-DR branch (`run_policy_on_env.py:151-153,165`) still returns a sane prompt after E2's template change, and add a comment that this path is for visualization only, never a scored number.

### E6. ✅ LANDED (`5105bf1`) — warmup gripper action (paired with C6)
See C6 — `_build_warmup_action` must open, not close, once the env starts with open jaws.

---

## V. Verification tooling (needed *before* the full collect — the doc's gate)

The doc's 8 checks were unimplemented, and the harness that produced its numbers lived in an ephemeral scratchpad. **All three items are now landed**; the runbook that ties them together is [verification_gate.md](./verification_gate.md).

### V1. ✅ LANDED — the scripted-arm harness is in the repo
* **`semi_autonomous/aera_semi_autonomous/scripts/measure_scripted_arm.py`** — drives the real collection code path (`Ar4Mk3RobotInterface`, same go_home → grasp_at → release_at script) headless with no data collector attached, tracing per mj-step. Subcommands:
  * `timing` — sim time / raw frames / EEF path / average speed, per dt (**check 1**)
  * `deltas` — per-policy-step Δ distribution and descent resolution at `--skips` (**checks 2, 3**)
  * `dynrange` — descent action as % of the normalized output span, in-sim proxy for **check 4**
  * `shove` — block disturbance between `grasp_at` and lock engagement
  * `close-sweep` — commanded close depth → final jaw qpos / pad contacts / pinch / lock (**check 8**, and the acceptance test for F1+C4+C5)
  * `dwell` — near-static frame attribution (parked-during-ramp vs parked-idle)
  * `--json` on every subcommand, so V3's gate can parse rather than scrape.
* **`.../scripts/ros_msg_stubs/`** — vendored minimal `geometry_msgs` / `sensor_msgs` / `std_msgs` / `cv_bridge`, injected onto `sys.path` **only when the real packages are missing**, so inside the ROS container nothing changes. Chosen over making the production imports optional: the stub is contained in the tooling and touches no collection code. `CvBridge` deliberately raises, since the image path is only reachable with a collector attached.
* **`.../scripts/check_camera_parameterization.py`** — prints collection's `(T_GEOMETRIC, Q_GEOMETRIC, geometric=True)` against eval's `(T, Q, False)` with and without DR offsets. This is E2's regression check; it reproduces the doc's measured divergence exactly (offset `[-0.031, 0.308, -0.004]` → collection `azim 209.3°, elev −31.2°, dist 1.11` vs eval `azim 264.2°, elev −7.9°, dist 0.855`).
* **`aera/autonomous/openpi/scripts/measure_action_dynrange.py`** — the authoritative **check 4**: q01/q99 from the dataset's own `meta/stats.json` (the exact quantiles openpi normalizes with), `|a_i|·2/(q99−q01)` on descent frames, worst joint, plus EEF millimetres via FK. Generalized from the one-off script (was hardcoded to the `16_06` repo ids) to `--repo-id` / `--frames-repo-id`.
* **Verified locally**: every subcommand reproduces the journal's figures — 0.69–0.75 s / 103–107 cm/s at dt=0.15, 4.81–6.61 s / 12.1–14.9 cm/s at dt=0.005, the −11.5/−11.0/0.0 close sweep row-for-row, shove 0.79 mm / yaw 0.03° / z-rise 1.46 mm, and 100% of parked frames attributed to gripper ramps.
* **Caveat when reading `deltas`**: the harness traces the *raw* sim stream, while the doc's "Measured Δ vs skip" table came from the built dataset (smoothed, static-filtered, DR on). Expect the harness to read a lower median and a higher near-static fraction on identical settings — compare harness-to-harness, and use V2 for dataset-to-dataset.

### V2. ✅ LANDED — `aera/autonomous/openpi/scripts/check_dataset_health.py`
Takes a LeRobot repo id, runs checks 2–7, prints measured-vs-threshold per check, and **exits 1 on any failure** so it can gate a pipeline directly. `--json` for machine-readable output, `--no-fail-exit` for calibration runs, every threshold overridable (they default to the plan's skip=10 slow-arm targets). Unit tests in `test_check_dataset_health.py` pin the gripper-structure parsing that checks 5–7 rest on.

**Calibrated on `16_06`, where the answers are already known** — the run reproduces every documented pathology:

| check | measured on `16_06` | verdict |
|---|---|---|
| 2 delta distribution | median 3.28 mm, p99 25.56, near-static 40.3%, max:median 9.2 | FAIL (as expected) |
| 3 descent resolution | median 0.78 mm, max 11.44 | FAIL on the max tail |
| 4 normalized output range | **descent 5.3%** of span, all-frames 31.8% | FAIL (the headline number, exactly as documented) |
| 5 grasp window | 4–5 frames/episode | PASS (floor is 3) |
| 6 binarization | 251 action levels; leading-closed **median 8, max 57** frames | FAIL (reproduces "8 normal / 57 recovery" exactly) |
| 7 no recovery | 3/3 episodes with extra cycles, releases at 0.49/0.45/0.10 m | FAIL (the subset is 3 recovery episodes) |

Cross-checked on `08_01_2026_skip10_delta` (76 episodes, fast arm at skip10) to confirm it behaves on a different dataset shape: median 19.32 mm, descent dynrange 103%, 1 episode with no detectable close.

**Two measurement subtleties it has to handle**, both found during calibration:
* **The grasp window must expand both ways from the close.** On unbinarized data the recorded "action" is the *measured next jaw qpos*, so it only crosses the closed threshold once the jaws are ~90% shut — anchoring forward-only reported 2 frames for a ramp that actually spans 5. The motion threshold is also a fraction of the episode's own close travel, not an absolute number: an open jaw jitters ~0.1 mm/frame against its limit stop, the same order as the tail of a close ramp, so a fixed epsilon either bleeds into the jitter or truncates the ramp (measured: window swung 4 → 30 frames across a 20× epsilon change before this was fixed).
* **Cycle counting needs the engage/release hysteresis band** (`-0.013 / -0.0135`, mirroring `Ar4Mk3Env._update_grasp_engagement`). A single threshold miscounts recovery on unbinarized data — it cut `08_01`'s false extra-cycle episodes from 75/76 to 60/76.

**Where it runs:** on the 10-episode batch (the gate), again on any re-transform if checks 2/3 send us to skip 5–6, and **once more on the full dataset before launching training** — 10 episodes won't surface rare IK aborts or a block-preset-dependent problem, and that run is the last cheap check before the expensive step. Check 4 needs a finalized dataset (`meta/stats.json`).

Check thresholds and what each one is for:
* **check 2** Δ distribution at skip: median 2.4–3.0 mm, p99 ~4.3 mm, near-static (<2 mm) ≤10%, max/median ≤2.
* **check 3** descent resolution: last 12 frames before the close command — median ≤3.5 mm, max ≤7 mm.
* **check 4** normalized output range: descent frames ≥50% of the `[-1,+1]` span (`16_06` = 5.3%). **The headline gate.** Also available standalone as `measure_action_dynrange.py`, which prints the per-joint quantiles alongside.
* **check 5** grasp-window frame count per episode (expect ~5–8, floor ~3 spanning the transition).
* **check 6** binarization sanity: exactly two action values `{-0.014, 0}`, state channel still continuous (>10 distinct values), **and zero leading spurious-closed frames** (C6's regression test).
* **check 7** no recovery signature: exactly one grasp cycle (2 transitions) per episode; reports the grip height of any non-final release, which is the `partial_grasp` release-while-aloft fingerprint.

### V3. ✅ LANDED — [verification_gate.md](./verification_gate.md)
The runbook, with copy-pasteable commands and per-failure routing:

| stage | what | catches |
|---|---|---|
| 0 | `measure_scripted_arm timing` + `close-sweep`, `check_camera_parameterization` | the 0%-locked-grasps class, **before** spending a collection run |
| 1 | collect 10, then grep the log | `Grasp not locked` (check 8) and `Max steps` (check 1) — both warnings, both easy to scroll past |
| 2 | convert + transform | — |
| 3 | `check_dataset_health` (+ `measure_action_dynrange`) | checks 2–7, exit 1 on failure |
| 4 | full collect, re-run stages 1 and 3 at scale | rare aborts / preset-dependent problems 10 episodes can't show |
| 5 | eval config sanity | `n_substeps == skip`, env-config parity, `replan_steps` |

Two things it pins that are easy to get wrong from memory: `convert_data_to_lerobot` derives `repo_id` from `--output-dir`'s *basename* (it does not take a repo id, and `--output-dir` names rather than places the dataset), and checks 2 and 3 pull against each other — coarser skip lifts the median toward target while coarsening the descent, so failing both simultaneously is a finding about the velocity profile, not a threshold to tune away.

---

## Suggested commit sequence — ✅ ALL LANDED (03.08.2026)

Every code item is in. The next action is the **10-episode verification batch**
([verification_gate.md](./verification_gate.md)), not more implementation.

| # | item | commit | acceptance |
|---|---|---|---|
| 0 | ~~V1 harness~~ | (earlier) | ✅ |
| 1 | **F1** jaw geometry module + tests | `2a6c621` | ✅ 10 tests; a test drives the real model and confirms `gap = 0.8mm − 2·qpos` |
| 2 | **C5 + C4** gripper close fixes | `5fd58fd` | ✅ `timing` flips `lock_engaged` **False → True, 3/3 seeds**, both close targets |
| 3 | **C6 + E6** jaws-open initial state | `5105bf1` | ✅ jaws open at reset and stay open; verified the OLD warmup drove them to 0.0 |
| 4 | **C1 + C7** slow arm, depth off | `59f4577` | ✅ **4.94–6.73 s / 11.9–14.5 cm/s**, lock 3/3, EEF path unchanged |
| 5 | **C3** drop the recovery segment | `dc06e77` | ✅ `grep -c perturb-recovery` = 2, both `--no-perturb-recovery` |
| 6 | **T1 + T3** `--binarize-gripper` | `39725ea` | ✅ 10 tests incl. the constant-label / moving-state interaction |
| 7 | **F2 + E2** env-config factory | `95099fb` | ✅ `check_camera_parameterization`: **4/4 poses agree** (exits 1 on drift) |
| 8 | **E1** drop the noDR arm | `6dabd29` | ✅ 40 episodes/checkpoint; summarize/flatten/log round-tripped |
| 9 | **E4** close-depth recalibration | `2a6c621` | ✅ folded into F1 — same lines; no stale `GraspEngageConfig` anywhere |
| 10 | **E3** rate defaults + spec | `6dabd29` | ✅ suite and `run_policy_on_env` now **MATCH** on all three |
| 11 | ~~V2/V3 health checks + runbook~~ | (earlier) | ✅ |

Deviations from the plan as written, all deliberate:
* **E4 landed inside F1**, because the rename and the `engage_qpos` rewiring are the same four lines; splitting them would have meant editing them twice.
* **A fourth wrong-depth site** the plan did not list: a comment at `ar4_mk3_base.py:250` documenting the gate in the old `-(half_width − 0.5mm)` terms. Corrected.
* **`is_pinchable` needed both bounds, not just the squeeze end.** The 30 mm preset's `engage_qpos` is exactly `−0.0140`, i.e. *at* the limit, so checking only that end would call it pinchable. The binding constraint is the open end: the pad gap at full open is 28.8 mm, narrower than the block, so the jaws cannot straddle it at all.
* **`collect_trajectories` held its own duplicate copy of the `T`/`Q` camera extrinsics** (identical values, verified before removing).
* **`E3`'s "point `Args` at `SuiteConfig`" was not possible as stated** — `suite.py` imports `run_policy_on_env`, so that direction is circular. Both now read shared constants in `task_env_factory`.
* **`measure_scripted_arm timing` gained `--width-derived-close`** and now defaults to collection's full close, so `lock_engaged` reports what a collection run actually gets.
* **`check_camera_parameterization` now reads the live configs** and exits non-zero on disagreement, instead of statically demonstrating the old divergence. `--show-legacy` keeps the original numbers visible.

### Expected side effects to keep in mind when reading the batch
* **Episodes got ~17% longer at a given `dt`** (0.69–0.75 s → 0.81–0.87 s at dt=0.15). C5's tighter tolerance means every close now runs its full ramp instead of exiting early — this is the intended mechanism, not drift.
* **`max_episode_steps` and `replan_steps` are starting points, not measurements.** Confirm with the 1–5 `replan_steps` sweep on the first checkpoint.
* The unperturbed path is verified; the **perturbed worst case** (`s=0.7` × −10% dt against unscaled `max_steps`) is still only covered by the 10-episode batch.

---

## Risks / things to watch

* **The descent-resolution regression is designed-in** (doc §Conclusion): a near-uniform velocity profile gives the descent the same 2.4–3.0 mm step as transit, vs 0.6–0.8 mm today, against a ±7 mm `pinch_tol`. V-3 is the check; the remedy is skip 5–6, which then drags `n_substeps` with it (E3).
* **C6 changes the eval env for existing checkpoints** — any comparison against `16_06`-trained checkpoints after this lands is not apples-to-apples (their data starts with jaws closed). Record the break in `NOTES.md`.
* **E4 changes the meaning of an existing config field**; a stale pickled/JSON `GraspEngageConfig` with the old `close_depth_tol` would be silently misread. Renaming the field makes that a loud failure instead.
* **X1 is a scaling decision, not a code detail** — at 8× frames the full 2855-episode collect is a materially different job (render time, RAM per episode, disk). Measure on the 10-episode batch before committing to the full run.
