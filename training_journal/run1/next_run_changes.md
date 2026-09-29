# Next-run changes — deep dives (branched from [NOTES.md](./NOTES.md))

Detailed analyses + decisions for the fixes to apply **before** the next training run. Split out of `NOTES.md` (which was getting large). Topics: action timescale / skip, recovery injection, gripper channel, eval DR/noDR. More topics (grasp-phase oversampling, more episodes, …) to be appended as we work through them.

## Decision (batched fixes): apply ALL identified fixes, then re-collect from scratch, then train
Training is expensive, so we will NOT run the cheap skip-re-transform on existing data. Instead: fix everything identified here in collection → collect a ~10-episode batch → **run the verification checks below and only then collect the full set / start training** (training is the expensive step, and every claim this plan rests on is measurable on a dataset) → next training on that. Target state: slow arm, skip=10, gripper close-to-0, no recovery.

## Consolidated checklist for the next run

**Collection** (`collect_trajectories.py` / helpers / DR generator)
1. **Slow the scripted arm** — `IKConfig.integration_dt` 0.15 → **~0.005** so the pick→place runs ~5–7 s sim-time (~12–15 cm/s EEF), not ~0.7 s. **Also raise `IKConfig.max_steps` 700 → ~2000–3000**, or every episode aborts (`could not move above target`). Buys ~10× the model output range on the descent (see the timescale section) and gives faithful deploy a realistic speed. (Contact-shove reduction is **not** a reason — measured, it is 0.8 mm → 0.07 mm, negligible either way.) Verify by measuring episode sim-time, not by ratio — the speed response to `integration_dt` is strongly sublinear.
2. **No recovery injection** — drop `partial_grasp` + `wrong_approach` (set `perturb_recovery` off in every collection segment). Keep `ik_noise` + `offset_approach` for approach variety.
3. **Gripper full-close at collection — LOAD-BEARING, not optional.** Command `0.0` instead of the computed preload target. **Measured (3/3 seeds): with the current preload target the kinematic lock never engages at all** — the close exits ~0.15 mm short of first pad contact, registers zero pad↔block contacts, and the `require_pinch_contact` gate correctly refuses. A commanded close must be **≥ −11.0 mm** (24 mm block) to engage; the formula asks for −11.5 mm. Full close reaches −10.47 mm (1.13 mm of squeeze), 8 pad contacts, lock engages every time. Without this change a re-collect on current code produces **no locked grasps**. See the gripper section for the sweep.
4. **Fix the jaw-contact depth model — two stacked errors, the tolerance is the bigger one.** (a) `get_object_grasp_gripper_pos` assumes the pads touch at `-(pinch_half_width)`. Measured, the gap between the pads' inner faces is exactly affine in jaw qpos — **`gap(mm) = 0.8 − 2·q`** — so first contact is at **`q = 0.4 − half_width`**, i.e. the formula is off by **+0.4 mm, constant across every preset** (verified 19/22/24/27 mm; the 30 mm preset would need q = −14.6 mm, outside the actuator range, so it can never be pinched — consistent with distractor-only). That turns the intended 0.5 mm preload into 0.1 mm. (b) `gripper_pos_tolerance = 1 mm` is **10× that residual preload**, so `_interpolate_gripper` exits before the jaws reach the block. Fix both: derive the target from the pad geometry, and set the tolerance below the preload it is supposed to deliver. Item 3 makes collection independent of the formula, but it is still used by the recovery path and the same model is baked into the eval gates (item 13).
5. **Lighting DR — NOT changed this run** (deferred, see the section below). The collection DR distribution stays exactly as it was in `16_06` — no visual/dynamics axis touched — so eval can mirror it verbatim (item 11) and the run stays a clean A/B on the base fixes.
6. **Sensor aug stays OFF** — `image_aug` / `state_aug` off (as in `16_06`). Revisit in the sim2real phase, applied per-epoch in the loader.

**Transform** (`transform_skip_dataset.py`)
7. **skip = 10** on the slow-arm data → ~250–330 frames/ep, ~2.4–3.0 mm per step, 50 Hz faithful deploy. Chosen over skip20 because skip costs nothing at training time (cost is `steps × batch`, not dataset size) while skip10 gives 2× the distinct frames and 2× the descent resolution; 50 Hz is fine for deploy since `replan_steps=4–5` means inference every 80–100 ms either way.
8. **Binarize the gripper action** — new `--binarize-gripper`: gripper action → {−0.014 open, 0 closed} (threshold e.g. > −0.013 → 0). Load-bearing (kills `close_shallow`). State gripper untouched.
9. **Static filter — no change needed.** `16_06` used `min_action_delta=0.0005` rad (L2 over 6 joints ≈ 0.01°/joint) — gentle, removes only genuinely-frozen frames, not the moving descent. **Measured** on a slow-arm run: joint-Δ median at skip10 is ~0.011 rad (L2 over 6 joints) ≈ 20× the threshold, and the arm moves at a near-uniform rate, so 0.0005 is nearly inert and descent frames survive. Keep ~0.0005 (or drop it) + the gripper guard. Note the filter cannot fix the near-static-frame problem anyway — the gripper guard deliberately preserves exactly those frames (see the timescale section).

**Eval** (`suite.py` / `run_policy_on_env.py`)
10. **Drop the DR/noDR split** — all eval seeds full-DR from the same generator as collection; remove the noDR arm + `dr/nodr` split in logging.
11. **Fix DR-on in-distribution mismatches** — build the eval env from the *same env-config factory as collection* rather than patching flags one at a time. Four known gaps: `randomize_cameras=True` **and** `use_geometric_lookat=True` (the camera flag alone is actively harmful — see below), `randomize_object_yaw=True`, and the prompt string → training's exact `"pick the … target"` (lowercase, no period). Rule: eval env config == collection env config.
12. **Deploy at faithful rate** — `n_substeps = skip` (= 10) for the scored eval (`suite.py` defaults to 3). **`replan_steps` and `max_episode_steps` must be re-decided on the new data, not inherited.** `suite.py` currently defaults to `replan_steps=10 / max_episode_steps=1000`; the ablation's best combo (`replan=4, n_substeps=20, max=300`) was found on rate-*mismatched* data — skip3 deltas stretched over a 40 ms window — so it was partly compensating for the very mismatch this re-collect removes, and it does not transfer. Plan: small `replan_steps` sweep on the first checkpoint (1–5) before locking the scored config. A demo-length episode is ~250–330 env steps at `n_substeps=10`, so `max_episode_steps ≈ 800–1000` (≈2.5–3× demo length, room for retries) is the starting point.
13. **Recalibrate the close-depth thresholds off measured pad contact, not `pinch_half_width`.** `metrics.py`'s `close_shallow` fires on `close_cmd < -(pinch_half_width + close_depth_tol)` (−12.5 mm for a 24 mm block) and the lock's own close-depth gate uses `-pinch_half_width` — both inherit the depth model in item 4. Measured, a command must be **≥ −11.0 mm** to actually engage, so **[−12.5, −11.0) mm is a dead band**: physically unable to pinch, but not flagged shallow — it surfaces as `no_pinch` or `unknown` instead. ⇒ the reported `close_shallow` rate is an **under**-count, and some of the `no_pinch` (0.06–0.15 of failed attempts) is really shallow closing. Note the value the current dataset teaches (−11.7 mm) sits **inside** this band. Fix the threshold before reading the next run's miss anatomy, or the diagnosis will be wrong again.
14. Keep `--domain_rand=False` as a dev/visualization tool only (not a scored metric).

**Train**: config largely unchanged (pi0.5 full finetune); re-measure the plateau on clean data before deciding to scale episodes or steps.

---

## Verify on the first new data — BEFORE launching the training

Collect a small batch (~10 episodes) and run these checks on it. Every load-bearing claim above is measurable on a dataset; training is the expensive step, so nothing should start until these pass. Each check notes whether a failure is fixable at **transform** (cheap, re-run the transform) or needs a **re-collect** (expensive).

1. **Timescale landed** — episode sim-time 5–7 s (= raw_frame_count × 2 ms, so ~2500–3500 raw frames/ep), avg EEF speed 12–15 cm/s. *Also grep the collection log for `IK failed to converge` / `Max steps reached` and compare the successful-collection count to the requested count* — that is how a too-low `ik.max_steps` shows up. **Fix: re-collect** (tune `integration_dt` / `max_steps` first on a single episode).
2. **Δ distribution at skip10** — median 2.4–3.0 mm, p99 ~4.3 mm, near-static (<2 mm) ≤ ~10%, max/median ≤ ~2 (the fast arm was 22). **Fix: transform** (adjust skip).
3. **Descent resolution — the known risk of this plan.** Descent step (last 12 frames before the close command) median ≤ ~3.5 mm and max ≤ 7 mm, i.e. inside the `pinch_tol` envelope. The slow arm has a near-uniform velocity profile, so the descent no longer gets the fine steps it had at skip3 (0.6–0.8 mm) — if it comes out coarse, **fix: transform**, drop to skip 5–6 (deploy `n_substeps` follows).
4. **Normalized output range — the headline number the whole timescale change is for.** Take q01/q99 per action dim from the new dataset's own `meta/stats.json`, then for the descent frames compute `|a_i| · 2 / (q99_i − q01_i)` (the action's size as a fraction of the `[q01,q99] → [-1,+1]` span that quantile-norm maps onto) and take the worst joint. Today it is **5.3%**; expect **≥ ~50%**. If it lands far below, the timescale change did not buy what it was supposed to and the run should not start. **Fix: transform** (skip) or **re-collect** (dt), depending on which term is off.
5. **Grasp-phase representation — the known side effect of this plan.** Count frames per episode inside the close window. Expect ~5–8 (~3% of the episode), down from ~20% today, because the gripper ramp stays a fixed 50 mj-steps while the episode stretches ~8×. Sanity floor: at least ~3 frames spanning the open→closed transition, so the model sees the event and not just its endpoints. If it is degenerate, **fix: transform** — duplicate / weight grasp-window frames (this is the "grasp-phase oversampling" question, decided with data instead of by guess).
6. **Binarization sane** — the gripper action takes exactly two values {−0.014, 0} and nothing between; "closed" begins at/just before contact and persists through transport to the release; the gripper **state** channel is still continuous (≈ −0.0095/−0.011/−0.012 while holding the three block sizes). **Fix: transform**.
   * **Known issue to check here — spurious "closed" at episode start.** `m.qpos0` for both jaw joints is `[0, 0]`: the arm starts **fully closed**, and `release_gripper` is only called inside `grasp_at`, so every episode opens with a 0 → −14 mm ramp that binarizes to "commanded closed" while nothing is held. Measured on `16_06`: **8 leading frames** in normal episodes and **57** in the recovery episode (the whole `wrong_approach` detour runs with the jaws shut). Dropping recovery (item 2) kills the 57-frame case; the ~8-frame case (≈5 frames at skip10) remains in every episode. Cheapest fixes: set the jaws open in the initial qpos, or begin recording after the first `release_gripper`, or drop leading frames before the first open at transform.
7. **No recovery in the data** — zero episodes containing a release-while-aloft (the `partial_grasp` signature). Confirms the collection segments were re-configured, not just the docs. **Fix: re-collect.**
8. **The lock actually engaged — check this FIRST, it is the cheapest failure to miss.** ≥95% of collected episodes should have the kinematic lock engaged during the grasp (grep the collection log for `Grasp not locked`, which is a *warning* and therefore easy to run past for a whole batch). With the preload target the current code engages **0%** of the time; with full-close it engaged 3/3 in sim. A batch that lifts blocks by friction alone looks fine in a video and is worthless as data. **Fix: re-collect.**

---

### Running tally (short form)
1. **Timescale** — slow arm (`ik.integration_dt` 0.15→~0.005, `ik.max_steps` 700→~2500) to ~12–15 cm/s; skip=10 @ 50 Hz.
2. **Recovery** — none (drop `partial_grasp` + `wrong_approach`); keep ik_noise + offset_approach.
3. **Gripper** — full-close (0) at collection **+** binarize action to {−0.014, 0} at transform. Both load-bearing: the current preload target lands 0.5 mm shy of the engagement threshold, so the lock never engages and the demonstrated close cannot grasp. Also fix the pad-contact depth model (+0.4 mm, constant) and the 1 mm convergence tolerance, plus the miss-reason thresholds that inherit the same model.
4. **Eval** — in-distribution only; drop DR/noDR split; rebuild the eval env from collection's config (cameras + geometric lookat + object yaw + prompt string).
5. **Lighting DR** — deferred, no change this run.
6. **Sensor aug** — leave OFF for run 1 (confirmed off in `16_06`); add per-epoch in the sim2real phase.

---

## Action timescale / skip analysis (31.07.2026, substantially revised 02.08.2026)

Investigation into whether `skip=3` is the right action timescale for the next run, and whether the demonstrated arm is too fast. Two measurement sources: the 3-episode subset `Purple69/..._subset3ep` (FK via scene.xml grip site, Δ re-aggregated within episodes to simulate larger skips on top of the native skip3 stream) plus, from 02.08, a **headless harness that re-runs the scripted pick→place directly** at arbitrary `IKConfig` values (ROS imports stubbed) so the effect of `integration_dt` is measured rather than extrapolated.

### Two clocks, resolved
  * **Sim time per episode ≈ 0.7 s.** Collection records **1 frame per single `mj_step` (2 ms)** — the IK loop in `_solve_ik_for_site_pose` does compute→`mj_step`→`mj_forward`→`_record_step` once per iteration, and calls `mj_step` directly (never `env.step`, so `n_substeps` is not involved during collection). So sim-time = raw_frame_count × 2 ms, and native dataset frames (post-skip3) are 6 ms apart. (A `demo_pick_and_place` run printed `env.data.time = 1.464 s`; a clean headless re-measurement of the same scripted pick→place gives **0.70–0.75 s / 347–375 raw frames**, so that print was inflated by the default perturbation stack. Don't use it to calibrate a slowdown factor — measure.)
  * **The ~13 s "demo" wall-clock is not sim time.** That run had `--interface.render-steps`, which renders + captures both cameras every 2 ms step — ~10× wall inflation. It tells us render cost, not arm duration.
  * **⇒ the demonstrated arm is FAST: 0.68–1.30 m/s average EEF speed** (ep0 130, ep1 91, ep2 68 cm/s), 5–20× a realistic collaborative arm — independently confirmed by the harness at ~104 cm/s. This is baked in by the scripted IK controller (`ik.integration_dt=0.15` drives the ctrl target aggressively ahead each iter, converging in tens of steps).

### Key conceptual result (revised 02.08.2026 — the earlier "only the product matters" version was wrong)
  * **Δ = v_demo × skip × 2 ms** is right for the *median* step and useless on its own, because **pi0.5 quantile-normalizes the actions**: `ModelType.PI05` ⇒ `use_quantile_norm=True` (`openpi/training/config.py:186`), so each action dim is mapped `[q01, q99] → [-1, +1]`. **Absolute Δ is normalized away.** Two datasets with the same Δ *distribution shape* are equivalent no matter their scale — verified: slow-arm skip10 and skip20 differ 2× in millimetres and are identical in normalized units.
  * ⇒ the quantity that actually governs learnability is **`Δ_critical_phase / (q99 − q01)`**: how much of the model's output range the descent commands use. Non-uniformity matters *only* through the denominator — a heteroscedastic policy is fine (π(a|o) is a function of an observable phase), but a fat tail spends the output range on sprint steps the descent never uses.
  * **Measured on the real training data** (subset3ep frames + the full dataset's own `meta/stats.json` quantiles): all frames EEF median 3.28 mm / p99 25.6 mm, worst-joint action = **31.8%** of the [-1,+1] span; **descent EEF median 0.78 mm, worst-joint action = 5.3% of span** (4.1 / 5.7 / 7.0% per episode). The descent — the phase that decides the grasp — is commanded with ~1/20th of the model's output range.
  * Slow arm + skip10 predicts descent 2.4–3.0 mm against a p99 of 4.3 mm ⇒ **~50–100% of span, an order of magnitude more output resolution where it matters.** This, not "bigger Δ", is what slowing the arm buys. Caveat: comparative, not absolute — we have no measurement of pi0.5's actual output precision floor.
  * Demo-speed *also* matters via (a) **collection-time contact physics** — **measured 02.08.2026 and it does not matter.** Tracking object0 from the start of `grasp_at` to lock engage, 3 seeds: max shove **0.70–0.81 mm** at dt=0.15 vs **0.06–0.08 mm** at dt=0.005 (yaw 0.03° vs 0.04–0.08°, z rise 1.5 mm vs 0.1 mm). A ~10× reduction of a sub-millimetre effect, against a ±7 mm pinch tolerance on a 24 mm block — do not cite contact quality as a reason to slow the arm. (Caveat: harness has no DR / yaw randomization / ik-noise, so a misaligned approach could shove harder; there is an order of magnitude of headroom.) And (b) **deploy arm speed at faithful rate** (= v_demo). Note (b) is not a hard requirement: deploy speed is Δ/(control period), and `n_substeps > skip` already yields a slow arm from fast data — that is exactly what the ablation's `n_substeps=20` on skip3 data did (3.3 mm over 40 ms ≈ 8 cm/s). Slowing collection is what makes a slow arm coincide with a **1:1** rate.

### Measured Δ vs skip (native skip3 = current data)
| skip | frames/ep | joint Δ median (rad) | joint Δ max | EEF Δ median (mm) | EEF Δ p90 | EEF Δ max | descent Δ median | descent Δ max | near-static <2mm | <5mm |
|------|-----------|----------------------|-------------|-------------------|-----------|-----------|------------------|---------------|------------------|------|
| 3  | 148 | 0.017 | 0.12 | 3.3  | 13 | 30  | 0.6 | 8  | 40% | 56% |
| 6  | 74  | 0.039 | 0.21 | 7.1  | 27 | 53  | 1.2 | 13 | 28% | 43% |
| 9  | 49  | 0.052 | 0.29 | 11   | 40 | 79  | 1.8 | 17 | 20% | 33% |
| 12 | 37  | 0.080 | 0.36 | 16   | 51 | 105 | 2.2 | 19 | 13% | 26% |
| 21 | 21  | 0.137 | 0.61 | 31   | 81 | 178 | 5.0 | 21 |  5% | 10% |

("descent" = last 12 native frames before the jaws close on the block.)

### Reading
  * **SNR is a non-issue — retracted.** The `actions` field is essentially an exact per-step state diff (residual median 0.0000 rad, p90 0.0008); smoothing left no noise floor. Coarser skip buys nothing on SNR.
  * **Motion is very non-uniform (9× skew: EEF median 3.3 mm vs max 30 mm at skip3)** — but **the earlier explanation was wrong**. The arm does *not* creep in precision phases: measured per-mj-step, **every near-static frame is a gripper-ramp frame** (`_interpolate_gripper` explicitly parks the arm — `data.ctrl[:6] = data.qpos[...]`, `ar4_mk3_robot_interface.py:323` — then ramps the jaws over `gripper_action_steps=50` mj-steps ≈ 100 ms). On the raw sim trace it is 100% of parked frames; on the real (smoothed, static-filtered) dataset **72% of near-static frames** are gripper-ramp frames (96/134), the rest being descent creep and settles. So the fine steps are *not* where we want them — the descent at skip3 is 0.6–0.8 mm, barely below the 3.3 mm median.
  * **Skip cannot fix the near-static mass**, because the ramps are long in *time* (100 ms each, two per episode) — decimating time keeps them proportionally: fast arm at skip20 is still 33% <2 mm. Slowing the arm removes them by **dilution** (the ramp stays 50 mj-steps while the motion stretches ~8×): 40% → 4.5%.
  * **The grasp descent stays fine well past skip3** — 0.6 mm/step at skip3, still 2.2 mm median at skip12 on a ±12 mm-capture block. The "gaps too large for the grasp" worry does not bite below ~skip12; the only high-skip risk is the descent **max-step tail** (~19–21 mm ≈ one block-width in a single step at skip12–21), which closed-loop replanning covers. **Note this table is fast-arm-only**: once the arm is slowed the velocity profile becomes near-uniform (max/median 22× → 1.6×), so the descent gets the *same* step as transit and these per-skip descent figures no longer apply.
  * **~Half the frames being near-static** (56% <5 mm, 40% <2 mm) is a real pathology — the imitation loss is dominated by "barely move" targets, a plausible driver of the dithering / "learns to sit still" behavior — but it is the **gripper ramp**, not the timescale, and the static filter can't remove it (the `--gripper-eps` guard deliberately preserves exactly those frames). Separately: 167 Hz faithful deploy (infeasible) and a 6 ms actuator window (< the ~10–15 ms settling time → the closed loop dithers around a lagging target; this is why the ablation's best deploy was `n_substeps=20`, a 40 ms window = the arm executing each 3.3 mm delta at ~8 cm/s). Those two *are* fixed by skip alone.

### Measured: what `integration_dt` actually does (headless harness, 3 seeds, no perturbations, pick→place only)

| `integration_dt` | sim time | avg EEF speed | raw frames |
|---|---|---|---|
| 0.15 (current) | 0.70–0.75 s | ~104 cm/s | 347–375 |
| 0.075 | 0.79–0.90 s | ~90 cm/s | 395–451 |
| 0.03 | 1.18–1.46 s | 55–61 cm/s | 590–732 |
| 0.015 | 1.9–2.5 s | 32–38 cm/s | 946–1236 |
| **0.005** | **4.9–6.6 s** | **12–15 cm/s** | 2434–3306 |
| 0.002 | 11.7–16.2 s | 5–6 cm/s | 5838–8079 |

The response is strongly **sublinear** — a 5× cut in `integration_dt` buys only ~1.9× slowdown — because `integration_dt` only rate-limits motion while the `max_update_norm=1.5` clamp binds, and at dt=0.15 that is only for errors >~12 cm, so most moves are already actuator-saturated sprints. **⇒ ~0.005, not ~0.03.** At that dt the default `ik.max_steps=700` is exhausted and every episode aborts; 1500 works, use ~2000–3000 (the speed-DR path then scales it by 1/s automatically). EEF path length is unchanged across all dt (0.80 m), so only the timing changes, not the trajectory shape.

### Conclusion — slow the arm, then skip=10
  * **Re-collect with `ik.integration_dt ≈ 0.005`, `ik.max_steps ≈ 2500`, then `skip=10`, deploy `n_substeps=10` @ 50 Hz.** Measured on the slow arm: ~250–330 frames/ep, per-step Δ 2.4–3.0 mm, p99 4.3 mm, near-static (<2 mm) 4–6%, descent at ~50–100% of the normalized output span vs 5.3% today.
  * **skip=10 over skip=20** because skip is training-cost-neutral (cost is `steps × batch`, independent of dataset size) while skip10 gives 2× the distinct frames and 2× the descent resolution. Context: the current run is 2855 episodes / 428k frames, so 20k steps @ bs32 = 1.5 epochs — and the eval funnel was already flat there, i.e. repetition was never the binding constraint, so more distinct frames is free upside.
  * **What this does NOT fix:** the slow arm has a near-uniform velocity profile, so the descent gets the same 2.4–3.0 mm step as transit — it does not restore the fine-grained approach the fast arm had (0.6–0.8 mm). Against the ±7 mm `pinch_tol` that is ~2–3 quanta of slack, which closed-loop replanning should cover, but it is the thing to watch if grasp precision does not improve.
  * **Side effect to check on the first new dataset:** dilution cuts both ways. The gripper ramp is a fixed 50 mj-steps regardless of arm speed, so the close event goes from ~20% of an episode's frames to ~3% (≈5–8 frames). Combined with binarization each close becomes a handful of frames with an unambiguous label, which is probably what we want — but that is the **grasp-phase oversampling** question in concrete form, and it is cheap to fix at transform time (duplicate / weight grasp-window frames) and expensive to discover after a run.
  * **Arc-length (constant-displacement) resampling** would achieve the same normalized ratio from the *existing* recordings, and the earlier rejection of it ("uniform-time skip keeps fine steps where the arm is slow") rested on the premise now disproved above. Not taken: collection is not the bottleneck (training is), and it brings its own pitfalls — non-constant time between frames breaks the `n_substeps = skip` deploy invariant, and the zero-displacement gripper ramps need a hand-tuned frame budget.

---

## Recovery injection: drop both modes (31.07.2026)

What was actually in `16_06` data (per `collect_mixed.sh`): recovery was one segment, not every episode — **70%** ik_noise no-recovery, **20%** ik_noise **+ recovery** (both `wrong_approach` AND `partial_grasp` fire every episode in this segment), **10%** offset_approach no-recovery. So ~20% of episodes carry a hover-misalign detour + a friction-slip-then-regrasp.

### Decision: drop BOTH `partial_grasp` and `wrong_approach` completely for the next re-collect. Zero recovery injection.

  * **`partial_grasp` — unsound premise + a direct premature-release signal.** The slip is driven by a secret friction change (`partial_grasp_slip_friction=0.55` on jaw+block geoms), **invisible in state and images**. So the policy is shown a *centered, correct-looking* grasp that fails for a cause it cannot perceive — the only lesson ("good grasps sometimes fail") is unactionable. Worse, when the marginal grip holds through the lift (`redetected.z > 0.045`) the expert calls `release_gripper`, recording an explicit **open while holding a lifted block** = a literal premature-release demo (minority of the 20%, but exactly the failure we see: 70k `premature_release_count_mean=1.98`). This is anti-learning; no salvage.
  * **`wrong_approach` — not poison, but dropped too.** Its cause (18–35 mm lateral offset at 50–100 mm hover) IS visible in the wrist cam and the correction is clean (never touches the block), so it teaches an observable, actionable skill. But: (1) it doesn't hit the actual bottleneck (`close_shallow` / grasp-width, not lateral approach); (2) it adds **grasp-phase multimodality** ("descend" vs "translate-then-descend" at a near-centered hover) — exactly what a from-scratch BC policy handles worst; (3) its value is **unmeasurable** until the base grasp works, and it confounds the A/B for the base fixes. So cut it this round.
  * **Approach diversity is preserved** without the fail→fix structure: `ik_noise` (IK jitter) + `offset_approach` (waypoints on a disk above target) already vary the approach path. Dropping the recovery segment removes only the deliberate-failure episodes.
  * **Real recovery strategy = post-training DAgger/HG-DAgger.** Train the clean baseline, roll it out, let `metrics.py` catch its ACTUAL failures (off-side hover, shallow close, wrong height, block shoved), have the scripted expert take over from those states and record the correction → recovery data on the state distribution the policy really visits (not a hand-guess), and measurable (does the DAgger round move the funnel?). All three pieces already exist (scripted expert from arbitrary state, sim, failure detection); it needs a policy in the loop, so it's a next-round move. This makes the upcoming run a **clean test of the base fixes** (timescale + gripper + slow arm) with no recovery multimodality confounding it.

---

## Gripper channel: full-close at collection + binarize the action (02.08.2026)

Targets the dominant miss reason `close_shallow` (0.57–0.85 of failed grasp attempts) and the premature-release / jaw-pulsing behavior.

### Root cause 1: the demonstrated close never actually touches the block (measured 02.08.2026)
**This supersedes the "changing the collection command is a no-op" claim below — that was wrong.**

Two errors stack. **(a) Depth model.** `get_object_grasp_gripper_pos` returns `-(pinch_half_width − 0.5 mm preload)` = **−11.5 mm** for a 24 mm block, assuming the pads contact at `-(half_width)` = −12.0 mm. Measured, the gap between the pads' inner faces is exactly affine in jaw qpos — `gap(mm) = 0.8 − 2·q` — so first contact is at `q = 0.4 − half_width` = **−11.6 mm**, a constant **+0.4 mm** off the formula for every preset. The intended 0.5 mm preload is therefore really 0.1 mm. **(b) Convergence tolerance.** `gripper_pos_tolerance = 1 mm` is 10× that residual, so `_interpolate_gripper` exits as soon as the ramp completes — ~0.15 mm *short* of contact.

Commanded-close sweep, 24 mm block (first contact −11.6 mm), counting only pad↔object contacts:

```
 cmd(mm)   final jaw   pad contacts   pinching   LOCK
  -12.00    -12.197         0          False    False
  -11.60    -11.836         0          False    False
  -11.50    -11.745         0          False    False   <- what the formula commands today
  -11.00    -11.429         8          True     True
  -10.50    -11.345         8          True     True
    0.00    -10.466         8          True     True    <- item 3, 1.13 mm of squeeze
```

⇒ engagement needs a command **≥ −11.0 mm**, i.e. ~0.6 mm past first contact (enough that the tolerance-limited close still lands in contact with registerable penetration). The formula asks for −11.5 mm and misses. 3/3 seeds, both ends.

  * ⇒ **since the pinch-contact gate (`5c30eb4`, 2026-07-14) a collection run on current code produces zero locked grasps.** `16_06` was collected 2026-06-16, before that gate, so it locked under the offset-only gate — **contactlessly**, jaws ~0.15 mm clear of the block. That is the journal's own manual-eval complaint ("half levitating with some gaps between object and jaws") present in the *demonstrations*, not just in eval.
  * ⇒ **the current dataset teaches a close command that physically cannot grasp.** Confirmed on the real data: the holding cluster is at **−11.69 / −11.75 / −11.76 mm** per episode, while engagement needs ≥ −11.0 mm. A policy imitating the demo perfectly cannot pass the eval's pinch gate. This is a better account of the flat `grasped ≈ 0.58–0.62` than block-size regression alone, and it is an independent reason **re-collection beats re-transform** (no relabelling fixes a demo whose physics never pinched).
  * ⇒ the miss-reason thresholds inherit the same depth model, leaving a [−12.5, −11.0) mm dead band; see checklist item 13.

### Root cause 2: the recorded action is the *measured next qpos*, not the command
Confirmed in `trajectory_data_collector.py` (line 495: "action: next state … gripper state"). So the gripper action during a grasp is the *physical* jaw qpos, which stalls wherever the jaws stop. Consequences:
  * The action **varies with block size**, so the policy must regress block half-width from pixels to know how far to close — a task with zero payoff (the kinematic lock, and a real current-limited gripper, make "just close all the way" safe on any block).
  * Secondary: the close is a 50-step interpolated ramp (`_interpolate_gripper`) recorded per mj-step, so the action passes through mid-values → the policy emits *partial* closes and wobbles across the open/close boundary → weak grips + premature drops + jaw pulsing.

### Two fixes (different jobs — both required)
| change | fixes | required? |
|---|---|---|
| **Collection: command 0** (`get_object_grasp_gripper_pos → 0.0`) | the jaws actually reach the block; the kinematic lock engages at all | **yes, load-bearing** — without it the next dataset has no locked grasps |
| **Transform: binarize action → {−0.014 open, 0 closed}** (`--binarize-gripper` in `transform_skip_dataset`; threshold e.g. > −0.013 → 0, else −0.014) | width-regression + ramp mid-values (`close_shallow`), and makes the recorded action independent of the broken depth formula | **yes, load-bearing** |

  * **State gripper channel: leave untouched.** Its physical value is a useful proprioceptive cue — open −0.014, holding a block ≈ −0.0095/−0.011/−0.012 (the three graspable sizes 19/22/24 mm), closed-on-nothing ≈ 0 (a miss). The policy may use it as input; it just never has to reproduce it as an action. **Width-invariance is the point:** one "close" command generalizes across every block size.
  * **Deploy consistency:** policy outputs closed = 0 → env commands 0 → jaws stall on the block at ≈ −0.0105 → next state ≈ −0.0105, matching the training state. The action(command)/state(result) "mismatch" is correct — that's what a command channel is.

### On the "binarize-only" mismatch (superseded 02.08.2026)
The original argument here was that binarize-only would be nearly equivalent because "the demo physics ran a −0.0105 command and the qpos stalls at −0.0105 either way". Both halves are false: the demo command is −11.5 mm, the jaws stop at −11.7 mm, and they **never reach the block at all** (root cause 1). So binarize-only would relabel a contactless close as "closed" — teaching the right label on top of physics that never grasped. Commanding 0 at collection is what makes the demo a real grasp; binarizing is what makes the *label* width-invariant. Neither substitutes for the other.

### Predicted, testable effect
`close_shallow` is defined in `metrics.py` as `close_cmd < -(pinch_half_width + tol)` — "commanded a gripper still too open." With the command pinned to 0, that can **never** fire → `close_shallow` should collapse to ~0 by construction. The real question the next eval answers: do those attempts convert to *engaged locks*? They should — a full close now genuinely reaches the block and produces the two-sided pinch contact the gate needs (verified in sim: 8 contacts, `pinching=True`, lock engages, 3/3 seeds). If grasp rate jumps and `close_shallow` vanishes together, the mechanism is confirmed. Note this prediction is only meaningful once the item-13 threshold recalibration lands; on the current threshold `close_shallow` is an under-count.

### Footnotes (non-issues here)
  * **"Thin block" degeneracy doesn't apply.** It would only bite if a graspable block's hold-state (~0) collided with closed-on-nothing — but the smallest graspable is 19 mm (−0.0095), well separated. Vision also disambiguates hold-vs-miss.
  * **Real-hardware caveat (deploy, not data):** commanding full-close 0 on a rigid block would stall a *position*-controlled gripper servo at max current. Fine in sim (lock) and on a current/torque-limited real gripper — confirm the real AR4 gripper is compliant / current-limited before deploy. Add to the sim2real checklist.
  * **Not minor after all:** `_interpolate_gripper` won't early-converge when target 0 can't be reached with a block in the jaws (TODO at `ar4_mk3_robot_interface.py:373`). That is now a *feature* — it is precisely why the full-close command reaches the block, where the preload target exits early via the 1 mm `gripper_pos_tolerance` and stops short. Do not "fix" the early-exit without re-checking that the lock still engages.
  * **Checked:** a full-close command does not shove the block harder in the pre-lock window — measured shove is 0.79/0.70/0.81 mm at dt=0.15 and 0.06–0.08 mm at dt=0.005, with the full-close command in both cases.

---

## Eval: drop the DR/noDR split, eval only in-distribution (02.08.2026)

### Verified in code
  * **Collection has zero clean episodes.** `generate_random_domain_rand_config` samples every visual axis unconditionally (colors, all materials incl. floor/wall/table/all robot links, lighting mood incl. dim/harsh, table geometry, mass/friction, block presets, 20–30 props, wall art, arm cables). Only `randomize_cameras` / `randomize_arm_dynamics` are gated flags; `collect_trajectories` calls it once per episode with no no-DR path. **100% of training frames are full-visual-DR.**
  * **DR-off eval renders raw `scene.xml` defaults.** `_apply_domain_randomization` early-returns when `domain_rand is None` (base.py:665), and the eval passes `None` for DR-off. Defaults: flat **white** robot links, flat **gray** wall (0.5), flat **brown** table (0.6 0.4 0.25), default **checker** floor, pure **yellow** opaque object (smallest 19 mm preset, no layer lines), pure **red translucent (α=0.5)** target, flat lighting (headlight 0.6 + dim scene_light 0.2, top light off, ~no shadows), and all props / wall-art / arm-cables **hidden** (α=0 defaults).
  * ⇒ **noDR is the single most-OOD point in appearance space** — it differs from *every* training episode on *every* axis at once (the exact center DR deliberately never samples). Nasty specifics: target is *translucent* in noDR but *opaque* in training (right where the place phase must localize it); object is always the smallest block.

### Decision (agreed): remove the DR/noDR notion from evals entirely
Rationale (user): (1) noDR is just another arbitrary sim scene, NOT the target deployment, so treating it as a baseline that "should be better" is meaningless — the honest pinch-gated numbers (noDR ≤ DR, e.g. 70k transported DR 0.50 vs noDR 0.15) are just OOD degradation, read backwards as "DR is hard." (2) The real deployment is a fixed known-background setup, but it does NOT look like today's noDR — it will bring its *own* OOD elements; so adding today's noDR to training buys nothing for deploy. (3) We are not at sim2real stage: the policy struggles even in-distribution in sim, so sim2real would fail catastrophically. **First find a recipe that works flawlessly in-distribution in sim; sim2real is out of scope for the next run.**

### Implementation
  1. **Drop the noDR arm from `suite.py`** — remove the `n_seeds`/DR-off group; all eval seeds draw a fresh full-DR config from the *same generator as collection*, k repeats each. Move the whole seed budget to DR seeds (raises precision of the in-distribution estimate; between-seed std was ~0.30–0.45). Rip out the `dr`/`nodr` split in the mlflow logging + summary.
  2. **Make the surviving DR-on path genuinely in-distribution** — four silent train/eval mismatches (in `run_policy_on_env._resolve_prompts` and `_build_env`) that matter more now that DR-on is the ONLY eval. **Verified in code + numerically, 02.08.2026.**
     * **Camera flag:** eval calls `generate_random_domain_rand_config()` with defaults ⇒ `randomize_cameras=False`, but collection used `--randomize-cameras`. So every DR-on eval scene sits at the single fixed default camera pose while training saw random anchor-hull poses. Set `randomize_cameras=True` in eval DR generation. (`randomize_arm_dynamics=True` already matches.)
     * **Geometric lookat — flipping the camera flag ALONE is worse than leaving it off.** `_build_env` never passes `use_geometric_lookat`, so eval runs the `(T, Q, False)` parameterization while collection runs `(T_GEOMETRIC, Q_GEOMETRIC, True)`. With no camera DR the two are identical by construction (verified: same lookat/azimuth/elevation/distance — that's what T_GEOMETRIC was calibrated for). But `_apply_camera_offset` *adds* the sampled offset to the base translation, and the anchor offsets are 0.1–1.0 m validated in collection's parameterization — so on the eval base they land somewhere else entirely. Measured, offset `[-0.031, 0.308, -0.004]`: collection → `azim 209.3°, elev −31.2°, dist 1.11`; eval → `azim 264.2°, elev −7.9°, dist 0.855` (camera on the other side of the scene). Same for the other samples. ⇒ `randomize_cameras=True` without `use_geometric_lookat=True` samples an OOD camera distribution, i.e. it would make eval *less* in-distribution, not more. Both flags or neither.
     * **Object yaw:** `collect_mixed.sh` passes `--randomize-object-yaw`; `_build_env` leaves the env default (`False`). So every eval block spawns axis-aligned while training saw random yaw — eval is *easier* than training here, and it hides the diagonal / OOD-grasp failure described in NOTES' Observations. Set `randomize_object_yaw=True`.
     * **Prompt string:** DR-on eval builds `"Pick the … target."` (capital P + trailing period) vs training's `"pick the … target"`. Match training's exact format.
     * **Rule going forward: eval env config == collection env config, exactly.** These four were found one at a time by reading `_build_env` against `collect_trajectories`; assume there are more. The durable fix is **one shared env-config factory** consumed by both collection and eval (the eval-only knobs — `n_substeps`, `kinematic_grasp`, `relative_action_scale`, `include_images_in_obs` — layered on top), not a list of flags kept in sync by hand.
  3. **Keep `--domain_rand=False` in `run_policy_on_env` as a dev/visualization tool** (eyeballing the policy in a clean scene is easier), just NOT as a scored suite metric. Distinction: clean scene as debugging aid (keep) vs noDR as a reported number (drop).

Net: a single, honest, in-distribution success metric — the thing to nail before sim2real is even on the table.

---

## Lighting DR: DEFERRED — no change for the next run (02.08.2026)

**Decision: skip this for run 1.** The next run is a clean test of the base fixes (timescale + gripper + eval in-distribution); changing the visual DR distribution at the same time confounds that A/B, and the analysis below says the intended fix wouldn't do much anyway. Kept here in case dark scenes turn out to matter after the run.

Original observation: some scenes render too dark (the `dim` / low-ambient end of the lighting-mood distribution), making the block hard to perceive. Intended fix was to raise the floor on the dark end so no scene is near-unperceivable, keeping lighting variation otherwise (varied light is required for sim2real; only the darkest tail trimmed).

Why it's also low-value as specified: `dim` is only **5%** of episodes (`_LIGHT_MOODS` in `domain_rand_config_generator.py`), so clamping it changes 1 episode in 20. The genuinely dark renders more likely come from `key` (**35%**, ambient 0.02–0.10) and `harsh` (**15%**, ambient 0.01–0.06), both combined with `p_active_aux` 0.5–0.65 — i.e. a sizeable share of scenes lit by the headlight plus a single aux light, with near-black fill in shadow. If we revisit this, the lever is the **ambient floor across `key`/`harsh` and/or `p_active_aux`**, not the `dim` bucket.

---

## Sensor augmentation: confirmed OFF in 16_06, leave off for run 1 (02.08.2026)

`obs_augmentation` (white balance, hue/sat, vignette, motion blur, frame drop/freeze, proprioception noise) is a **sim2real sensor bridge** — none of it exists in the sim eval. Checked whether `16_06` used it (`transform_skip_dataset` applies it at *build* time, flags default False, name has no `aug` marker):
  * **state_aug: OFF, proven** — the state 2nd-difference std is 0.0013–0.0069 rad/joint (real fast-arm accel), not the ~0.030 rad white-noise floor that the 0.7°/frame jitter would produce.
  * **image_aug: off, high confidence** — paired flag with state_aug (shared strength/seed), absent from the exhaustive descriptive name, no frame-freeze duplicates (the mild corner darkening is ordinary scene-light falloff).

So `16_06` had **no** sensor aug — "keeping" it would mean adding it new. Decision: **leave it off for run 1** so the in-distribution grasp goal isn't burdened by extra observation difficulty (aug was never a factor, so nothing is lost). Add it in the sim2real phase, and when added, apply it **per-epoch in the training loader** (not baked at transform) — a single frozen draw per frame gets memorized across the 5–22 epochs the run trains, giving the difficulty cost without the invariance benefit. (pi0.5 can likely learn through baked aug, but per-epoch is the correct form.)
