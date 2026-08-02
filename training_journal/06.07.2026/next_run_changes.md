# Next-run changes — deep dives (branched from [NOTES.md](./NOTES.md))

Detailed analyses + decisions for the fixes to apply **before** the next training run. Split out of `NOTES.md` (which was getting large). Topics: action timescale / skip, recovery injection, gripper channel, eval DR/noDR. More topics (grasp-phase oversampling, more episodes, …) to be appended as we work through them.

## Decision (batched fixes): apply ALL identified fixes, then re-collect from scratch, then train
Training is expensive, so we will NOT run the cheap skip-re-transform on existing data. Instead: fix everything identified here in collection → re-collect a fresh dataset (slow arm, skip≈20, gripper close-to-0, no recovery) → start the next training on that.

## Consolidated checklist for the next run

**Collection** (`collect_trajectories.py` / helpers / DR generator)
1. **Slow the scripted arm** — `IKConfig.integration_dt` ~0.15 → ~0.03 (+ more interp steps as needed) so the pick→place runs ~5–8 s sim-time (~15 cm/s EEF), not ~1.5 s. Fixes contact-shove artifacts + gives faithful deploy a realistic speed.
2. **No recovery injection** — drop `partial_grasp` + `wrong_approach` (set `perturb_recovery` off in every collection segment). Keep `ik_noise` + `offset_approach` for approach variety.
3. **Gripper full-close at collection** — `get_object_grasp_gripper_pos → 0.0` (command full close; jaws stall on the block, lock engages on contact).
4. **Lighting DR — NOT changed this run** (deferred, see the section below). The collection DR distribution stays exactly as it was in `16_06` — no visual/dynamics axis touched — so eval can mirror it verbatim (item 10) and the run stays a clean A/B on the base fixes.
5. **Sensor aug stays OFF** — `image_aug` / `state_aug` off (as in `16_06`). Revisit in the sim2real phase, applied per-epoch in the loader.

**Transform** (`transform_skip_dataset.py`)
6. **skip ≈ 20** (target ~25 Hz faithful deploy on the slow-arm data).
7. **Binarize the gripper action** — new `--binarize-gripper`: gripper action → {−0.014 open, 0 closed} (threshold e.g. > −0.013 → 0). Load-bearing (kills `close_shallow`). State gripper untouched.
8. **Static filter — no change needed.** `16_06` used `min_action_delta=0.0005` rad (L2 over 6 joints ≈ 0.01°/joint) — gentle, removes only genuinely-frozen frames, not the moving descent. At skip20 every real motion is ~40× bigger, so 0.0005 is nearly inert and descent frames survive regardless. Keep ~0.0005 (or drop it) + the gripper guard; the skip3 "half the frames near-static" issue was inherent to small deltas, fixed by skip20, not by the filter.

**Eval** (`suite.py` / `run_policy_on_env.py`)
9. **Drop the DR/noDR split** — all eval seeds full-DR from the same generator as collection; remove the noDR arm + `dr/nodr` split in logging.
10. **Fix DR-on in-distribution mismatches** — build the eval env from the *same env-config factory as collection* rather than patching flags one at a time. Four known gaps: `randomize_cameras=True` **and** `use_geometric_lookat=True` (the camera flag alone is actively harmful — see below), `randomize_object_yaw=True`, and the prompt string → training's exact `"pick the … target"` (lowercase, no period). Rule: eval env config == collection env config.
11. **Deploy at faithful rate** — `n_substeps = skip` (≈20) for the scored eval.
12. Keep `--domain_rand=False` as a dev/visualization tool only (not a scored metric).

**Train**: config largely unchanged (pi0.5 full finetune); re-measure the plateau on clean data before deciding to scale episodes or steps.

---

### Running tally (short form)
1. **Timescale** — slow arm (`ik.integration_dt` ~0.15→~0.03) to ~15 cm/s; skip≈20 @ 25 Hz.
2. **Recovery** — none (drop `partial_grasp` + `wrong_approach`); keep ik_noise + offset_approach.
3. **Gripper** — full-close (0) at collection **+** binarize action to {−0.014, 0} at transform.
4. **Eval** — in-distribution only; drop DR/noDR split; rebuild the eval env from collection's config (cameras + geometric lookat + object yaw + prompt string).
5. **Lighting DR** — deferred, no change this run.
6. **Sensor aug** — leave OFF for run 1 (confirmed off in `16_06`); add per-epoch in the sim2real phase.

---

## Action timescale / skip analysis (31.07.2026)

Investigation into whether `skip=3` is the right action timescale for the next run, and whether the demonstrated arm is too fast. Measured on the 3-episode subset `Purple69/..._subset3ep` (analysis script `scratchpad`/job-tmp; FK via scene.xml grip site, Δ re-aggregated within episodes to simulate larger skips on top of the native skip3 stream).

### Two clocks, resolved
  * **Sim time per episode ≈ 1.2–1.5 s.** Confirmed directly: a `demo_pick_and_place` run printed `env.data.time = 1.464 s` for the full pick→place (go-home excluded, as in training). Collection records **1 frame per single `mj_step` (2 ms)** — the IK loop in `_solve_ik_for_site_pose` does compute→`mj_step`→`mj_forward`→`_record_step` once per iteration, and calls `mj_step` directly (never `env.step`, so `n_substeps` is not involved during collection). So sim-time = raw_frame_count × 2 ms, and native dataset frames (post-skip3) are 6 ms apart.
  * **The ~13 s "demo" wall-clock is not sim time.** That run had `--interface.render-steps`, which renders + captures both cameras every 2 ms step — ~10× wall inflation. It tells us render cost, not arm duration.
  * **⇒ the demonstrated arm is FAST: 0.68–1.30 m/s average EEF speed** (ep0 130, ep1 91, ep2 68 cm/s), 5–20× a realistic collaborative arm. This is baked in by the scripted IK controller (`ik.integration_dt=0.15` drives the ctrl target aggressively ahead each iter, converging in tens of steps).

### Key conceptual result (agreed after discussion)
  * **For the learning signal, demo-speed and skip are NOT independent axes** — only their product matters: **Δ = spatial displacement per policy step = path/frame_count = v_demo × skip × 2 ms**. State is joint pos, action is joint-pos delta, images are pose snapshots — none contains time or velocity, so two datasets with equal Δ are byte-identical regardless of how speed×skip factorize. On a fixed recording, **skip is the whole representational knob** (cheap re-transform).
  * Demo-speed only genuinely matters via (a) **collection-time contact physics** (a fast arm shoves/flips/presses the block → disturbed poses in the data; secondary, since the dominant failure `close_shallow` is speed-independent), and (b) **deploy arm speed at faithful rate** (= v_demo; real-world safety/contact, not learning).

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
  * **Motion is very non-uniform (9× skew: EEF median 3.3 mm vs max 30 mm at skip3).** The arm creeps in precision phases, jumps in transit. This is *benign for uniform-time skip* — it keeps fine steps exactly where the arm is slow (descent) and coarse steps in transit, which is what you want. Not an argument for arc-length resampling.
  * **The grasp descent stays fine well past skip3** — 0.6 mm/step at skip3, still 2.2 mm median at skip12 on a ±12 mm-capture block. The "gaps too large for the grasp" worry does not bite below ~skip12; the only high-skip risk is the descent **max-step tail** (~19–21 mm ≈ one block-width in a single step at skip12–21), which closed-loop replanning covers. skip20 is borderline, not broken; **~10–12 is clearly safe**.
  * **The real pathology at skip3 is that ~half the frames are near-static** (56% <5 mm, 40% <2 mm) — the imitation loss is dominated by "barely move" targets, a plausible driver of the dithering / "learns to sit still" behavior. Plus 167 Hz faithful deploy (infeasible) and a 6 ms actuator window (< the ~10–15 ms settling time → the closed loop dithers around a lagging target; this is why the ablation's best deploy was `n_substeps=20`, a 40 ms window = the arm executing each 3.3 mm delta at ~8 cm/s). Moderate skip fixes all three: skip12 → 13%/26% near-static, ~28 Hz deploy.

### Conclusion — two paths (not skip3 vs skip20)
  * **Cheap (re-transform only): skip ≈ 10–12 on existing data.** Halves the near-static share, feasible-ish deploy, descent stays ~2 mm. Deploy faithfully at `n_substeps=10–12` and check the dither is gone. NOT skip20 (its descent is coarse *because the arm is fast* — 5 mm median / 21 mm tail). [Not taken — see batched-fixes decision above.]
  * **Real fix (re-collect): slow the scripted arm** (`ik.integration_dt` 0.15→~0.03, more interp steps) to ~15 cm/s, **then skip≈20 @ 25 Hz.** The same skip20 that is coarse on the fast arm becomes ideal on a slow one: transit Δ ~6 mm, descent ~1–2 mm, ~130 frames, feasible rate, a long actuator window with **zero deploy hack** (faithful deploy = realistic speed), and it removes the contact-physics artifacts simultaneously. Faithful deploy on the *fast* data can only ever reproduce the fast arm; the ablation's `n_substeps=20` win was a stretch-hack standing in for a slow arm.
  * Note: demo-speed enters here only for **deploy speed + contact quality**, not for the learning representation (that's skip/Δ alone).

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

### Root cause: the recorded action is the *measured next qpos*, not the command
Confirmed in `trajectory_data_collector.py` (line 495: "action: next state … gripper state"). So the gripper action during a grasp is the *physical* jaw qpos, which stalls on the rigid block at ≈ `-(pinch_half_width)` (e.g. −0.0105 for a 24 mm block). Consequences:
  * The action **varies with block size**, so the policy must regress block half-width from pixels to know how far to close — that IS the `close_shallow` failure, a task with zero payoff (the kinematic lock, and a real current-limited gripper, make "just close all the way" safe on any block).
  * Changing only the *collection command* to 0 is a **no-op on the data** — the jaws still stall at the block, so the recorded action is unchanged. The width-regression is baked into the recorded *state*, so the load-bearing fix must be at the **transform** (relabel the action), not in collection.
  * Secondary: the close is a 50-step interpolated ramp (`_interpolate_gripper`) recorded per mj-step, so the action passes through mid-values → the policy emits *partial* closes and wobbles across the open/close boundary → weak grips + premature drops + jaw pulsing.

### Two fixes (different jobs)
| change | fixes | required? |
|---|---|---|
| **Transform: binarize action → {−0.014 open, 0 closed}** (`--binarize-gripper` in `transform_skip_dataset`; threshold e.g. > −0.013 → 0, else −0.014) | width-regression + ramp mid-values (`close_shallow`) | **yes, load-bearing** |
| **Collection: command 0 instead of −0.0105** (`get_object_grasp_gripper_pos → 0.0`) | residual force mismatch + real-grip realism | optional, nearly free — do it since we're re-collecting |

  * **State gripper channel: leave untouched.** Its physical value is a useful proprioceptive cue — open −0.014, holding a block ≈ −0.0095/−0.011/−0.012 (the three graspable sizes 19/22/24 mm), closed-on-nothing ≈ 0 (a miss). The policy may use it as input; it just never has to reproduce it as an action. **Width-invariance is the point:** one "close" command generalizes across every block size.
  * **Deploy consistency:** policy outputs closed = 0 → env commands 0 → jaws stall on the block at ≈ −0.0105 → next state ≈ −0.0105, matching the training state. The action(command)/state(result) "mismatch" is correct — that's what a command channel is.

### On the "binarize-only" mismatch (why we also command 0 at collection)
Binarize-only would relabel the action to 0 while the demo physics ran a −0.0105 command. The divergence is confined to **post-contact grip force during the ~1–2 steps before the lock engages**: identical images/dynamics up to contact (the ramp to −0.0105 is the same), qpos stalls at −0.0105 either way (invisible in state), and the lock welds the block right after — so it's negligible for what the policy sees, and in the *helpful* direction for real transfer (full-close = secure friction grip). Commanding 0 at collection (keeping the ramp) removes even that residue for ~free: contact still happens gently at the same point; only post-contact force rises against the stall.

### Predicted, testable effect
`close_shallow` is defined in `metrics.py` as `close_cmd < -(pinch_half_width + tol)` — "commanded a gripper still too open." With the command pinned to 0, that can **never** fire → `close_shallow` should collapse to ~0 by construction. The real question the next eval answers: do those attempts convert to *engaged locks*? They should (a full close reaches the block and produces the two-sided pinch contact the gate needs). If grasp rate jumps and `close_shallow` vanishes together, the mechanism is confirmed.

### Footnotes (non-issues here)
  * **"Thin block" degeneracy doesn't apply.** It would only bite if a graspable block's hold-state (~0) collided with closed-on-nothing — but the smallest graspable is 19 mm (−0.0095), well separated. Vision also disambiguates hold-vs-miss.
  * **Real-hardware caveat (deploy, not data):** commanding full-close 0 on a rigid block would stall a *position*-controlled gripper servo at max current. Fine in sim (lock) and on a current/torque-limited real gripper — confirm the real AR4 gripper is compliant / current-limited before deploy. Add to the sim2real checklist.
  * **Minor:** `_interpolate_gripper` won't early-converge when target 0 can't be reached with a block in the jaws (already anticipated by the TODO at `ar4_mk3_robot_interface.py:373`); harmless, just records a few more stalled frames that skip/static-filter drop.
  * Worth a 2-minute sim check that a full-close command doesn't shove/flip the block harder in the pre-lock window than the gentle close does (analysis says no — contact dynamics identical up to the stall, extra force is post-lock).

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
