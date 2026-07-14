# Training journal — 2026-07-06

## Run

- Run name: `pi05_ar4_mk3_2026-07-04_16-51-56` (mlflow run `f8cd833f3a4a434fb323f2012113016c`, snapshot in `mlflow.db` alongside this note)
- Config: `pi05_ar4_mk3` — pi0.5 full finetune, action_horizon 10, batch 32, effectively flat LR 5e-5 (`peak_lr == decay_lr`), EMA 0.999, planned 300k steps
- Dataset: `Purple69/aera_semi_pnp_dr_16_06_2026_skip3_delta_no_go_home_no_static_smoothed_v2`
- Eval: decoupled eval worker, 20 episodes per checkpoint, kinematic grasp lock on

## Eval funnel by checkpoint

| step | reached | grasped | transported | success | grasp_drop_rate |
|------|---------|---------|-------------|---------|-----------------|
| 10k  | 0.90 | 0.75 | 0.15 | 0.25 | 0.60 |
| 20k  | 0.90 | 0.65 | 0.30 | 0.35 | 0.39 |
| 30k  | 0.95 | 0.80 | 0.30 | 0.40 | 0.38 |
| 40k  | 0.80 | 0.75 | 0.30 | 0.20 | 0.67 |
| 50k  | 0.90 | 0.90 | 0.55 | 0.35 | 0.50 |
| 60k  | 0.85 | 0.60 | 0.45 | 0.50 | 0.33 |
| 70k  | 0.90 | 0.70 | 0.40 | 0.45 | 0.36 |


## Manual Eval replan_steps=10 n_substeps=3 max_episode_steps=1000

This observations were made with suboptimal replan/substeps setup. Keeping it here for completnes. Below we have numbers for other configurations that show better performance.

### checkpoint 50k
Never works in manual tests. Arm approaches fine but then it tries to grasp a bit off (like in front and next to the block) and obviously keeps failing to grasp and lift. It goes down tries to graps, fails goes a bit up and tries again and so on, usually the longer it goes the worse it gets. Sometimes it manages an awkward grasp (not nicely enclosed, but half levitating with some gaps between object and jaws) but usually drops it. This won't be fixed changing the fully closed to 0, the position of gripper is not precise enough. Even a few it did grasp, it didn't drop it off.

Works better with DR off. Sometimes it grasp okish but then randomly drops it, it look a bit like an artifact of the recovery/partial grasp (policy seems to learn to just drop???), have to revisit how this is implemented/perhaps do not use this feature. Maybe it's similar story with wrong approach perturbation? It learns to go approach a bit off?
I'm thinking, since the training just sees random step in trajectory (TODO confirm that is really just gets current state, not a N recent steps), it doesn't actually learn the intended "fix your approach"/"fix your grasp", it learns to approach awkwardly and randomly drop. If you think about, policy will learn to pick up after drop, simply because it is just trained to pick up when block is on the table, so when it slips, policy will naturally understand that now is time to pick it up.

Overall, useless policy.

### checkpoint 40k
Worked 1 out of 10 tries with DR. In general symptoms are similar like in 50k, but approach is a bit better (it sill doesn't puts the jaws around object, but less frequently). I also see sometimes the jaws pulsating back and forth when in grasping pose, like the arm cannot properly close them and then tries to open and try again (here maybe the full closure would help). 

Doesn't seem to be any better with DR off.

Overall, useless policy.

### checkpoint 30k
Seems a bit better overall, but not much, it was able to grasp few more times (not super precisely and still with the visible gaps) but drops prematurely. Seems to behave better with DR off.

Overall, useless policy.

### checkpoint 20k
Seems again slightly better than 30k. Positioning around the object is a bit better, it grasped few times, be the grip was rather bad (gaps, etc.). Overall not much difference from 30k, maybe less often approach is off to the side/in front (but more often jaws press on the object). 

Overall, useless policy.

### checkpoint 60k
This is peculiar, but I found 60k to be doing better, probably the best checkpoint so far. In DR off it actually grasped in all 5 attempts (crapy grasps mostly, but it's something), it succeeded once. The problem with arm going to much to the side or in front seems to be much less prevalent here. It actually succeeded 2/10 in DR also. But still many of previously described failure modes are present.

Overall, useless policy. But can sometimes do the job, which is better than the other ones.

### checkpoint 70k
Nothing new, but seems worse than 60k (no success with DR nor without DR)


### Additional details about manual tests:
  * It doesn't use the same seed as eval done during training, but the same seeds between checkpoint.
  * Seeds 1-10 were tested with DR and seed 1-5 without DR
  * Eval numbers are most likely skewed upwards because 2 out of 20 start with object already on target (so real success ~= success -0.1)
  * Eval command `uv run aera/autonomous/openpi/scripts/run_policy_on_env.py --replan_steps=10 --n_substeps=3 --max_episode_steps=1000 --seed=<seed> --domain_rand`
  * Server start-up command `uv run aera/autonomous/openpi/scripts/serve_policy.py checkpoint:checkpoint --checkpoint.config=pi05_ar4_mk3 --checkpoint.dir=checkpoints/pi05_ar4_mk3/pi05_ar4_mk3_2026-07-04_16-51-56/70000`

### Summary
  * It seems weird that, I was not able to match even the modest success of eval during training. I think first order of business is to figure out this one to eliminate all errors coming from some training, on training eval, and manual eval. In particular it seems that checkpoint 20k or 60k is the best, but this is not reflected by eval stats.
  * With DR off, the results are better. Not surprising, DR introduced a lot of visual noise, shadows, etc. With DR off, there is just few object and colors of objects and target are very easily distinguishable from background.
  * We should have automated evals that would be able to find more detailed failure patters. Manually testing a describing these as above is not feasible. Even for the same seed with have different behaviours, so running just once per seed is not enough. The manual test I did probably have huge variance/
  * Taking all these into account, I think the description of behaviour per checkpoint can be highly misleading for the actual performance. Because of 1) variance 2) My bias (it just get tiring) 3) My failure to put in words 3d arm movement and all the failure modes (there are some that I omitted, e.g. happens often that jaws push on the object (pushing it into the table), this would be catastrophic in real execution.)
  * Some seeds may be genuinely hard, because color of object and some other elements in viz are very similar.

## Multi-seed eval-variance run (first pass)

Script: `aera/autonomous/openpi/scripts/eval_variance.py`

Parameters:
  * `--config pi05_ar4_mk3`
  * `--n-dr-seeds 10 --n-seeds 6 --k-repeats 3` (seeds 0-9 with domain_rand on, seeds 0-5 with domain_rand off, each repeated 3x with an identical reset seed)
  * `--n-substeps 3 --replan-steps 10 --max-episode-steps 1000 --kinematic-grasp`
  * prompt: `pick the yellow block and place it on the red target`
  * Checkpoints: `checkpoints/pi05_ar4_mk3/pi05_ar4_mk3_2026-07-04_16-51-56/{20000,30000,40000,50000,60000,70000}` (10k not evaluated, not present locally)
  * Output: `eval_results/pi05_ar4_mk3_2026-07-04_16-51-56/<step>/{summary.json,episodes.jsonl}`

### Funnel rates

| step | mode | n | reached | grasped | lifted | transported | placed |
|------|------|---|---------|---------|--------|--------------|--------|
| 20k | DR   | 30 | 0.73 | 0.40 | 0.40 | 0.03 | 0.03 |
| 20k | noDR | 18 | 0.67 | 0.61 | 0.50 | 0.06 | 0.00 |
| 20k | all  | 48 | 0.71 | 0.48 | 0.44 | 0.04 | 0.02 |
| 30k | DR   | 30 | 0.73 | 0.47 | 0.40 | 0.07 | 0.07 |
| 30k | noDR | 18 | 1.00 | 0.94 | 0.94 | 0.17 | 0.06 |
| 30k | all  | 48 | 0.83 | 0.65 | 0.60 | 0.10 | 0.06 |
| 40k | DR   | 30 | 0.67 | 0.43 | 0.47 | 0.07 | 0.03 |
| 40k | noDR | 18 | 0.61 | 0.50 | 0.50 | 0.11 | 0.00 |
| 40k | all  | 48 | 0.65 | 0.46 | 0.48 | 0.08 | 0.02 |
| 50k | DR   | 30 | 0.73 | 0.33 | 0.37 | 0.03 | 0.03 |
| 50k | noDR | 18 | 0.89 | 0.78 | 0.78 | 0.06 | 0.00 |
| 50k | all  | 48 | 0.79 | 0.50 | 0.52 | 0.04 | 0.02 |
| 60k | DR   | 30 | 0.80 | 0.53 | 0.53 | 0.13 | 0.10 |
| 60k | noDR | 18 | 0.61 | 0.61 | 0.56 | 0.11 | 0.11 |
| 60k | all  | 48 | 0.73 | 0.56 | 0.54 | 0.12 | 0.10 |
| 70k | DR   | 30 | 0.77 | 0.40 | 0.37 | 0.03 | 0.03 |
| 70k | noDR | 18 | 0.67 | 0.56 | 0.44 | 0.06 | 0.00 |
| 70k | all  | 48 | 0.73 | 0.46 | 0.40 | 0.04 | 0.02 |

### Between-seed std / within-seed std (mean), per stage

| step | mode | reached | grasped | lifted | transported | placed |
|------|------|---------|---------|--------|--------------|--------|
| 20k | DR   | 0.33 / 0.19 | 0.33 / 0.28 | 0.33 / 0.28 | 0.10 / 0.05 | 0.10 / 0.05 |
| 20k | noDR | 0.38 / 0.16 | 0.36 / 0.24 | 0.37 / 0.24 | 0.12 / 0.08 | 0.00 / 0.00 |
| 20k | all  | 0.35 / 0.18 | 0.35 / 0.27 | 0.35 / 0.27 | 0.11 / 0.06 | 0.08 / 0.03 |
| 30k | DR   | 0.39 / 0.09 | 0.45 / 0.09 | 0.39 / 0.19 | 0.13 / 0.09 | 0.13 / 0.09 |
| 30k | noDR | 0.00 / 0.00 | 0.12 / 0.08 | 0.12 / 0.08 | 0.25 / 0.16 | 0.12 / 0.08 |
| 30k | all  | 0.33 / 0.06 | 0.43 / 0.09 | 0.41 / 0.15 | 0.19 / 0.12 | 0.13 / 0.09 |
| 40k | DR   | 0.39 / 0.14 | 0.40 / 0.19 | 0.40 / 0.19 | 0.13 / 0.09 | 0.10 / 0.05 |
| 40k | noDR | 0.30 / 0.31 | 0.37 / 0.24 | 0.37 / 0.24 | 0.25 / 0.08 | 0.00 / 0.00 |
| 40k | all  | 0.36 / 0.21 | 0.39 / 0.21 | 0.39 / 0.21 | 0.19 / 0.09 | 0.08 / 0.03 |
| 50k | DR   | 0.42 / 0.05 | 0.37 / 0.19 | 0.38 / 0.19 | 0.10 / 0.05 | 0.10 / 0.05 |
| 50k | noDR | 0.16 / 0.16 | 0.25 / 0.24 | 0.25 / 0.24 | 0.12 / 0.08 | 0.00 / 0.00 |
| 50k | all  | 0.35 / 0.09 | 0.39 / 0.21 | 0.39 / 0.21 | 0.11 / 0.06 | 0.08 / 0.03 |
| 60k | DR   | 0.34 / 0.09 | 0.45 / 0.09 | 0.45 / 0.09 | 0.31 / 0.05 | 0.30 / 0.00 |
| 60k | noDR | 0.45 / 0.08 | 0.45 / 0.08 | 0.42 / 0.16 | 0.25 / 0.08 | 0.25 / 0.08 |
| 60k | all  | 0.39 / 0.09 | 0.45 / 0.09 | 0.44 / 0.12 | 0.29 / 0.06 | 0.28 / 0.03 |
| 70k | DR   | 0.33 / 0.14 | 0.39 / 0.19 | 0.38 / 0.19 | 0.10 / 0.05 | 0.10 / 0.05 |
| 70k | noDR | 0.38 / 0.16 | 0.31 / 0.31 | 0.37 / 0.24 | 0.12 / 0.08 | 0.00 / 0.00 |
| 70k | all  | 0.36 / 0.15 | 0.37 / 0.24 | 0.38 / 0.21 | 0.11 / 0.06 | 0.08 / 0.03 |

## Ablation: exact original training rollout args, new suite

Script: `aera/autonomous/openpi/scripts/eval_variance.py`

Parameters:
  * `--config pi05_ar4_mk3`
  * Suite shape (defaults, unchanged): DR seeds `1000-1014` (15 seeds), no-DR seeds `1000-1009` (10 seeds), `k_repeats=2`
  * `--n-substeps 20 --replan-steps 5 --max-episode-steps 400` (exact original training-eval rollout args, overriding the suite's own defaults of 3/10/1000)
  * `--kinematic-grasp`
  * prompt: `pick the yellow block and place it on the red target`
  * Checkpoints: `checkpoints/pi05_ar4_mk3/pi05_ar4_mk3_2026-07-04_16-51-56/{20000,30000,40000,50000,60000,70000}`
  * Output: `eval_results/pi05_ar4_mk3_2026-07-04_16-51-56_trainargs/<step>/{summary.json,episodes.jsonl}`

### Funnel rates (overall / DR / noDR)

| step | mode | n | reached | grasped | lifted | transported | placed |
|------|------|---|---------|---------|--------|--------------|--------|
| 20k | overall | 50 | 0.76 | 0.58 | 0.56 | 0.30 | 0.28 |
| 20k | DR      | 30 | 0.83 | 0.60 | 0.60 | 0.30 | 0.27 |
| 20k | noDR    | 20 | 0.65 | 0.55 | 0.50 | 0.30 | 0.30 |
| 30k | overall | 50 | 0.86 | 0.72 | 0.58 | 0.44 | 0.54 |
| 30k | DR      | 30 | 0.87 | 0.67 | 0.60 | 0.47 | 0.53 |
| 30k | noDR    | 20 | 0.85 | 0.80 | 0.55 | 0.40 | 0.55 |
| 40k | overall | 50 | 0.86 | 0.76 | 0.66 | 0.44 | 0.60 |
| 40k | DR      | 30 | 0.90 | 0.80 | 0.73 | 0.53 | 0.70 |
| 40k | noDR    | 20 | 0.80 | 0.70 | 0.55 | 0.30 | 0.45 |
| 50k | overall | 50 | 0.86 | 0.82 | 0.74 | 0.48 | 0.58 |
| 50k | DR      | 30 | 0.83 | 0.77 | 0.70 | 0.57 | 0.70 |
| 50k | noDR    | 20 | 0.90 | 0.90 | 0.80 | 0.35 | 0.40 |
| 60k | overall | 50 | 0.80 | 0.64 | 0.60 | 0.44 | 0.48 |
| 60k | DR      | 30 | 0.83 | 0.63 | 0.63 | 0.40 | 0.43 |
| 60k | noDR    | 20 | 0.75 | 0.65 | 0.55 | 0.50 | 0.55 |
| 70k | overall | 50 | 0.80 | 0.74 | 0.66 | 0.54 | 0.64 |
| 70k | DR      | 30 | 0.93 | 0.87 | 0.83 | 0.70 | 0.77 |
| 70k | noDR    | 20 | 0.60 | 0.55 | 0.40 | 0.30 | 0.45 |

### Between-seed std / within-seed std (mean), per stage (DR / noDR only — overall has no seed-group decomposition)

| step | mode | reached | grasped | lifted | transported | placed |
|------|------|---------|---------|--------|--------------|--------|
| 20k | DR   | 0.30 / 0.10 | 0.37 / 0.20 | 0.37 / 0.20 | 0.36 / 0.17 | 0.36 / 0.13 |
| 20k | noDR | 0.39 / 0.15 | 0.35 / 0.25 | 0.39 / 0.20 | 0.33 / 0.20 | 0.33 / 0.20 |
| 30k | DR   | 0.34 / 0.00 | 0.39 / 0.13 | 0.42 / 0.13 | 0.39 / 0.20 | 0.39 / 0.20 |
| 30k | noDR | 0.32 / 0.05 | 0.33 / 0.10 | 0.42 / 0.15 | 0.30 / 0.30 | 0.35 / 0.25 |
| 40k | DR   | 0.27 / 0.03 | 0.36 / 0.07 | 0.40 / 0.07 | 0.46 / 0.07 | 0.40 / 0.10 |
| 40k | noDR | 0.33 / 0.10 | 0.33 / 0.20 | 0.35 / 0.25 | 0.33 / 0.20 | 0.35 / 0.25 |
| 50k | DR   | 0.35 / 0.03 | 0.36 / 0.10 | 0.40 / 0.10 | 0.36 / 0.23 | 0.31 / 0.23 |
| 50k | noDR | 0.30 / 0.00 | 0.30 / 0.00 | 0.33 / 0.10 | 0.32 / 0.25 | 0.37 / 0.20 |
| 60k | DR   | 0.30 / 0.10 | 0.39 / 0.17 | 0.39 / 0.17 | 0.33 / 0.27 | 0.31 / 0.30 |
| 60k | noDR | 0.40 / 0.05 | 0.39 / 0.15 | 0.47 / 0.05 | 0.45 / 0.10 | 0.47 / 0.05 |
| 70k | DR   | 0.25 / 0.00 | 0.34 / 0.00 | 0.35 / 0.03 | 0.36 / 0.17 | 0.31 / 0.17 |
| 70k | noDR | 0.37 / 0.20 | 0.35 / 0.25 | 0.37 / 0.20 | 0.33 / 0.20 | 0.35 / 0.25 |

### Failure-mode counts (overall / DR / noDR, out of n episodes above)

| step | mode | success | never_reached | no_grasp_attempt | grasp_missed | grasped_not_lifted | dropped_early | dropped_or_missed_at_goal | timeout_holding |
|------|------|---------|---------------|------------------|--------------|---------------------|----------------|-----------------------------|------------------|
| 20k | overall | 14 | 8 | 0 | 8 | 0 | 13 | 4 | 3 |
| 20k | DR      | 8  | 3 | 0 | 6 | 0 | 9  | 2 | 2 |
| 20k | noDR    | 6  | 5 | 0 | 2 | 0 | 4  | 2 | 1 |
| 30k | overall | 27 | 3 | 1 | 6 | 2 | 5  | 2 | 4 |
| 30k | DR      | 16 | 2 | 1 | 5 | 0 | 2  | 1 | 3 |
| 30k | noDR    | 11 | 1 | 0 | 1 | 2 | 3  | 1 | 1 |
| 40k | overall | 30 | 3 | 0 | 3 | 2 | 8  | 0 | 4 |
| 40k | DR      | 21 | 1 | 0 | 1 | 0 | 4  | 0 | 3 |
| 40k | noDR    | 9  | 2 | 0 | 2 | 2 | 4  | 0 | 1 |
| 50k | overall | 29 | 3 | 0 | 1 | 0 | 10 | 3 | 4 |
| 50k | DR      | 21 | 3 | 0 | 1 | 0 | 3  | 1 | 1 |
| 50k | noDR    | 8  | 0 | 0 | 0 | 0 | 7  | 2 | 3 |
| 60k | overall | 24 | 6 | 1 | 6 | 2 | 5  | 2 | 4 |
| 60k | DR      | 13 | 3 | 1 | 4 | 0 | 4  | 1 | 4 |
| 60k | noDR    | 11 | 3 | 0 | 2 | 2 | 1  | 1 | 0 |
| 70k | overall | 32 | 6 | 0 | 1 | 2 | 5  | 3 | 1 |
| 70k | DR      | 23 | 0 | 0 | 0 | 0 | 3  | 3 | 1 |
| 70k | noDR    | 9  | 6 | 0 | 1 | 2 | 2  | 0 | 0 |

### Missed-grasp anatomy + other diagnostics (overall, i.e. all 50 episodes pooled)

| step | coarse_far | pinch | finger | height | close_shallow | unknown | grasp_attempts_mean | failed_grasp_attempts_mean | grasp_attempt_success_rate | premature_release_count_mean | press/episode_rate | pushed_dist_pre_grasp_mean | gripper_close_cycles_mean |
|------|------------|-------|--------|--------|----------------|---------|----------------------|-------------------------------|------------------------------|-------------------------------|----------------------|------------------------------|------------------------------|
| 20k | 0.12 | 0.16 | 0.38 | 0.29 | 0.72 | 0.00 | 3.28 | 2.22 | 0.32 | 0.72 | 0.62 | 0.0092 | 3.52 |
| 30k | 0.10 | 0.22 | 0.16 | 0.11 | 0.83 | 0.00 | 2.48 | 1.26 | 0.49 | 0.64 | 0.64 | 0.0150 | 2.98 |
| 40k | 0.20 | 0.16 | 0.17 | 0.19 | 0.68 | 0.00 | 3.06 | 1.38 | 0.55 | 1.08 | 0.70 | 0.0059 | 3.12 |
| 50k | 0.09 | 0.19 | 0.34 | 0.22 | 0.58 | 0.00 | 2.88 | 1.34 | 0.53 | 0.84 | 0.62 | 0.0114 | 2.84 |
| 60k | 0.11 | 0.14 | 0.21 | 0.07 | 0.84 | 0.00 | 2.20 | 1.12 | 0.49 | 0.52 | 0.60 | 0.0175 | 2.10 |
| 70k | 0.10 | 0.13 | 0.18 | 0.12 | 0.85 | 0.00 | 3.84 | 1.20 | 0.69 | 1.98 | 0.60 | 0.0045 | 3.82 |

Column meanings:
  * `coarse_far` / `pinch` / `finger` / `height` / `close_shallow` / `unknown` — rate (of failed grasp attempts) tagged with each miss reason; not exclusive, one attempt can have multiple reasons
  * `grasp_attempts_mean` — avg. close commands issued near the block per episode
  * `failed_grasp_attempts_mean` — avg. of those that didn't engage the lock
  * `grasp_attempt_success_rate` — fraction of attempts (not episodes) that engaged
  * `premature_release_count_mean` — avg. drops before success per episode
  * `press/episode_rate` — fraction of episodes where a jaw pressed the block into the table (>3x its weight) while nothing was held
  * `pushed_dist_pre_grasp_mean` — avg. distance (m) the block was dragged/shoved before ever being grasped
  * `gripper_close_cycles_mean` — avg. open→close command cycles per episode (retry loops / jaw pulsing)


## Ablation: replan_steps x n_substeps grid, checkpoint 70k

Script: `aera/autonomous/openpi/scripts/eval_variance.py`

Parameters:
  * `--config pi05_ar4_mk3 --checkpoint-dir checkpoints/pi05_ar4_mk3/pi05_ar4_mk3_2026-07-04_16-51-56/70000`
  * Suite shape (defaults, unchanged): DR seeds `1000-1014` (15), no-DR seeds `1000-1009` (10), `k_repeats=2` (50 episodes/combo)
  * Grid: `--replan-steps` in `{1,2,3,4,5}` x `--n-substeps`/`--max-episode-steps` pairs `{(3,800),(10,500),(20,300)}` (paired so total sim-time per n_substeps setting is roughly comparable), 15 combos total
  * `--kinematic-grasp`, prompt: `pick the yellow block and place it on the red target`
  * Output: `eval_results/pi05_ar4_mk3_2026-07-04_16-51-56_ablation_70k/nsub{N}_max{M}_replan{R}/70000/{summary.json,episodes.jsonl}`

### Funnel rates (overall / DR / noDR)

| nsub | max | replan | mode | n | reached | grasped | lifted | transported | placed |
|------|-----|--------|------|---|---------|---------|--------|--------------|--------|
| 3 | 800 | 1 | overall | 50 | 0.88 | 0.62 | 0.22 | 0.10 | 0.20 |
| 3 | 800 | 1 | DR | 30 | 0.87 | 0.67 | 0.37 | 0.17 | 0.20 |
| 3 | 800 | 1 | noDR | 20 | 0.90 | 0.55 | 0.00 | 0.00 | 0.20 |
| 3 | 800 | 2 | overall | 50 | 0.90 | 0.72 | 0.28 | 0.12 | 0.20 |
| 3 | 800 | 2 | DR | 30 | 0.90 | 0.70 | 0.43 | 0.20 | 0.27 |
| 3 | 800 | 2 | noDR | 20 | 0.90 | 0.75 | 0.05 | 0.00 | 0.10 |
| 3 | 800 | 3 | overall | 50 | 0.86 | 0.68 | 0.42 | 0.18 | 0.26 |
| 3 | 800 | 3 | DR | 30 | 0.93 | 0.67 | 0.60 | 0.30 | 0.30 |
| 3 | 800 | 3 | noDR | 20 | 0.75 | 0.70 | 0.15 | 0.00 | 0.20 |
| 3 | 800 | 4 | overall | 50 | 0.82 | 0.74 | 0.58 | 0.20 | 0.40 |
| 3 | 800 | 4 | DR | 30 | 0.83 | 0.73 | 0.67 | 0.27 | 0.40 |
| 3 | 800 | 4 | noDR | 20 | 0.80 | 0.75 | 0.45 | 0.10 | 0.40 |
| 3 | 800 | 5 | overall | 50 | 0.84 | 0.74 | 0.56 | 0.24 | 0.38 |
| 3 | 800 | 5 | DR | 30 | 0.87 | 0.73 | 0.60 | 0.27 | 0.33 |
| 3 | 800 | 5 | noDR | 20 | 0.80 | 0.75 | 0.50 | 0.20 | 0.45 |
| 10 | 500 | 1 | overall | 50 | 0.92 | 0.70 | 0.30 | 0.18 | 0.28 |
| 10 | 500 | 1 | DR | 30 | 0.93 | 0.73 | 0.50 | 0.30 | 0.37 |
| 10 | 500 | 1 | noDR | 20 | 0.90 | 0.65 | 0.00 | 0.00 | 0.15 |
| 10 | 500 | 2 | overall | 50 | 0.92 | 0.78 | 0.40 | 0.24 | 0.38 |
| 10 | 500 | 2 | DR | 30 | 0.93 | 0.80 | 0.67 | 0.40 | 0.53 |
| 10 | 500 | 2 | noDR | 20 | 0.90 | 0.75 | 0.00 | 0.00 | 0.15 |
| 10 | 500 | 3 | overall | 50 | 0.88 | 0.72 | 0.38 | 0.24 | 0.36 |
| 10 | 500 | 3 | DR | 30 | 0.90 | 0.67 | 0.53 | 0.30 | 0.43 |
| 10 | 500 | 3 | noDR | 20 | 0.85 | 0.80 | 0.15 | 0.15 | 0.25 |
| 10 | 500 | 4 | overall | 50 | 0.84 | 0.78 | 0.60 | 0.44 | 0.56 |
| 10 | 500 | 4 | DR | 30 | 0.87 | 0.77 | 0.70 | 0.53 | 0.57 |
| 10 | 500 | 4 | noDR | 20 | 0.80 | 0.80 | 0.45 | 0.30 | 0.55 |
| 10 | 500 | 5 | overall | 50 | 0.82 | 0.74 | 0.66 | 0.38 | 0.52 |
| 10 | 500 | 5 | DR | 30 | 0.87 | 0.80 | 0.73 | 0.47 | 0.57 |
| 10 | 500 | 5 | noDR | 20 | 0.75 | 0.65 | 0.55 | 0.25 | 0.45 |
| 20 | 300 | 1 | overall | 50 | 0.92 | 0.70 | 0.30 | 0.20 | 0.32 |
| 20 | 300 | 1 | DR | 30 | 0.93 | 0.73 | 0.47 | 0.33 | 0.47 |
| 20 | 300 | 1 | noDR | 20 | 0.90 | 0.65 | 0.05 | 0.00 | 0.10 |
| 20 | 300 | 2 | overall | 50 | 0.90 | 0.70 | 0.44 | 0.34 | 0.38 |
| 20 | 300 | 2 | DR | 30 | 0.93 | 0.80 | 0.73 | 0.57 | 0.57 |
| 20 | 300 | 2 | noDR | 20 | 0.85 | 0.55 | 0.00 | 0.00 | 0.10 |
| 20 | 300 | 3 | overall | 50 | 0.88 | 0.78 | 0.58 | 0.34 | 0.42 |
| 20 | 300 | 3 | DR | 30 | 0.90 | 0.80 | 0.73 | 0.43 | 0.50 |
| 20 | 300 | 3 | noDR | 20 | 0.85 | 0.75 | 0.35 | 0.20 | 0.30 |
| 20 | 300 | 4 | overall | 50 | 0.90 | 0.82 | 0.66 | 0.52 | 0.60 |
| 20 | 300 | 4 | DR | 30 | 0.90 | 0.83 | 0.73 | 0.57 | 0.60 |
| 20 | 300 | 4 | noDR | 20 | 0.90 | 0.80 | 0.55 | 0.45 | 0.60 |
| 20 | 300 | 5 | overall | 50 | 0.80 | 0.62 | 0.54 | 0.36 | 0.42 |
| 20 | 300 | 5 | DR | 30 | 0.80 | 0.67 | 0.60 | 0.40 | 0.50 |
| 20 | 300 | 5 | noDR | 20 | 0.80 | 0.55 | 0.45 | 0.30 | 0.30 |

### Between-seed std / within-seed std (mean), per stage (DR / noDR only)

| nsub | max | replan | mode | reached | grasped | lifted | transported | placed |
|------|-----|--------|------|---------|---------|--------|--------------|--------|
| 3 | 800 | 1 | DR | 0.29 / 0.07 | 0.43 / 0.07 | 0.43 / 0.10 | 0.35 / 0.03 | 0.40 / 0.00 |
| 3 | 800 | 1 | noDR | 0.30 / 0.00 | 0.42 / 0.15 | 0.00 / 0.00 | 0.00 / 0.00 | 0.40 / 0.00 |
| 3 | 800 | 2 | DR | 0.27 / 0.03 | 0.40 / 0.10 | 0.44 / 0.10 | 0.36 / 0.07 | 0.40 / 0.07 |
| 3 | 800 | 2 | noDR | 0.30 / 0.00 | 0.34 / 0.15 | 0.15 / 0.05 | 0.00 / 0.00 | 0.30 / 0.00 |
| 3 | 800 | 3 | DR | 0.25 / 0.00 | 0.39 / 0.13 | 0.42 / 0.13 | 0.36 / 0.17 | 0.31 / 0.23 |
| 3 | 800 | 3 | noDR | 0.40 / 0.05 | 0.46 / 0.00 | 0.32 / 0.05 | 0.00 / 0.00 | 0.40 / 0.00 |
| 3 | 800 | 4 | DR | 0.35 / 0.03 | 0.40 / 0.07 | 0.43 / 0.07 | 0.36 / 0.13 | 0.42 / 0.13 |
| 3 | 800 | 4 | noDR | 0.40 / 0.00 | 0.40 / 0.05 | 0.47 / 0.05 | 0.30 / 0.00 | 0.49 / 0.00 |
| 3 | 800 | 5 | DR | 0.29 / 0.07 | 0.40 / 0.07 | 0.42 / 0.13 | 0.40 / 0.07 | 0.39 / 0.13 |
| 3 | 800 | 5 | noDR | 0.40 / 0.00 | 0.40 / 0.05 | 0.45 / 0.10 | 0.33 / 0.10 | 0.47 / 0.05 |
| 10 | 500 | 1 | DR | 0.25 / 0.00 | 0.36 / 0.13 | 0.45 / 0.10 | 0.40 / 0.10 | 0.39 / 0.17 |
| 10 | 500 | 1 | noDR | 0.30 / 0.00 | 0.45 / 0.05 | 0.00 / 0.00 | 0.00 / 0.00 | 0.32 / 0.05 |
| 10 | 500 | 2 | DR | 0.25 / 0.00 | 0.40 / 0.00 | 0.43 / 0.07 | 0.42 / 0.13 | 0.43 / 0.13 |
| 10 | 500 | 2 | noDR | 0.30 / 0.00 | 0.40 / 0.05 | 0.00 / 0.00 | 0.00 / 0.00 | 0.32 / 0.05 |
| 10 | 500 | 3 | DR | 0.27 / 0.03 | 0.43 / 0.07 | 0.43 / 0.13 | 0.36 / 0.17 | 0.40 / 0.17 |
| 10 | 500 | 3 | noDR | 0.32 / 0.05 | 0.33 / 0.10 | 0.23 / 0.15 | 0.23 / 0.15 | 0.40 / 0.05 |
| 10 | 500 | 4 | DR | 0.34 / 0.00 | 0.36 / 0.10 | 0.40 / 0.10 | 0.43 / 0.13 | 0.36 / 0.23 |
| 10 | 500 | 4 | noDR | 0.40 / 0.00 | 0.40 / 0.00 | 0.42 / 0.15 | 0.33 / 0.20 | 0.42 / 0.15 |
| 10 | 500 | 5 | DR | 0.29 / 0.07 | 0.36 / 0.07 | 0.40 / 0.07 | 0.43 / 0.13 | 0.40 / 0.17 |
| 10 | 500 | 5 | noDR | 0.34 / 0.15 | 0.32 / 0.25 | 0.42 / 0.15 | 0.40 / 0.05 | 0.42 / 0.15 |
| 20 | 300 | 1 | DR | 0.25 / 0.00 | 0.40 / 0.07 | 0.39 / 0.20 | 0.39 / 0.13 | 0.43 / 0.13 |
| 20 | 300 | 1 | noDR | 0.30 / 0.00 | 0.32 / 0.25 | 0.15 / 0.05 | 0.00 / 0.00 | 0.30 / 0.00 |
| 20 | 300 | 2 | DR | 0.25 / 0.00 | 0.31 / 0.13 | 0.40 / 0.07 | 0.40 / 0.17 | 0.36 / 0.23 |
| 20 | 300 | 2 | noDR | 0.32 / 0.05 | 0.35 / 0.25 | 0.00 / 0.00 | 0.00 / 0.00 | 0.30 / 0.00 |
| 20 | 300 | 3 | DR | 0.27 / 0.03 | 0.36 / 0.07 | 0.40 / 0.07 | 0.44 / 0.10 | 0.41 / 0.17 |
| 20 | 300 | 3 | noDR | 0.32 / 0.05 | 0.34 / 0.15 | 0.32 / 0.25 | 0.24 / 0.20 | 0.33 / 0.20 |
| 20 | 300 | 4 | DR | 0.27 / 0.03 | 0.30 / 0.10 | 0.36 / 0.13 | 0.44 / 0.10 | 0.37 / 0.20 |
| 20 | 300 | 4 | noDR | 0.30 / 0.00 | 0.33 / 0.10 | 0.35 / 0.25 | 0.42 / 0.15 | 0.44 / 0.10 |
| 20 | 300 | 5 | DR | 0.36 / 0.07 | 0.35 / 0.20 | 0.37 / 0.20 | 0.42 / 0.13 | 0.41 / 0.17 |
| 20 | 300 | 5 | noDR | 0.33 / 0.10 | 0.42 / 0.15 | 0.35 / 0.25 | 0.33 / 0.20 | 0.46 / 0.00 |

### Failure-mode counts (overall / DR / noDR, out of n episodes above)

| nsub | max | replan | mode | success | never_reached | no_grasp_attempt | grasp_missed | wrong_object_grasp | grasped_not_lifted | dropped_early | dropped_or_missed_at_goal | timeout_holding |
|------|-----|--------|------|----|----|----|----|----|----|----|----|----|
| 3 | 800 | 1 | overall | 10 | 2 | 3 | 8 | 0 | 7 | 1 | 0 | 19 |
| 3 | 800 | 1 | DR | 6 | 2 | 1 | 5 | 0 | 2 | 1 | 0 | 13 |
| 3 | 800 | 1 | noDR | 4 | 0 | 2 | 3 | 0 | 5 | 0 | 0 | 6 |
| 3 | 800 | 2 | overall | 10 | 1 | 3 | 6 | 0 | 1 | 1 | 0 | 28 |
| 3 | 800 | 2 | DR | 8 | 1 | 2 | 4 | 0 | 1 | 1 | 0 | 13 |
| 3 | 800 | 2 | noDR | 2 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 15 |
| 3 | 800 | 3 | overall | 13 | 3 | 2 | 7 | 0 | 3 | 7 | 1 | 14 |
| 3 | 800 | 3 | DR | 9 | 0 | 2 | 6 | 0 | 1 | 6 | 1 | 5 |
| 3 | 800 | 3 | noDR | 4 | 3 | 0 | 1 | 0 | 2 | 1 | 0 | 9 |
| 3 | 800 | 4 | overall | 20 | 5 | 2 | 2 | 0 | 2 | 13 | 0 | 6 |
| 3 | 800 | 4 | DR | 12 | 3 | 1 | 2 | 0 | 0 | 10 | 0 | 2 |
| 3 | 800 | 4 | noDR | 8 | 2 | 1 | 0 | 0 | 2 | 3 | 0 | 4 |
| 3 | 800 | 5 | overall | 19 | 4 | 0 | 5 | 0 | 2 | 12 | 1 | 7 |
| 3 | 800 | 5 | DR | 10 | 2 | 0 | 4 | 0 | 1 | 6 | 0 | 7 |
| 3 | 800 | 5 | noDR | 9 | 2 | 0 | 1 | 0 | 1 | 6 | 1 | 0 |
| 10 | 500 | 1 | overall | 14 | 0 | 3 | 7 | 0 | 4 | 0 | 0 | 22 |
| 10 | 500 | 1 | DR | 11 | 0 | 1 | 5 | 0 | 1 | 0 | 0 | 12 |
| 10 | 500 | 1 | noDR | 3 | 0 | 2 | 2 | 0 | 3 | 0 | 0 | 10 |
| 10 | 500 | 2 | overall | 19 | 0 | 2 | 4 | 0 | 0 | 4 | 0 | 21 |
| 10 | 500 | 2 | DR | 16 | 0 | 0 | 4 | 0 | 0 | 4 | 0 | 6 |
| 10 | 500 | 2 | noDR | 3 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 15 |
| 10 | 500 | 3 | overall | 18 | 2 | 1 | 6 | 0 | 7 | 4 | 0 | 12 |
| 10 | 500 | 3 | DR | 13 | 1 | 1 | 5 | 0 | 1 | 4 | 0 | 5 |
| 10 | 500 | 3 | noDR | 5 | 1 | 0 | 1 | 0 | 6 | 0 | 0 | 7 |
| 10 | 500 | 4 | overall | 28 | 4 | 1 | 2 | 0 | 3 | 7 | 1 | 4 |
| 10 | 500 | 4 | DR | 17 | 2 | 1 | 2 | 0 | 0 | 5 | 1 | 2 |
| 10 | 500 | 4 | noDR | 11 | 2 | 0 | 0 | 0 | 3 | 2 | 0 | 2 |
| 10 | 500 | 5 | overall | 26 | 5 | 0 | 4 | 0 | 0 | 10 | 0 | 5 |
| 10 | 500 | 5 | DR | 17 | 2 | 0 | 2 | 0 | 0 | 4 | 0 | 5 |
| 10 | 500 | 5 | noDR | 9 | 3 | 0 | 2 | 0 | 0 | 6 | 0 | 0 |
| 20 | 300 | 1 | overall | 16 | 0 | 4 | 5 | 0 | 2 | 0 | 0 | 23 |
| 20 | 300 | 1 | DR | 14 | 0 | 0 | 4 | 0 | 1 | 0 | 0 | 11 |
| 20 | 300 | 1 | noDR | 2 | 0 | 4 | 1 | 0 | 1 | 0 | 0 | 12 |
| 20 | 300 | 2 | overall | 19 | 1 | 5 | 4 | 0 | 3 | 1 | 1 | 16 |
| 20 | 300 | 2 | DR | 17 | 0 | 1 | 2 | 0 | 0 | 1 | 1 | 8 |
| 20 | 300 | 2 | noDR | 2 | 1 | 4 | 2 | 0 | 3 | 0 | 0 | 8 |
| 20 | 300 | 3 | overall | 21 | 2 | 0 | 3 | 0 | 2 | 6 | 2 | 14 |
| 20 | 300 | 3 | DR | 15 | 1 | 0 | 1 | 0 | 1 | 6 | 2 | 4 |
| 20 | 300 | 3 | noDR | 6 | 1 | 0 | 2 | 0 | 1 | 0 | 0 | 10 |
| 20 | 300 | 4 | overall | 30 | 1 | 2 | 2 | 0 | 4 | 5 | 0 | 6 |
| 20 | 300 | 4 | DR | 18 | 1 | 1 | 1 | 0 | 1 | 4 | 0 | 4 |
| 20 | 300 | 4 | noDR | 12 | 0 | 1 | 1 | 0 | 3 | 1 | 0 | 2 |
| 20 | 300 | 5 | overall | 21 | 6 | 6 | 1 | 0 | 0 | 5 | 1 | 10 |
| 20 | 300 | 5 | DR | 15 | 4 | 1 | 1 | 0 | 0 | 4 | 1 | 4 |
| 20 | 300 | 5 | noDR | 6 | 2 | 5 | 0 | 0 | 0 | 1 | 0 | 6 |

### Missed-grasp anatomy + other diagnostics (overall, i.e. all 50 episodes pooled)

| nsub | max | replan | coarse_far | pinch | finger | height | close_shallow | unknown | grasp_attempts_mean | failed_grasp_attempts_mean | grasp_attempt_success_rate | premature_release_count_mean | press/episode_rate | pushed_dist_pre_grasp_mean | gripper_close_cycles_mean |
|------|-----|--------|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 3 | 800 | 1 | 0.02 | 0.02 | 0.07 | 0.02 | 0.97 | 0.00 | 9.46 | 7.48 | 0.21 | 1.52 | 0.62 | 0.0126 | 8.40 |
| 3 | 800 | 2 | 0.00 | 0.03 | 0.10 | 0.03 | 0.95 | 0.00 | 8.92 | 6.26 | 0.30 | 1.80 | 0.56 | 0.0069 | 7.42 |
| 3 | 800 | 3 | 0.19 | 0.13 | 0.23 | 0.16 | 0.79 | 0.00 | 15.78 | 9.04 | 0.43 | 6.18 | 0.56 | 0.0014 | 15.00 |
| 3 | 800 | 4 | 0.58 | 0.08 | 0.16 | 0.14 | 0.60 | 0.00 | 8.50 | 5.70 | 0.33 | 2.20 | 0.56 | 0.0050 | 8.48 |
| 3 | 800 | 5 | 0.35 | 0.13 | 0.21 | 0.20 | 0.68 | 0.00 | 6.78 | 5.22 | 0.23 | 0.96 | 0.54 | 0.0015 | 6.58 |
| 10 | 500 | 1 | 0.09 | 0.04 | 0.12 | 0.00 | 0.94 | 0.00 | 3.48 | 2.30 | 0.34 | 0.54 | 0.56 | 0.0138 | 3.14 |
| 10 | 500 | 2 | 0.12 | 0.09 | 0.20 | 0.09 | 0.81 | 0.00 | 4.18 | 2.34 | 0.44 | 1.10 | 0.50 | 0.0120 | 3.56 |
| 10 | 500 | 3 | 0.14 | 0.01 | 0.16 | 0.11 | 0.78 | 0.00 | 7.86 | 2.90 | 0.63 | 4.22 | 0.52 | 0.0105 | 7.74 |
| 10 | 500 | 4 | 0.05 | 0.04 | 0.19 | 0.19 | 0.87 | 0.00 | 6.70 | 1.82 | 0.73 | 4.18 | 0.60 | 0.0066 | 6.74 |
| 10 | 500 | 5 | 0.13 | 0.01 | 0.30 | 0.23 | 0.68 | 0.00 | 2.98 | 1.42 | 0.52 | 0.96 | 0.54 | 0.0068 | 2.98 |
| 20 | 300 | 1 | 0.03 | 0.00 | 0.03 | 0.00 | 0.97 | 0.00 | 1.42 | 0.58 | 0.59 | 0.16 | 0.52 | 0.0205 | 1.36 |
| 20 | 300 | 2 | 0.00 | 0.02 | 0.03 | 0.00 | 0.97 | 0.00 | 4.70 | 2.52 | 0.46 | 1.56 | 0.54 | 0.0273 | 4.04 |
| 20 | 300 | 3 | 0.07 | 0.03 | 0.13 | 0.00 | 0.92 | 0.00 | 3.90 | 1.42 | 0.64 | 1.80 | 0.64 | 0.0113 | 3.80 |
| 20 | 300 | 4 | 0.10 | 0.15 | 0.31 | 0.10 | 0.79 | 0.00 | 2.44 | 0.78 | 0.68 | 0.92 | 0.68 | 0.0142 | 2.36 |
| 20 | 300 | 5 | 0.08 | 0.00 | 0.08 | 0.06 | 0.92 | 0.00 | 3.18 | 0.72 | 0.77 | 1.86 | 0.60 | 0.0197 | 3.18 |

## Best combo (nsub=20, replan=4, max=300) on checkpoint 40k

Script: `aera/autonomous/openpi/scripts/eval_variance.py`

Parameters:
  * `--config pi05_ar4_mk3 --checkpoint-dir checkpoints/pi05_ar4_mk3/pi05_ar4_mk3_2026-07-04_16-51-56/40000`
  * Suite shape (defaults, unchanged): DR seeds `1000-1014` (15), no-DR seeds `1000-1009` (10), `k_repeats=2` (50 episodes)
  * `--n-substeps 20 --replan-steps 4 --max-episode-steps 300` (best combo found on checkpoint 70k)
  * `--kinematic-grasp`, prompt: `pick the yellow block and place it on the red target`
  * Output: `eval_results/pi05_ar4_mk3_2026-07-04_16-51-56_ablation_40k/40000/{summary.json,episodes.jsonl}`

### Funnel rates

| mode | n | reached | grasped | lifted | transported | placed |
|------|---|---------|---------|--------|--------------|--------|
| overall | 50 | 0.84 | 0.74 | 0.62 | 0.46 | 0.60 |
| DR | 30 | 0.87 | 0.77 | 0.67 | 0.57 | 0.70 |
| noDR | 20 | 0.80 | 0.70 | 0.55 | 0.30 | 0.45 |

### Between-seed std / within-seed std (mean), per stage (DR / noDR only)

| mode | reached | grasped | lifted | transported | placed |
|------|---------|---------|--------|--------------|--------|
| DR | 0.29 / 0.07 | 0.36 / 0.10 | 0.39 / 0.13 | 0.44 / 0.10 | 0.36 / 0.17 |
| noDR | 0.33 / 0.10 | 0.40 / 0.10 | 0.42 / 0.15 | 0.33 / 0.20 | 0.42 / 0.15 |

### Failure-mode counts (overall / DR / noDR, out of n episodes above)

| mode | success | never_reached | no_grasp_attempt | grasp_missed | wrong_object_grasp | grasped_not_lifted | dropped_early | dropped_or_missed_at_goal | timeout_holding |
|------|----|----|----|----|----|----|----|----|----|
| overall | 30 | 4 | 0 | 3 | 0 | 1 | 5 | 0 | 7 |
| DR | 21 | 2 | 0 | 1 | 0 | 0 | 2 | 0 | 4 |
| noDR | 9 | 2 | 0 | 2 | 0 | 1 | 3 | 0 | 3 |

### Missed-grasp anatomy + other diagnostics (overall, i.e. all 50 episodes pooled)

| coarse_far | pinch | finger | height | close_shallow | unknown | grasp_attempts_mean | failed_grasp_attempts_mean | grasp_attempt_success_rate | premature_release_count_mean | press/episode_rate | pushed_dist_pre_grasp_mean | gripper_close_cycles_mean |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0.12 | 0.07 | 0.23 | 0.05 | 0.77 | 0.00 | 2.26 | 0.86 | 0.62 | 0.66 | 0.52 | 0.0135 | 2.26 |

## To improve in evals:
  * [Done] Write a script that evaluates checkpoints with more attempts/seeds
  * [In Progress] Run evaluation with current checkpoints
  * [Done] Understand why there is difference between evals at training time and done offline. - This was mostly caused by n_replan=10 vs n_replan=5 during training eval (also reducing n_substeps may have some impact, higher could be better, but not confirmed)
  * [Done] Make the evals during training more representative (but not too heavy, we cannot just run 100s of eval trajectories without starving training from resources for too long)
  * [Done] Improve the evals, so that we have better understanding of the failure mode (how?) — metrics.py now tracks failed grasp attempts with tool-frame miss offsets (pinch/finger/height), commanded releases (premature drops), gripper open/close cycles (retry loops / jaw pulsing), block-pressed-into-table contact force, pre-grasp shoving, and a per-episode failure_mode label; eval_variance reports the breakdown per group/seed and tags videos with the mode.
  * [Done] In eval, the arm can sometimes lift the block even if it's not between the jaws, it gets "glued" to the front or side of the jaws, it seems that the lock engages too early. Some successes are "caused" by this behaviour. — Root cause: the lock welded at close-*command* time with only a centre-offset gate (7/20/27 mm around the grip site), never checking the jaws touch the block; the jaws were usually still open (zero contact), so the block froze mid-air at its offset. Fixed by a pinch-contact engage gate (see 13.07.2026 tooling changes).

## Eval tooling changes (12.07.2026)

What was added to the evals (commits a3eb8bd, 0f8db07), effective from the next run:

  * **Failure-mode diagnostics** (`eval/metrics.py`): each episode now records *how* it failed, not just where in the funnel it stopped. All derived from the kinematic lock's command semantics:
    - Grasp attempts: a close command near the block that doesn't engage = missed grasp, with the grip→block offset at closest approach in the gripper tool frame (pinch/finger/height — the engage gate's own axes) and a miss reason (`pinch`/`finger`/`height`/`close_shallow`/`coarse_far`). Signed offset bias shows systematic "approaches in front / to the side" errors.
    - Releases: with the lock, a drop can only be a commanded release — each one records where it happened (dist to goal, height, hold length) and whether premature.
    - Gripper open/close command cycles (retry loops, jaw pulsing), block-pressed-into-table contact force (>3x block weight while a jaw touches it), pre-grasp shoving distance.
    - One `failure_mode` label per episode (`never_reached`, `no_grasp_attempt`, `grasp_missed`, `wrong_object_grasp`, `grasped_not_lifted`, `dropped_early`, `dropped_or_missed_at_goal`, `timeout_holding`, `success`) = the terminal outcome; the event lists keep everything that went wrong on the way. Videos are tagged with the label.
  * **One shared eval suite** (`eval/suite.py`): eval_worker and eval_variance now run the *same* {DR on x seeds, DR off x seeds} x K-repeats grid (env rebuilt per seed so each DR seed has its own reproducible draw) with *identical defaults*: 15 DR seeds x 2 + 10 no-DR seeds x 2 = 50 episodes, seed starts 1000. A default offline run reproduces the training-time suite exactly — removes the suite mismatch as an explanation for training-vs-offline eval differences (the old worker rolled 20 sequential seeds in one env, no repeats).
  * **mlflow layout**: headline metrics stay pooled under plain `eval/...` names (`eval/success_rate`, `eval/funnel/*`, plus new `eval/failure/*`, `eval/miss/*`, release/press/cycle stats); per-group breakdowns under `eval/dr/...` / `eval/nodr/...` incl. between-/within-seed std. The worker also attaches `episodes.jsonl` + `summary.json` per checkpoint as run artifacts, so raw per-attempt/per-release events from training-time evals are preserved.

Caveat: the eval-variance tables below predate these changes (old defaults, no failure-mode fields in their episodes.jsonl); numbers from the new suite are comparable with each other but not 1:1 with those tables.

## Eval tooling changes (13.07.2026): pinch-contact engage gate

Fixes the "block glued to the front/side of the jaws" false grasps. Root cause: the eval env attempted `engage()` on every control step while a close-ish command was latched, and the gate only bounded the block's *centre offset* from the grip site (pinch 7 / finger 20 / height 27 mm) — it never checked that the jaws touch the block. Since the close command fires while the jaws are still open, the weld usually attached with **zero jaw contact**, freezing the block mid-air at whatever offset it had (hence "half-levitating with gaps", glued to the tips, lifted without being between the jaws). The offset tolerances allow poses well outside the jaw pads' own extent (pads span only y∈[-14.4, 8.6], z∈[-2.1, 15.5] mm around the grip site).

  * **`GraspEngageConfig.require_pinch_contact` (default on)** in `envs/kinematic_grasp.py`: `engage()` now only welds while BOTH jaw contact pads are in contact with the candidate with contact normals along the pinch axis (|cos| ≥ 0.5). Consequences:
    - The close command alone welds nothing; the weld attaches a few control steps later, once the jaws have physically closed onto the block. During those steps the block is under real contact physics — it gets recentred (or shoved away) by the closing jaws exactly like collection's scripted close — so the welded pose is the genuinely pinched pose, not a command-time snapshot.
    - A block outside/below/next-to the closed jaws can never weld (no two-sided pinch contact); closed jaws pressing on the block's top face are rejected too (contact normals vertical, not along the pinch axis).
    - Applies to collection as well via the shared config (harmless there: collection engages after a completed scripted close with 0.5 mm preload, which passes). Old permissive behaviour available with `require_pinch_contact=False` for A/B.
  * **New `no_pinch` miss reason** in `eval/metrics.py`: attempt passed every offset/depth gate but the jaws never physically pinched the block during the attempt window (shoved it away, or closed beside/above it). Attempts now carry a `pinched` field in `episodes.jsonl`.
  * Verified with a standalone MuJoCo harness (scene.xml, no env): genuine physical close still welds; the old config reproducibly welds a contactless mid-air block at (finger 17, height 10) mm offset while the new gate refuses; press-on-top rejected; plus an env+metrics integration smoke (engage deferred until pinch, release on open command, metrics finalize).

Caveat: success/funnel numbers from runs before this change may include glue-assisted grasps (the journal suspected "some successes are caused by this behaviour"), so they are optimistic relative to the next run's numbers — expect grasped/success to drop where the policy relied on the free weld, and `grasp_missed`/`no_pinch` to absorb those episodes.

Open question (to decide after testing): the both-pads contact requirement effectively tightens the required close depth — the old close-depth gate accepted a command up to 0.5 mm *short* of the block surface, but contact needs the command to reach/penetrate the surface. With variable block widths in training, the policy may not have that sub-mm close precision, so genuine straddles that stop a hair short would now count as `no_pinch` misses (on the deep side there's no precision demand — the jaws just stall on the block). Watch the `no_pinch` miss rate to see how often this bites; fallbacks if it's too strict: accept one-pad contact, add a small contact margin, or `require_pinch_contact=False`. The planned "jaws always close to 0" change for the next training iteration removes the issue entirely.




## Ablation: best combo (nsub=20, replan=4, max=300) on ALL checkpoints, NEW pinch-contact engage gate

First eval after the 13.07.2026 pinch-contact engage gate (see "Eval tooling changes (13.07.2026)"). Re-runs the best rollout combo found on 70k against every checkpoint, so the funnel is now free of glue-assisted false grasps.

Script: `aera/autonomous/openpi/scripts/eval_variance.py` (driver: `scratchpad/run_ablation_pinch.sh`)

Parameters:
  * `--config pi05_ar4_mk3`
  * Suite shape (defaults): DR seeds `1000-1014` (15), no-DR seeds `1000-1009` (10), `k_repeats=2` (50 episodes/checkpoint)
  * `--n-substeps 20 --replan-steps 4 --max-episode-steps 300 --kinematic-grasp` (best combo from the 70k grid)
  * prompt: `pick the yellow block and place it on the red target`
  * Checkpoints: `checkpoints/pi05_ar4_mk3/pi05_ar4_mk3_2026-07-04_16-51-56/{20000,30000,40000,50000,60000,70000}`
  * Output: `eval_results/pi05_ar4_mk3_2026-07-04_16-51-56_pinch_ablation/<step>/{summary.json,episodes.jsonl}`

### Funnel rates (overall / DR / noDR)

| step | mode | n | reached | grasped | lifted | transported | placed |
|------|------|---|---------|---------|--------|--------------|--------|
| 20k | overall | 50 | 0.84 | 0.60 | 0.56 | 0.40 | 0.48 |
| 20k | DR      | 30 | 0.83 | 0.60 | 0.57 | 0.47 | 0.53 |
| 20k | noDR    | 20 | 0.85 | 0.60 | 0.55 | 0.30 | 0.40 |
| 30k | overall | 50 | 0.82 | 0.58 | 0.42 | 0.28 | 0.42 |
| 30k | DR      | 30 | 0.83 | 0.50 | 0.43 | 0.30 | 0.40 |
| 30k | noDR    | 20 | 0.80 | 0.70 | 0.40 | 0.25 | 0.45 |
| 40k | overall | 50 | 0.86 | 0.60 | 0.46 | 0.30 | 0.38 |
| 40k | DR      | 30 | 0.87 | 0.60 | 0.60 | 0.43 | 0.47 |
| 40k | noDR    | 20 | 0.85 | 0.60 | 0.25 | 0.10 | 0.25 |
| 50k | overall | 50 | 0.90 | 0.62 | 0.50 | 0.28 | 0.40 |
| 50k | DR      | 30 | 0.90 | 0.57 | 0.47 | 0.23 | 0.40 |
| 50k | noDR    | 20 | 0.90 | 0.70 | 0.55 | 0.35 | 0.40 |
| 60k | overall | 50 | 0.78 | 0.60 | 0.54 | 0.34 | 0.42 |
| 60k | DR      | 30 | 0.83 | 0.63 | 0.60 | 0.40 | 0.47 |
| 60k | noDR    | 20 | 0.70 | 0.55 | 0.45 | 0.25 | 0.35 |
| 70k | overall | 50 | 0.84 | 0.60 | 0.48 | 0.36 | 0.44 |
| 70k | DR      | 30 | 0.90 | 0.70 | 0.63 | 0.50 | 0.53 |
| 70k | noDR    | 20 | 0.75 | 0.45 | 0.25 | 0.15 | 0.30 |

### Between-seed std / within-seed std (mean), per stage (DR / noDR only)

| step | mode | reached | grasped | lifted | transported | placed |
|------|------|---------|---------|--------|--------------|--------|
| 20k | DR   | 0.30 / 0.10 | 0.45 / 0.07 | 0.44 / 0.10 | 0.43 / 0.13 | 0.39 / 0.20 |
| 20k | noDR | 0.32 / 0.05 | 0.37 / 0.20 | 0.35 / 0.25 | 0.33 / 0.20 | 0.30 / 0.30 |
| 30k | DR   | 0.35 / 0.03 | 0.41 / 0.17 | 0.44 / 0.10 | 0.40 / 0.10 | 0.45 / 0.07 |
| 30k | noDR | 0.40 / 0.00 | 0.40 / 0.10 | 0.37 / 0.20 | 0.34 / 0.15 | 0.42 / 0.15 |
| 40k | DR   | 0.29 / 0.07 | 0.37 / 0.20 | 0.42 / 0.13 | 0.40 / 0.17 | 0.39 / 0.20 |
| 40k | noDR | 0.32 / 0.05 | 0.37 / 0.20 | 0.34 / 0.15 | 0.20 / 0.10 | 0.34 / 0.15 |
| 50k | DR   | 0.27 / 0.03 | 0.48 / 0.03 | 0.46 / 0.07 | 0.31 / 0.17 | 0.37 / 0.20 |
| 50k | noDR | 0.30 / 0.00 | 0.40 / 0.10 | 0.35 / 0.25 | 0.32 / 0.25 | 0.37 / 0.20 |
| 60k | DR   | 0.35 / 0.03 | 0.43 / 0.10 | 0.45 / 0.07 | 0.45 / 0.07 | 0.50 / 0.00 |
| 60k | noDR | 0.33 / 0.20 | 0.35 / 0.25 | 0.35 / 0.25 | 0.34 / 0.15 | 0.39 / 0.15 |
| 70k | DR   | 0.27 / 0.03 | 0.36 / 0.17 | 0.39 / 0.17 | 0.41 / 0.17 | 0.29 / 0.33 |
| 70k | noDR | 0.34 / 0.15 | 0.42 / 0.15 | 0.25 / 0.25 | 0.23 / 0.15 | 0.33 / 0.20 |

### Failure-mode counts (overall / DR / noDR, out of n episodes above)

| step | mode | success | never_reached | no_grasp_attempt | grasp_missed | wrong_object_grasp | grasped_not_lifted | dropped_early | dropped_or_missed_at_goal | timeout_holding |
|------|------|----|----|----|----|----|----|----|----|----|
| 20k | overall | 24 | 4 | 2 | 9  | 0 | 2 | 3 | 0 | 6 |
| 20k | DR      | 16 | 3 | 0 | 6  | 0 | 1 | 1 | 0 | 3 |
| 20k | noDR    | 8  | 1 | 2 | 3  | 0 | 1 | 2 | 0 | 3 |
| 30k | overall | 21 | 5 | 4 | 5  | 0 | 4 | 3 | 0 | 8 |
| 30k | DR      | 12 | 3 | 3 | 5  | 0 | 0 | 1 | 0 | 6 |
| 30k | noDR    | 9  | 2 | 1 | 0  | 0 | 4 | 2 | 0 | 2 |
| 40k | overall | 19 | 3 | 0 | 11 | 0 | 7 | 4 | 0 | 6 |
| 40k | DR      | 14 | 2 | 0 | 7  | 0 | 0 | 1 | 0 | 6 |
| 40k | noDR    | 5  | 1 | 0 | 4  | 0 | 7 | 3 | 0 | 0 |
| 50k | overall | 20 | 1 | 1 | 11 | 0 | 3 | 7 | 1 | 6 |
| 50k | DR      | 12 | 1 | 0 | 9  | 0 | 0 | 5 | 0 | 3 |
| 50k | noDR    | 8  | 0 | 1 | 2  | 0 | 3 | 2 | 1 | 3 |
| 60k | overall | 21 | 7 | 1 | 6  | 0 | 2 | 5 | 0 | 8 |
| 60k | DR      | 14 | 3 | 0 | 4  | 0 | 0 | 3 | 0 | 6 |
| 60k | noDR    | 7  | 4 | 1 | 2  | 0 | 2 | 2 | 0 | 2 |
| 70k | overall | 22 | 4 | 1 | 8  | 1 | 4 | 4 | 1 | 5 |
| 70k | DR      | 16 | 1 | 0 | 5  | 0 | 1 | 2 | 1 | 4 |
| 70k | noDR    | 6  | 3 | 1 | 3  | 1 | 3 | 2 | 0 | 1 |

### Missed-grasp anatomy + other diagnostics (overall, all 50 episodes pooled)

Note the new `no_pinch` miss reason (attempt passed every offset/depth gate but the jaws never physically pinched the block — the gate's new check).

| step | coarse_far | pinch | finger | height | close_shallow | no_pinch | grasp_attempts_mean | failed_grasp_attempts_mean | grasp_attempt_success_rate | premature_release_count_mean | press/episode_rate | pushed_dist_pre_grasp_mean | gripper_close_cycles_mean |
|------|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 20k | 0.18 | 0.04 | 0.20 | 0.15 | 0.67 | 0.12 | 2.90 | 1.64 | 0.43 | 0.64 | 0.64 | 0.0096 | 2.80 |
| 30k | 0.25 | 0.02 | 0.11 | 0.03 | 0.77 | 0.06 | 4.34 | 1.76 | 0.59 | 2.04 | 0.58 | 0.0168 | 4.36 |
| 40k | 0.26 | 0.08 | 0.20 | 0.05 | 0.68 | 0.10 | 3.42 | 2.00 | 0.42 | 0.88 | 0.64 | 0.0150 | 3.28 |
| 50k | 0.01 | 0.27 | 0.39 | 0.16 | 0.70 | 0.13 | 4.16 | 2.30 | 0.45 | 1.44 | 0.74 | 0.0167 | 4.14 |
| 60k | 0.18 | 0.06 | 0.24 | 0.08 | 0.57 | 0.12 | 2.32 | 1.02 | 0.56 | 0.84 | 0.60 | 0.0158 | 2.34 |
| 70k | 0.09 | 0.09 | 0.10 | 0.05 | 0.70 | 0.15 | 4.00 | 1.60 | 0.60 | 1.62 | 0.66 | 0.0173 | 3.92 |

### Read

  * **Training beyond 20k brings essentially no benefit.** Success (placed): 20k=0.48, 30k=0.42, 40k=0.38, 50k=0.40, 60k=0.42, 70k=0.44. With n=50 and 2 correlated repeats/seed the effective SE is ~8-10pp, so all six sit within ~1 SE of each other — the curve is *flat*. Safe statement is "no improvement past 20k", NOT "20k is best" (we can't rank them). Can't tell if 20k is already past the plateau: 10k wasn't saved locally and there's no earlier checkpoint.
  * **`grasped` is pinned at 0.58-0.62 across every checkpoint.** The policy reaches its final reach+grasp skill by 20k and never improves it; extra steps move neither grasp nor the downstream lift->transport bleed. The remaining loss is systematic: `close_shallow` dominates missed grasps (0.57-0.77) and `grasp_missed`/`grasped_not_lifted` are the top non-success buckets — the policy commands closes that don't reach the block surface. This is a **data / action-space limitation, not undertraining**, which is why more gradient steps do nothing and why the planned "jaws always close to 0" change is the right lever.
  * **The pinch gate did what its caveat predicted, and flattened the apparent training curve.** The old (glue-permitting) trainargs table showed a rising curve (20k placed=0.28 -> 70k=0.64) that looked like a training benefit; a chunk of it was glue-assisted welds inflating later checkpoints. At the two checkpoints with a same-combo baseline (the 70k grid at replan4/max300, and the 40k best-combo run), the gate knocked 40k 0.60->0.38 and 70k 0.60->0.44. With honest grasps the curve is flat — "training helps" was partly an eval artifact.
  * **`no_pinch` misses are only 0.06-0.15 of failed attempts** — the both-pads contact requirement bites a little but is not dominant, so the gate is not obviously too strict (the open question from 13.07). `close_shallow` remains the real problem.
  * **Bottleneck is grasp/close mechanics baked into the data, not training duration.** Scaling steps further on this dataset is not the move; the next-iteration data changes (below) are.

## To change for next training iteration
  * Do not include partial grasp, maybe even not wrong approach (have to think about that)
  * Make the jaws close to 0 always, don't force arm to estimate the size of block to precisely close around the object.

### Observations
   Things to change but not yet clear how.

  * There is this behavior in eval where arm pushed down on block, it rotates (while going partially into the table) and because of the rotation it position itself inside jaws. Sometimes it flips to exactly flat position so that we ends-up with good grasp, sometimes it ends-up grasping "diagonally" and we have OOD grasp.
