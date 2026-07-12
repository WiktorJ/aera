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


## checkpoint 50k
Never works in manual tests. Arm approaches fine but then it tries to grasp a bit off (like in front and next to the block) and obviously keeps failing to grasp and lift. It goes down tries to graps, fails goes a bit up and tries again and so on, usually the longer it goes the worse it gets. Sometimes it manages an awkward grasp (not nicely enclosed, but half levitating with some gaps between object and jaws) but usually drops it. This won't be fixed changing the fully closed to 0, the position of gripper is not precise enough. Even a few it did grasp, it didn't drop it off.

Works better with DR off. Sometimes it grasp okish but then randomly drops it, it look a bit like an artifact of the recovery/partial grasp (policy seems to learn to just drop???), have to revisit how this is implemented/perhaps do not use this feature. Maybe it's similar story with wrong approach perturbation? It learns to go approach a bit off?
I'm thinking, since the training just sees random step in trajectory (TODO confirm that is really just gets current state, not a N recent steps), it doesn't actually learn the intended "fix your approach"/"fix your grasp", it learns to approach awkwardly and randomly drop. If you think about, policy will learn to pick up after drop, simply because it is just trained to pick up when block is on the table, so when it slips, policy will naturally understand that now is time to pick it up.

Overall, useless policy.

## checkpoint 40k
Worked 1 out of 10 tries with DR. In general symptoms are similar like in 50k, but approach is a bit better (it sill doesn't puts the jaws around object, but less frequently). I also see sometimes the jaws pulsating back and forth when in grasping pose, like the arm cannot properly close them and then tries to open and try again (here maybe the full closure would help). 

Doesn't seem to be any better with DR off.

Overall, useless policy.

## checkpoint 30k
Seems a bit better overall, but not much, it was able to grasp few more times (not super precisely and still with the visible gaps) but drops prematurely. Seems to behave better with DR off.

Overall, useless policy.

## checkpoint 20k
Seems again slightly better than 30k. Positioning around the object is a bit better, it grasped few times, be the grip was rather bad (gaps, etc.). Overall not much difference from 30k, maybe less often approach is off to the side/in front (but more often jaws press on the object). 

Overall, useless policy.

## checkpoint 60k
This is peculiar, but I found 60k to be doing better, probably the best checkpoint so far. In DR off it actually grasped in all 5 attempts (crapy grasps mostly, but it's something), it succeeded once. The problem with arm going to much to the side or in front seems to be much less prevalent here. It actually succeeded 2/10 in DR also. But still many of previously described failure modes are present.

Overall, useless policy. But can sometimes do the job, which is better than the other ones.

## checkpoint 70k
Nothing new, but seems worse than 60k (no success with DR nor without DR)


## Additional details about manual tests:
  * It doesn't use the same seed as eval done during training, but the same seeds between checkpoint.
  * Seeds 1-10 were tested with DR and seed 1-5 without DR
  * Eval numbers are most likely skewed upwards because 2 out of 20 start with object already on target (so real success ~= success -0.1)
  * Eval command `uv run aera/autonomous/openpi/scripts/run_policy_on_env.py --replan_steps=10 --n_substeps=3 --max_episode_steps=1000 --seed=<seed> --domain_rand`
  * Server start-up command `uv run aera/autonomous/openpi/scripts/serve_policy.py checkpoint:checkpoint --checkpoint.config=pi05_ar4_mk3 --checkpoint.dir=checkpoints/pi05_ar4_mk3/pi05_ar4_mk3_2026-07-04_16-51-56/70000`

## Summary
  * It seems weird that, I was not able to match even the modest success of eval during training. I think first order of business is to figure out this one to eliminate all errors coming from some training, on training eval, and manual eval. In particular it seems that checkpoint 20k or 60k is the best, but this is not reflected by eval stats.
  * With DR off, the results are better. Not surprising, DR introduced a lot of visual noise, shadows, etc. With DR off, there is just few object and colors of objects and target are very easily distinguishable from background.
  * We should have automated evals that would be able to find more detailed failure patters. Manually testing a describing these as above is not feasible. Even for the same seed with have different behaviours, so running just once per seed is not enough. The manual test I did probably have huge variance/
  * Taking all these into account, I think the description of behaviour per checkpoint can be highly misleading for the actual performance. Because of 1) variance 2) My bias (it just get tiring) 3) My failure to put in words 3d arm movement and all the failure modes (there are some that I omitted, e.g. happens often that jaws push on the object (pushing it into the table), this would be catastrophic in real execution.)

## To improve in evals:
  * [Done] Write a script that evaluates checkpoints with more attempts/seeds
  * [In Progress] Run evaluation with current checkpoints
  * Understand why there is difference between evals at training time and done offline.
  * Make the evals during training more representative (but not too heavy, we cannot just run 100s of eval trajectories without starving training from resources for too long)
  * [Done] Improve the evals, so that we have better understanding of the failure mode (how?) — metrics.py now tracks failed grasp attempts with tool-frame miss offsets (pinch/finger/height), commanded releases (premature drops), gripper open/close cycles (retry loops / jaw pulsing), block-pressed-into-table contact force, pre-grasp shoving, and a per-episode failure_mode label; eval_variance reports the breakdown per group/seed and tags videos with the mode.

## To change for next training iteration
  * Do not include partial grasp, maybe even not wrong approach (have to think about that)
  * Make the jaws close to 0 always, don't force arm to estimate the size of block to precisely close around the object.

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
