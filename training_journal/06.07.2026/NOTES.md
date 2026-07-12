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
I'm thinking, since the training just sees random step in trajectory (TODO confirm that is really just gets current state, not a N recent steps), it doesn't actually learn the intended "fix your approach"/"fix your grasp", it learns to approach awkwardly and randomly drop.

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
