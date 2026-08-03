"""Shared eval suite: one rollout/summary code path for training-time and
offline evals.

Before this module existed, the decoupled eval worker and the offline
eval-variance script ran *different* suites: the worker rolled 20 sequential
seeds in a single env (so with domain rand on, every episode would share one
DR draw) with no repeats, while eval_variance ran a structured
{DR on x N_DR seeds, DR off x N_S seeds} x K-repeats grid with the env rebuilt
per seed. That made on-training curves and offline deep-dives incomparable by
construction (training_journal/06.07.2026: "understand why there is difference
between evals at training time and done offline").

Now both consumers run *this* suite with the *same defaults* (15 DR seeds x 2
+ 10 no-DR seeds x 2 = 50 episodes, same seed starts), so by default the
training-time eval and an offline eval are the identical suite — same
scenarios, same numbers — and either can be scaled up/down via flags:

  - Each seed fixes a scenario: spawn geometry via env.reset(seed=...), plus -
    for DR seeds - the sampled visual domain-rand config. The env is rebuilt
    per seed because the DR config is baked in at env construction, not
    per-reset.
  - Each seed is rolled out k_repeats times with no change to the scenario, so
    across-repeat spread isolates policy/inference variance from across-seed
    spread (scenario variance).

Consumers:
  - scripts/eval_variance.py: one-shot suite on one checkpoint, results to
    JSON on disk (pass higher seed/repeat counts for deep dives).
  - scripts/eval_worker.py: the suite per new checkpoint, summary scalars to
    mlflow (via flatten_for_mlflow) + raw episodes as an artifact.
"""

import dataclasses
import json
import logging
import pathlib

import numpy as np

from aera.autonomous.envs.task_env_factory import (
    DEPLOY_MAX_EPISODE_STEPS,
    DEPLOY_N_SUBSTEPS,
    DEPLOY_REPLAN_STEPS,
)
from aera.autonomous.openpi.eval import metrics as _metrics
from aera.autonomous.openpi.scripts.run_policy_on_env import (
    Args as RolloutArgs,
    _build_env,
    _resolve_prompts,
    _run_episode,
    _save_episode_video,
)

FUNNEL_STAGES = ("reached", "grasped", "lifted", "transported", "placed")
PROGRESS_SCALARS = ("reach_progress", "place_progress")


@dataclasses.dataclass
class SuiteConfig:
    """Suite shape + rollout parameters, shared by all eval consumers.

    Defaults are the ONE canonical suite both the training-time worker and the
    offline eval_variance CLI run: 20 DR seeds x 2 = 40 episodes. The seed start
    sits at 1000, deliberately separate from training seeds. Keep the two
    consumers' defaults identical (both mirror these) so their numbers stay
    directly comparable; scale via flags for deep dives.

    Every episode is full-DR, drawn from the same generator as collection.
    There is no no-DR arm: collection produces zero clean episodes, so a no-DR
    scene differs from EVERY training episode on EVERY visual axis at once —
    the exact centre the DR distribution deliberately never samples. It is the
    most OOD point in appearance space, not a baseline that "should" be easier,
    and reporting it as one read ordinary OOD degradation backwards as "DR is
    hard". It also isn't the deployment target, which will bring its own OOD
    elements. What we need first is one honest in-distribution number.
    """

    # --- Suite shape ---
    n_dr_seeds: int = 20  # every seed is domain_rand on
    k_repeats: int = 2  # rollouts per seed (isolates policy/inference variance)
    dr_seed_start: int = 1000

    # --- Rollout parameters ---
    prompt: str = "pick the yellow block and place it on the red target"
    # Rate defaults come from task_env_factory so this and run_policy_on_env
    # cannot drift again. ~3x the demonstrated episode length, leaving room for
    # retries without letting a stuck policy burn the whole budget.
    max_episode_steps: int = DEPLOY_MAX_EPISODE_STEPS
    # NOT inherited from the old ablation's replan=4: that was found at
    # n_substeps=20 on skip3 data, i.e. under a 4x rate mismatch, so the number
    # means something different here and the coincidence is not evidence.
    # Confirm with a 1-5 sweep (eval_variance --replan-steps) on the first
    # checkpoint.
    replan_steps: int = DEPLOY_REPLAN_STEPS
    # mj-steps per env.step. MUST match the dataset `--skip` the checkpoint was
    # trained on — see the invariant in task_env_factory.
    n_substeps: int = DEPLOY_N_SUBSTEPS
    kinematic_grasp: bool = True

    # --- Videos ---
    save_videos: bool = False
    video_out_path: str = "data/ar4_mk3/eval_suite_videos"


@dataclasses.dataclass
class EpisodeRecord:
    seed: int
    repeat: int
    metrics: _metrics.EpisodeMetrics

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "repeat": self.repeat,
            **self.metrics.to_dict(),
        }


def _rollout_args(cfg: SuiteConfig, *, seed: int) -> RolloutArgs:
    return RolloutArgs(
        prompt=cfg.prompt,
        replan_steps=cfg.replan_steps,
        num_episodes=1,
        max_episode_steps=cfg.max_episode_steps,
        domain_rand=True,
        headless=True,
        kinematic_grasp=cfg.kinematic_grasp,
        n_substeps=cfg.n_substeps,
        two_phase_prompt=False,
        seed=seed,
        video_out_path=cfg.video_out_path,
    )


def _run_seed_repeats(
    cfg: SuiteConfig,
    policy,
    model_path: str,
    seed: int,
) -> list[EpisodeRecord]:
    """Build one env for `seed` (baking in its DR config if enabled) and roll it
    out `k_repeats` times, resetting to the *same* seed every time so the K
    repeats share one scenario."""
    # Seed the global RNG before resolving prompts: this is what makes the
    # domain-rand visual config (materials/lighting/props/colors) reproducible
    # per seed, since DR is baked in at env construction, not per-reset.
    np.random.seed(seed)
    rollout_args = _rollout_args(cfg, seed=seed)
    pick_prompt, place_prompt, dr_config = _resolve_prompts(rollout_args)
    env = _build_env(rollout_args, model_path, dr_config)

    records = []
    try:
        for repeat in range(cfg.k_repeats):
            # episode_idx is always 0 so _run_episode's env.reset(seed=seed+0)
            # is identical every repeat -- only policy/inference can differ.
            ep, replay_images, final_prompt = _run_episode(
                rollout_args, env, policy, pick_prompt, place_prompt, 0, None
            )
            records.append(EpisodeRecord(seed=seed, repeat=repeat, metrics=ep))
            if cfg.save_videos:
                # Failure mode in the filename so reviewing a specific mode
                # (e.g. all grasp_missed episodes) is a glob, not a full watch.
                _save_episode_video(
                    replay_images,
                    cfg.video_out_path,
                    episode_idx=f"seed{seed}_rep{repeat}_{ep.failure_mode}",
                    prompt=final_prompt,
                    success=ep.placed,
                )
            logging.info(
                "  [seed=%d rep=%d/%d] reached=%s grasped=%s transported=%s "
                "placed=%s mode=%s",
                seed,
                repeat + 1,
                cfg.k_repeats,
                ep.reached,
                ep.grasped,
                ep.transported,
                ep.placed,
                ep.failure_mode,
            )
    finally:
        env.close()
    return records


def run_suite(cfg: SuiteConfig, policy, model_path: str) -> list[EpisodeRecord]:
    """Run the seeds x repeats grid, every episode full-DR. Returns all episode
    records; summarize with :func:`summarize`."""
    if cfg.save_videos:
        pathlib.Path(cfg.video_out_path).mkdir(parents=True, exist_ok=True)
    records: list[EpisodeRecord] = []
    seeds = _dr_seeds(cfg)
    for i, seed in enumerate(seeds):
        logging.info("seed %d (%d/%d)", seed, i + 1, len(seeds))
        records.extend(_run_seed_repeats(cfg, policy, model_path, seed))
    return records


def _dr_seeds(cfg: SuiteConfig) -> range:
    return range(cfg.dr_seed_start, cfg.dr_seed_start + cfg.n_dr_seeds)


def _per_seed_stats(episodes: list[_metrics.EpisodeMetrics], seed: int) -> dict:
    out: dict = {"seed": seed, "n": len(episodes)}
    for stage in FUNNEL_STAGES:
        vals = [float(getattr(e, stage)) for e in episodes]
        out[f"{stage}_rate"] = float(np.mean(vals))
        out[f"{stage}_std"] = float(np.std(vals))
    for name in PROGRESS_SCALARS:
        vals = [float(getattr(e, name)) for e in episodes]
        out[f"{name}_mean"] = float(np.mean(vals))
        out[f"{name}_std"] = float(np.std(vals))
    # Per-seed outcome counts: shows whether a seed fails *consistently* the
    # same way across its K repeats or scatters across modes.
    out["failure_modes"] = _metrics.failure_mode_counts(episodes)
    return out


def _summary_block(records: list[EpisodeRecord], seeds: range) -> dict:
    episodes = [r.metrics for r in records]
    per_seed = [
        _per_seed_stats([r.metrics for r in records if r.seed == s], s) for s in seeds
    ]
    summary: dict = {
        "aggregate": _metrics.aggregate(episodes),
        "failure_modes": _metrics.failure_mode_counts(episodes),
        "per_seed": per_seed,
        # Between-seed spread: std, across seeds, of each seed's own mean rate.
        # This is scenario variance (different spawn geometry / DR draw).
        "between_seed_std": {},
        # Within-seed spread: mean, across seeds, of each seed's own std across
        # its K repeats. This is policy/inference variance at a fixed scenario.
        "within_seed_std_mean": {},
    }
    for stage in FUNNEL_STAGES:
        summary["between_seed_std"][stage] = float(
            np.std([p[f"{stage}_rate"] for p in per_seed])
        )
        summary["within_seed_std_mean"][stage] = float(
            np.mean([p[f"{stage}_std"] for p in per_seed])
        )
    return summary


def summarize(records: list[EpisodeRecord], cfg: SuiteConfig) -> dict:
    """Summary over all episodes, with the seed-variance decomposition.

    There is one block now ("overall"). With the no-DR arm gone every episode
    is drawn from the same in-distribution scenario distribution, so the old
    per-group split would just be the overall block repeated. Note the
    ``domain_rand`` / ``no_domain_rand`` keys that eval_variance's summary.json
    used to carry are gone with it.
    """
    return {"overall": _summary_block(records, _dr_seeds(cfg))}


def flatten_for_mlflow(summary: dict) -> dict[str, float]:
    """Flatten to scalar metrics under the plain ``eval/...`` names, so
    `eval/success_rate`, `eval/funnel/...` etc. stay the headline curves and
    remain continuous with the pre-suite worker's naming.

    The ``eval/dr/*`` and ``eval/nodr/*`` breakdown series STOP here: with a
    single in-distribution arm, ``eval/...`` now IS the DR number. Historical
    runs keep their old series; new runs simply don't extend them.
    """
    out: dict[str, float] = dict(summary["overall"]["aggregate"])
    for stage, value in summary["overall"]["between_seed_std"].items():
        out[f"eval/between_seed_std/{stage}"] = value
    for stage, value in summary["overall"]["within_seed_std_mean"].items():
        out[f"eval/within_seed_std/{stage}"] = value
    return out


def log_summary(summary: dict, cfg: SuiteConfig) -> None:
    """Pretty-print the funnel, seed-variance spread, failure-mode distribution
    and missed-grasp anatomy. One block: every episode is in-distribution."""
    block = summary["overall"]
    agg = block["aggregate"]
    n = int(agg.get("eval/num_episodes", 0))
    funnel = " / ".join(
        f"{stage}={agg.get(f'eval/funnel/{stage}_rate', 0.0) * 100:.0f}%"
        for stage in FUNNEL_STAGES
    )
    logging.info(
        "in-distribution (n_seeds=%d, k=%d, episodes=%d): %s",
        cfg.n_dr_seeds, cfg.k_repeats, n, funnel,
    )
    between = " / ".join(
        f"{stage}={block['between_seed_std'][stage] * 100:.0f}pp"
        for stage in FUNNEL_STAGES
    )
    within = " / ".join(
        f"{stage}={block['within_seed_std_mean'][stage] * 100:.0f}pp"
        for stage in FUNNEL_STAGES
    )
    logging.info("  between-seed std: %s (scenario variance)", between)
    logging.info("  within-seed  std: %s (repeat/policy variance)", within)
    modes = " / ".join(
        f"{mode}={count / max(n, 1) * 100:.0f}%"
        for mode, count in block["failure_modes"].items()
    )
    logging.info("  failure modes: %s", modes)
    miss_keys = [
        ("side (pinch)", "eval/miss/pinch_rate"),
        ("front/back (finger)", "eval/miss/finger_rate"),
        ("too high (height)", "eval/miss/height_rate"),
        ("shallow close", "eval/miss/close_shallow_rate"),
        ("far", "eval/miss/coarse_far_rate"),
    ]
    if any(k in agg for _, k in miss_keys):
        miss = " / ".join(
            f"{lbl}={agg.get(key, 0.0) * 100:.0f}%" for lbl, key in miss_keys
        )
        logging.info(
            "  missed-grasp anatomy (%d failed attempts): %s",
            int(agg.get("eval/failed_grasp_attempts_mean", 0.0) * n),
            miss,
        )


def write_episodes_jsonl(records: list[EpisodeRecord], path: pathlib.Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
