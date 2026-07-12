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
    offline eval_variance CLI run: 15 DR seeds x 2 + 10 no-DR seeds x 2 = 50
    episodes. Seed starts sit at 1000, deliberately separate from training
    seeds. Keep the two consumers' defaults identical (both mirror these) so
    their numbers stay directly comparable; scale via flags for deep dives.
    """

    # --- Suite shape ---
    n_dr_seeds: int = 15  # seeds evaluated with domain_rand on
    n_seeds: int = 10  # seeds evaluated with domain_rand off
    k_repeats: int = 2  # rollouts per seed (isolates policy/inference variance)
    dr_seed_start: int = 1000
    seed_start: int = 1000

    # --- Rollout parameters ---
    prompt: str = "pick the yellow block and place it on the red target"
    max_episode_steps: int = 1000
    replan_steps: int = 10
    # mj-steps per env.step. MUST match the dataset `--skip` the checkpoint was
    # trained on (see run_policy_on_env.Args.n_substeps).
    n_substeps: int = 3
    kinematic_grasp: bool = True

    # --- Videos ---
    save_videos: bool = False
    video_out_path: str = "data/ar4_mk3/eval_suite_videos"


@dataclasses.dataclass
class EpisodeRecord:
    seed: int
    repeat: int
    domain_rand: bool
    metrics: _metrics.EpisodeMetrics

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "repeat": self.repeat,
            "domain_rand": self.domain_rand,
            **self.metrics.to_dict(),
        }


def _rollout_args(cfg: SuiteConfig, *, domain_rand: bool, seed: int) -> RolloutArgs:
    return RolloutArgs(
        prompt=cfg.prompt,
        replan_steps=cfg.replan_steps,
        num_episodes=1,
        max_episode_steps=cfg.max_episode_steps,
        domain_rand=domain_rand,
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
    domain_rand: bool,
) -> list[EpisodeRecord]:
    """Build one env for `seed` (baking in its DR config if enabled) and roll it
    out `k_repeats` times, resetting to the *same* seed every time so the K
    repeats share one scenario."""
    # Seed the global RNG before resolving prompts: this is what makes the
    # domain-rand visual config (materials/lighting/props/colors) reproducible
    # per seed, since DR is baked in at env construction, not per-reset.
    np.random.seed(seed)
    rollout_args = _rollout_args(cfg, domain_rand=domain_rand, seed=seed)
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
            records.append(
                EpisodeRecord(
                    seed=seed, repeat=repeat, domain_rand=domain_rand, metrics=ep
                )
            )
            if cfg.save_videos:
                tag = "dr" if domain_rand else "nodr"
                # Failure mode in the filename so reviewing a specific mode
                # (e.g. all grasp_missed episodes) is a glob, not a full watch.
                _save_episode_video(
                    replay_images,
                    cfg.video_out_path,
                    episode_idx=f"{tag}_seed{seed}_rep{repeat}_{ep.failure_mode}",
                    prompt=final_prompt,
                    success=ep.placed,
                )
            logging.info(
                "  [%s seed=%d rep=%d/%d] reached=%s grasped=%s transported=%s "
                "placed=%s mode=%s",
                "dr" if domain_rand else "nodr",
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
    """Run the full {DR on, DR off} x seeds x repeats grid. Returns all episode
    records; split/summarize with :func:`summarize`."""
    if cfg.save_videos:
        pathlib.Path(cfg.video_out_path).mkdir(parents=True, exist_ok=True)
    records: list[EpisodeRecord] = []
    for domain_rand, seeds in (
        (True, _dr_seeds(cfg)),
        (False, _nodr_seeds(cfg)),
    ):
        label = "domain_rand=on" if domain_rand else "domain_rand=off"
        for i, seed in enumerate(seeds):
            logging.info("[%s] seed %d (%d/%d)", label, seed, i + 1, len(seeds))
            records.extend(
                _run_seed_repeats(cfg, policy, model_path, seed, domain_rand)
            )
    return records


def _dr_seeds(cfg: SuiteConfig) -> range:
    return range(cfg.dr_seed_start, cfg.dr_seed_start + cfg.n_dr_seeds)


def _nodr_seeds(cfg: SuiteConfig) -> range:
    return range(cfg.seed_start, cfg.seed_start + cfg.n_seeds)


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


def _group_summary(records: list[EpisodeRecord], seeds: range) -> dict:
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
    """Overall summary over all episodes, plus per-group (DR on/off) summaries
    keyed like eval_variance's summary.json always was. The overall block is
    the primary read (one number set to reason about); the groups are the
    breakdown. Seed-variance decomposition stays per-group only, since pooling
    DR and no-DR seeds would mix two different scenario distributions."""
    all_episodes = [r.metrics for r in records]
    out: dict = {
        "overall": {
            "aggregate": _metrics.aggregate(all_episodes),
            "failure_modes": _metrics.failure_mode_counts(all_episodes),
        }
    }
    if cfg.n_dr_seeds:
        out["domain_rand"] = _group_summary(
            [r for r in records if r.domain_rand], _dr_seeds(cfg)
        )
    if cfg.n_seeds:
        out["no_domain_rand"] = _group_summary(
            [r for r in records if not r.domain_rand], _nodr_seeds(cfg)
        )
    return out


def flatten_for_mlflow(summary: dict) -> dict[str, float]:
    """Flatten to scalar metrics. The overall aggregate keeps its plain
    ``eval/...`` names (so `eval/success_rate`, `eval/funnel/...` etc. are the
    headline curves, continuous with the pre-suite worker's naming); each
    group's aggregate is repeated under ``eval/dr/...`` / ``eval/nodr/...`` as
    the breakdown, plus the per-group seed-variance decomposition."""
    out: dict[str, float] = dict(summary["overall"]["aggregate"])
    for group_key, prefix in (("domain_rand", "dr"), ("no_domain_rand", "nodr")):
        group = summary.get(group_key)
        if group is None:
            continue
        for key, value in group["aggregate"].items():
            out[key.replace("eval/", f"eval/{prefix}/", 1)] = value
        for stage, value in group["between_seed_std"].items():
            out[f"eval/{prefix}/between_seed_std/{stage}"] = value
        for stage, value in group["within_seed_std_mean"].items():
            out[f"eval/{prefix}/within_seed_std/{stage}"] = value
    return out


def log_summary(summary: dict, cfg: SuiteConfig) -> None:
    """Pretty-print the overall funnel + failure modes, then each group's
    funnel, seed-variance spread, failure-mode distribution, and missed-grasp
    anatomy."""
    overall = summary["overall"]["aggregate"]
    n_total = int(overall.get("eval/num_episodes", 0))
    funnel = " / ".join(
        f"{stage}={overall.get(f'eval/funnel/{stage}_rate', 0.0) * 100:.0f}%"
        for stage in FUNNEL_STAGES
    )
    logging.info("overall (episodes=%d): %s", n_total, funnel)
    modes = " / ".join(
        f"{mode}={count / max(n_total, 1) * 100:.0f}%"
        for mode, count in summary["overall"]["failure_modes"].items()
    )
    logging.info("  failure modes: %s", modes)

    for group_key, label, n_seeds in (
        ("domain_rand", "domain_rand=on", cfg.n_dr_seeds),
        ("no_domain_rand", "domain_rand=off", cfg.n_seeds),
    ):
        group = summary.get(group_key)
        if group is None:
            continue
        agg = group["aggregate"]
        n = int(agg.get("eval/num_episodes", 0))
        funnel = " / ".join(
            f"{stage}={agg.get(f'eval/funnel/{stage}_rate', 0.0) * 100:.0f}%"
            for stage in FUNNEL_STAGES
        )
        logging.info(
            "%s (n_seeds=%d, k=%d, episodes=%d): %s",
            label, n_seeds, cfg.k_repeats, n, funnel,
        )
        between = " / ".join(
            f"{stage}={group['between_seed_std'][stage] * 100:.0f}pp"
            for stage in FUNNEL_STAGES
        )
        within = " / ".join(
            f"{stage}={group['within_seed_std_mean'][stage] * 100:.0f}pp"
            for stage in FUNNEL_STAGES
        )
        logging.info("  between-seed std: %s", between)
        logging.info("  within-seed  std: %s (repeat/policy variance)", within)
        modes = " / ".join(
            f"{mode}={count / max(n, 1) * 100:.0f}%"
            for mode, count in group["failure_modes"].items()
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
