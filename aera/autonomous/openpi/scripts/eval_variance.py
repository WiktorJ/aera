#!/usr/bin/env python3
"""Multi-seed, multi-repeat eval for a single checkpoint, to quantify eval variance.

Manual testing (training_journal/06.07.2026/NOTES.md) found that funnel numbers
swing a lot even for the *same* seed on the *same* checkpoint, and that the
handful of seeds used in manual spot checks don't reproduce the eval-during-
training numbers. This script runs a much larger, structured suite per
checkpoint so the resulting funnel is trustworthy enough to compare
checkpoints against each other:

  - N_DR seeds with domain_rand on, N_S seeds with domain_rand off. Each seed
    fixes a scenario (spawn geometry via env.reset(seed=...), plus - for DR
    seeds - the sampled visual domain-rand config, since DR is baked in at env
    construction, not per-reset).
  - Each seed is rolled out K times with *no* change to the scenario, so
    across-repeat spread isolates policy/inference variance from
    across-seed spread, which is scenario variance.

Reuses the exact rollout (env construction, episode loop, metric tracking)
from run_policy_on_env.py / eval_worker.py, and loads the policy in-process
(like eval_worker.py) so this doesn't need a running policy server.

Example:
    uv run aera/autonomous/openpi/scripts/eval_variance.py \
        --config pi05_ar4_mk3 \
        --checkpoint-dir checkpoints/pi05_ar4_mk3/pi05_ar4_mk3_2026-07-04_16-51-56/50000 \
        --n-substeps 3
"""

import dataclasses
import json
import logging
import pathlib
import time

import numpy as np
import tyro

import openpi.policies.policy_config as _policy_config

import aera.autonomous.openpi.training_config as _training_config
from aera.autonomous.openpi.eval import metrics as _metrics
from aera.autonomous.openpi.scripts.run_policy_on_env import (
    Args as RolloutArgs,
    _build_env,
    _find_model_path,
    _resolve_prompts,
    _run_episode,
    _save_episode_video,
)

_FUNNEL_STAGES = ("reached", "grasped", "lifted", "transported", "placed")
_PROGRESS_SCALARS = ("reach_progress", "place_progress")


@dataclasses.dataclass
class Args:
    """Arguments for the multi-seed eval-variance suite."""

    # Training config name (used to rebuild the model to load the checkpoint).
    config: str
    # Path to one per-step checkpoint dir (e.g. .../checkpoints/<exp>/50000).
    checkpoint_dir: str

    # --- Seed suite ---
    n_dr_seeds: int = 30  # N_DR: seeds evaluated with domain_rand on
    n_seeds: int = 15  # N_S: seeds evaluated with domain_rand off
    k_repeats: int = 5  # K: rollouts per seed (isolates policy/inference variance)
    dr_seed_start: int = 0
    seed_start: int = 0

    # --- Rollout parameters ---
    # Defaults below match the *verified-correct* manual eval command from
    # training_journal/06.07.2026/NOTES.md, not eval_worker.py's defaults: the
    # decoupled eval worker for that run was launched without EVAL_ARGS, so it
    # silently ran with n_substeps=20 against a skip=3 dataset (~6.7x too fast
    # per policy step), making that run's on-training eval curve meaningless.
    # Always double check n_substeps against the checkpoint's dataset `--skip`.
    prompt: str = "pick the yellow block and place it on the red target"
    max_episode_steps: int = 1000
    replan_steps: int = 10
    # mj-steps per env.step. MUST match the dataset `--skip` the checkpoint was
    # trained on (see run_policy_on_env.Args.n_substeps).
    n_substeps: int = 3
    kinematic_grasp: bool = True

    # --- Output ---
    out_dir: str = "eval_results"
    save_videos: bool = False
    video_out_path: str = "data/ar4_mk3/eval_variance_videos"


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


def _rollout_args(args: Args, *, domain_rand: bool, seed: int) -> RolloutArgs:
    return RolloutArgs(
        prompt=args.prompt,
        replan_steps=args.replan_steps,
        num_episodes=1,
        max_episode_steps=args.max_episode_steps,
        domain_rand=domain_rand,
        headless=True,
        kinematic_grasp=args.kinematic_grasp,
        n_substeps=args.n_substeps,
        two_phase_prompt=False,
        seed=seed,
        video_out_path=args.video_out_path,
    )


def _run_seed_repeats(
    args: Args,
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
    # per seed, matching eval_worker.py's approach for its fixed suite.
    np.random.seed(seed)
    rollout_args = _rollout_args(args, domain_rand=domain_rand, seed=seed)
    pick_prompt, place_prompt, dr_config = _resolve_prompts(rollout_args)
    env = _build_env(rollout_args, model_path, dr_config)

    records = []
    try:
        for repeat in range(args.k_repeats):
            # episode_idx is always 0 so _run_episode's env.reset(seed=seed+0)
            # is identical every repeat -- only policy/inference can differ.
            ep, replay_images, final_prompt = _run_episode(
                rollout_args, env, policy, pick_prompt, place_prompt, 0, None
            )
            records.append(EpisodeRecord(seed=seed, repeat=repeat, domain_rand=domain_rand, metrics=ep))
            if args.save_videos:
                tag = "dr" if domain_rand else "nodr"
                _save_episode_video(
                    replay_images,
                    args.video_out_path,
                    episode_idx=f"{tag}_seed{seed}_rep{repeat}",
                    prompt=final_prompt,
                    success=ep.placed,
                )
            logging.info(
                "  [%s seed=%d rep=%d/%d] reached=%s grasped=%s transported=%s placed=%s",
                "dr" if domain_rand else "nodr",
                seed,
                repeat + 1,
                args.k_repeats,
                ep.reached,
                ep.grasped,
                ep.transported,
                ep.placed,
            )
    finally:
        env.close()
    return records


def _run_group(
    args: Args, policy, model_path: str, seeds: range, domain_rand: bool
) -> list[EpisodeRecord]:
    label = "domain_rand=on" if domain_rand else "domain_rand=off"
    records: list[EpisodeRecord] = []
    for i, seed in enumerate(seeds):
        logging.info("[%s] seed %d (%d/%d)", label, seed, i + 1, len(seeds))
        records.extend(_run_seed_repeats(args, policy, model_path, seed, domain_rand))
    return records


def _per_seed_stats(records: list[EpisodeRecord], seed: int) -> dict:
    eps = [r.metrics for r in records if r.seed == seed]
    out: dict = {"seed": seed, "n": len(eps)}
    for stage in _FUNNEL_STAGES:
        vals = [float(getattr(e, stage)) for e in eps]
        out[f"{stage}_rate"] = float(np.mean(vals))
        out[f"{stage}_std"] = float(np.std(vals))
    for name in _PROGRESS_SCALARS:
        vals = [float(getattr(e, name)) for e in eps]
        out[f"{name}_mean"] = float(np.mean(vals))
        out[f"{name}_std"] = float(np.std(vals))
    return out


def _group_summary(records: list[EpisodeRecord], seeds: range) -> dict:
    per_seed = [_per_seed_stats(records, s) for s in seeds]
    summary: dict = {
        "aggregate": _metrics.aggregate([r.metrics for r in records]),
        "per_seed": per_seed,
        # Between-seed spread: std, across seeds, of each seed's own mean rate.
        # This is scenario variance (different spawn geometry / DR draw).
        "between_seed_std": {},
        # Within-seed spread: mean, across seeds, of each seed's own std across
        # its K repeats. This is policy/inference variance at a fixed scenario.
        "within_seed_std_mean": {},
    }
    for stage in _FUNNEL_STAGES:
        summary["between_seed_std"][stage] = float(
            np.std([p[f"{stage}_rate"] for p in per_seed])
        )
        summary["within_seed_std_mean"][stage] = float(
            np.mean([p[f"{stage}_std"] for p in per_seed])
        )
    return summary


def _log_group_summary(summary: dict, label: str, n_seeds: int, k_repeats: int) -> None:
    agg = summary["aggregate"]
    funnel = " / ".join(
        f"{stage}={agg.get(f'eval/funnel/{stage}_rate', 0.0) * 100:.0f}%"
        for stage in _FUNNEL_STAGES
    )
    logging.info(
        "%s (n_seeds=%d, k=%d, episodes=%d): %s",
        label, n_seeds, k_repeats, int(agg.get("eval/num_episodes", 0)), funnel,
    )
    between = " / ".join(
        f"{stage}={summary['between_seed_std'][stage] * 100:.0f}pp"
        for stage in _FUNNEL_STAGES
    )
    within = " / ".join(
        f"{stage}={summary['within_seed_std_mean'][stage] * 100:.0f}pp"
        for stage in _FUNNEL_STAGES
    )
    logging.info("  between-seed std: %s", between)
    logging.info("  within-seed  std: %s (repeat/policy variance)", within)


def run_eval(args: Args) -> dict:
    logging.basicConfig(level=logging.INFO)
    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    step = ckpt_dir.name if ckpt_dir.name.isdigit() else None

    config = _training_config.get_config(args.config).base_config
    policy = _policy_config.create_trained_policy(config, ckpt_dir, default_prompt=args.prompt)

    model_path = _find_model_path()
    if model_path is None:
        raise FileNotFoundError("Could not find AR4 MK3 scene.xml model file.")

    dr_seeds = range(args.dr_seed_start, args.dr_seed_start + args.n_dr_seeds)
    nodr_seeds = range(args.seed_start, args.seed_start + args.n_seeds)

    start = time.time()
    dr_records = _run_group(args, policy, model_path, dr_seeds, domain_rand=True)
    nodr_records = _run_group(args, policy, model_path, nodr_seeds, domain_rand=False)
    elapsed = time.time() - start

    dr_summary = _group_summary(dr_records, dr_seeds)
    nodr_summary = _group_summary(nodr_records, nodr_seeds)

    logging.info("Finished checkpoint %s in %.1fs", ckpt_dir, elapsed)
    _log_group_summary(dr_summary, "domain_rand=on", args.n_dr_seeds, args.k_repeats)
    _log_group_summary(nodr_summary, "domain_rand=off", args.n_seeds, args.k_repeats)

    result = {
        "checkpoint_dir": str(ckpt_dir),
        "step": step,
        "config": args.config,
        "n_dr_seeds": args.n_dr_seeds,
        "n_seeds": args.n_seeds,
        "k_repeats": args.k_repeats,
        "elapsed_s": elapsed,
        "domain_rand": dr_summary,
        "no_domain_rand": nodr_summary,
    }

    out_dir = pathlib.Path(args.out_dir) / (step or ckpt_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    with (out_dir / "episodes.jsonl").open("w") as f:
        for r in [*dr_records, *nodr_records]:
            f.write(json.dumps(r.to_dict()) + "\n")
    logging.info("Wrote %s", out_dir)

    return result


if __name__ == "__main__":
    run_eval(tyro.cli(Args))
