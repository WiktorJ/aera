#!/usr/bin/env python3
"""Multi-seed, multi-repeat eval for a single checkpoint, to quantify eval variance.

Manual testing (training_journal/06.07.2026/NOTES.md) found that funnel numbers
swing a lot even for the *same* seed on the *same* checkpoint, and that the
handful of seeds used in manual spot checks don't reproduce the eval-during-
training numbers. This script runs a structured multi-seed suite per checkpoint
so the resulting funnel is trustworthy enough to compare checkpoints against
each other.

The suite itself (seed/repeat grid, env-per-seed DR semantics, summaries) lives
in aera.autonomous.openpi.eval.suite and is shared with the decoupled
eval_worker — with identical defaults, so by default this runs the *same*
suite (same scenarios) as the training-time eval; raise the seed/repeat counts
for bigger offline deep dives. This script is just the offline CLI: load one
checkpoint, run the suite, write summary.json + episodes.jsonl.

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

import tyro

import openpi.policies.policy_config as _policy_config

import aera.autonomous.openpi.training_config as _training_config
from aera.autonomous.openpi.eval import suite as _suite
from aera.autonomous.openpi.scripts.run_policy_on_env import _find_model_path


@dataclasses.dataclass
class Args:
    """Arguments for the multi-seed eval-variance suite.

    Suite-shape and rollout fields mirror SuiteConfig (kept flat so the CLI
    flags stay `--n-substeps` etc.) and take their defaults from it, so the
    offline eval and the training-time eval_worker share one canonical suite.
    Rollout defaults match the *verified-correct* manual eval command from
    training_journal/06.07.2026/NOTES.md. Always double check n_substeps
    against the checkpoint's dataset `--skip`.
    """

    # Training config name (used to rebuild the model to load the checkpoint).
    config: str
    # Path to one per-step checkpoint dir (e.g. .../checkpoints/<exp>/50000).
    checkpoint_dir: str

    # --- Seed suite (defaults shared with eval_worker via SuiteConfig) ---
    n_dr_seeds: int = _suite.SuiteConfig.n_dr_seeds
    n_seeds: int = _suite.SuiteConfig.n_seeds
    k_repeats: int = _suite.SuiteConfig.k_repeats
    dr_seed_start: int = _suite.SuiteConfig.dr_seed_start
    seed_start: int = _suite.SuiteConfig.seed_start

    # --- Rollout parameters ---
    prompt: str = _suite.SuiteConfig.prompt
    max_episode_steps: int = _suite.SuiteConfig.max_episode_steps
    replan_steps: int = _suite.SuiteConfig.replan_steps
    n_substeps: int = _suite.SuiteConfig.n_substeps
    kinematic_grasp: bool = _suite.SuiteConfig.kinematic_grasp

    # --- Output ---
    out_dir: str = "eval_results"
    save_videos: bool = False
    video_out_path: str = "data/ar4_mk3/eval_variance_videos"

    def suite_config(self) -> _suite.SuiteConfig:
        fields = {f.name for f in dataclasses.fields(_suite.SuiteConfig)}
        return _suite.SuiteConfig(
            **{k: v for k, v in dataclasses.asdict(self).items() if k in fields}
        )


def run_eval(args: Args) -> dict:
    logging.basicConfig(level=logging.INFO)
    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    step = ckpt_dir.name if ckpt_dir.name.isdigit() else None
    suite_cfg = args.suite_config()

    config = _training_config.get_config(args.config).base_config
    policy = _policy_config.create_trained_policy(
        config, ckpt_dir, default_prompt=args.prompt
    )

    model_path = _find_model_path()
    if model_path is None:
        raise FileNotFoundError("Could not find AR4 MK3 scene.xml model file.")

    start = time.time()
    records = _suite.run_suite(suite_cfg, policy, model_path)
    elapsed = time.time() - start

    summary = _suite.summarize(records, suite_cfg)
    logging.info("Finished checkpoint %s in %.1fs", ckpt_dir, elapsed)
    _suite.log_summary(summary, suite_cfg)

    result = {
        "checkpoint_dir": str(ckpt_dir),
        "step": step,
        "config": args.config,
        "n_dr_seeds": args.n_dr_seeds,
        "n_seeds": args.n_seeds,
        "k_repeats": args.k_repeats,
        "elapsed_s": elapsed,
        **summary,
    }

    out_dir = pathlib.Path(args.out_dir) / (step or ckpt_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    _suite.write_episodes_jsonl(records, out_dir / "episodes.jsonl")
    logging.info("Wrote %s", out_dir)

    return result


if __name__ == "__main__":
    run_eval(tyro.cli(Args))
