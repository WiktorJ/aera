#!/usr/bin/env python3
"""Decoupled eval worker.

Polls a checkpoint base directory, and for every new per-step checkpoint it
loads the policy in-process, runs a fixed eval suite, and logs the granular
funnel + scalar metrics to the *same* mlflow run as the training job (at the
checkpoint's step). Keeping eval in a separate process means the training loop
is never slowed by the (expensive) MuJoCo rollouts, and the eval can sit on a
different GPU/host.

Architecture note: the eval itself is the shared suite in
aera.autonomous.openpi.eval.suite — the *same* {DR on x seeds, DR off x seeds}
x K-repeats grid, with the same defaults, as the offline eval_variance script
— so training-time curves and offline deep-dives are directly comparable (an
offline run at defaults reproduces the training-time suite exactly). The
trained `openpi` `Policy` object exposes the same
`.infer(obs)["actions"]` interface as the websocket client, so it is passed
straight into the suite's rollout. Per checkpoint, the flattened summary
scalars go to mlflow metrics (eval/dr/... and eval/nodr/...) and the raw
per-episode records (episodes.jsonl + summary.json) are attached as run
artifacts under eval/<step>/.

Example:
    python aera/autonomous/openpi/scripts/eval_worker.py \
        --config pi0_fast_ar4_mk3_low_mem_finetune \
        --checkpoint-base-dir checkpoints/ar4_mk3/my_exp \
        --n-substeps 3
"""

import dataclasses
import json
import logging
import pathlib
import tempfile
import time

import mlflow
import tyro

import openpi.policies.policy_config as _policy_config

import aera.autonomous.openpi.training_config as _training_config
from aera.autonomous.openpi.eval import suite as _suite
from aera.autonomous.openpi.scripts.run_policy_on_env import _find_model_path
from aera.autonomous.openpi.scripts.train import _maybe_override_checkpoint_dir

_EVALUATED_FILE = "evaluated_steps.txt"
_RUN_ID_FILE = "mlflow_run_id.txt"


@dataclasses.dataclass
class WorkerArgs:
    """Arguments for the eval worker."""

    # Training config name (used to rebuild the model + data config for loading).
    config: str
    # The training run's --exp-name. Together with `config` this resolves to the
    # exact checkpoint dir train.py writes to (honouring AERA_CHECKPOINT_DIR), so
    # the worker watches the right place and reads the right mlflow run id without
    # any hand-passed paths. Ignored if `checkpoint_dir` is given explicitly.
    exp_name: str | None = None
    # Explicit override for the dir holding per-step checkpoint subdirs +
    # mlflow_run_id.txt. Usually unset (resolved from config + exp_name).
    checkpoint_dir: str | None = None

    # mlflow tracking server. Defaults to the MLFLOW_TRACKING_URI env var (set in
    # the training container), falling back to mlflow's own default.
    mlflow_tracking_uri: str | None = None
    # mlflow run to log into. Defaults to reading `<checkpoint_dir>/mlflow_run_id.txt`,
    # which the training job writes at startup.
    mlflow_run_id: str | None = None

    # --- Eval suite (fixed across checkpoints so curves are comparable) ---
    # Defaults come straight from SuiteConfig — the one canonical suite shared
    # with eval_variance.py (15 DR seeds x 2 + 10 no-DR seeds x 2 = 50
    # episodes, seed starts at 1000) — so training-time and offline evals run
    # the same scenarios by default. Rollout defaults match the
    # verified-correct manual eval command from
    # training_journal/06.07.2026/NOTES.md: a prior run launched without
    # EVAL_ARGS silently used n_substeps=20 against a skip=3 dataset (~6.7x too
    # fast per policy step), making that run's on-training eval curve
    # meaningless. Always double check n_substeps against the checkpoint's
    # dataset `--skip`.
    n_dr_seeds: int = _suite.SuiteConfig.n_dr_seeds
    n_seeds: int = _suite.SuiteConfig.n_seeds
    k_repeats: int = _suite.SuiteConfig.k_repeats
    dr_seed_start: int = _suite.SuiteConfig.dr_seed_start
    seed_start: int = _suite.SuiteConfig.seed_start
    prompt: str = _suite.SuiteConfig.prompt
    max_episode_steps: int = _suite.SuiteConfig.max_episode_steps
    replan_steps: int = _suite.SuiteConfig.replan_steps
    # mj-steps per env.step. MUST match the dataset `--skip` the checkpoint was
    # trained on (see run_policy_on_env.Args.n_substeps), else the arm moves at
    # the wrong rate and eval understates the policy.
    n_substeps: int = _suite.SuiteConfig.n_substeps
    kinematic_grasp: bool = _suite.SuiteConfig.kinematic_grasp

    # --- Polling ---
    poll_interval_s: float = 60.0
    once: bool = False  # eval all currently-present checkpoints, then exit
    save_videos: bool = False
    video_out_path: str = "data/ar4_mk3/eval_worker_videos"

    def suite_config(self) -> _suite.SuiteConfig:
        fields = {f.name for f in dataclasses.fields(_suite.SuiteConfig)}
        return _suite.SuiteConfig(
            **{k: v for k, v in dataclasses.asdict(self).items() if k in fields}
        )


def _discover_checkpoints(base: pathlib.Path) -> list[int]:
    """Return committed checkpoint steps: digit-named subdirs containing `params`."""
    if not base.exists():
        return []
    steps = []
    for child in base.iterdir():
        if child.is_dir() and child.name.isdigit() and (child / "params").exists():
            steps.append(int(child.name))
    return sorted(steps)


def _resolve_checkpoint_dir(args: WorkerArgs) -> pathlib.Path:
    """Resolve the dir train.py writes to, from an explicit path or config+exp_name.

    Mirrors train.py exactly (config.name / exp_name under the
    AERA_CHECKPOINT_DIR-overridden base) so the worker watches the same place."""
    if args.checkpoint_dir:
        return pathlib.Path(args.checkpoint_dir)
    if not args.exp_name:
        raise ValueError("Provide either --checkpoint-dir or --exp-name.")
    # get_config returns the ExtendedTrainConfig wrapper; exp_name and the
    # checkpoint dir live on its base_config (same path train.py uses).
    base_config = _training_config.get_config(args.config).base_config
    base_config = dataclasses.replace(base_config, exp_name=args.exp_name)
    base_config = _maybe_override_checkpoint_dir(base_config)
    return base_config.checkpoint_dir


def _resolve_run_id(args: WorkerArgs, base: pathlib.Path) -> str:
    if args.mlflow_run_id:
        return args.mlflow_run_id
    run_id_path = base / _RUN_ID_FILE
    if not run_id_path.exists():
        raise FileNotFoundError(
            f"--mlflow-run-id not given and {run_id_path} does not exist. "
            "Start training first (it writes the run id) or pass the id explicitly."
        )
    return run_id_path.read_text().strip()


def _load_evaluated(base: pathlib.Path) -> set[int]:
    marker = base / _EVALUATED_FILE
    if not marker.exists():
        return set()
    return {int(line) for line in marker.read_text().split() if line.strip().isdigit()}


def _mark_evaluated(base: pathlib.Path, step: int) -> None:
    with (base / _EVALUATED_FILE).open("a") as f:
        f.write(f"{step}\n")


def _log_worker_args(client: mlflow.tracking.MlflowClient, run_id: str, args: WorkerArgs) -> None:
    """Log the resolved eval args to mlflow so a run's eval curve is always
    traceable to exactly what settings produced it (e.g. whether n_substeps
    matched the dataset's --skip). Without this, EVAL_ARGS (the shell env var
    used to override these at launch) leaves no record anywhere once the
    process exits -- which previously made a wrong n_substeps unrecoverable."""
    for key, value in dataclasses.asdict(args).items():
        try:
            client.log_param(run_id, f"eval_worker.{key}", value)
        except mlflow.exceptions.MlflowException:
            # Params are immutable once logged; a restarted worker re-logging
            # the same run's args should just no-op rather than crash.
            logging.debug("eval_worker.%s already logged for run %s", key, run_id)


def _eval_checkpoint(
    args: WorkerArgs, run_id: str, step: int, ckpt_dir: pathlib.Path
) -> None:
    logging.info(f"Evaluating checkpoint step {step} at {ckpt_dir}")
    config = _training_config.get_config(args.config).base_config
    policy = _policy_config.create_trained_policy(
        config, ckpt_dir, default_prompt=args.prompt
    )

    model_path = _find_model_path()
    if model_path is None:
        raise FileNotFoundError("Could not find AR4 MK3 scene.xml model file.")

    suite_cfg = args.suite_config()
    records = _suite.run_suite(suite_cfg, policy, model_path)
    summary = _suite.summarize(records, suite_cfg)
    _suite.log_summary(summary, suite_cfg)

    flat = _suite.flatten_for_mlflow(summary)
    client = mlflow.tracking.MlflowClient()
    for key, value in flat.items():
        client.log_metric(run_id, key, value, step=step)

    # Attach the raw per-episode records (incl. per-attempt / per-release
    # failure-mode events, which don't fit scalar metrics) to the run, so
    # post-hoc analysis never needs to re-run rollouts.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = pathlib.Path(tmp)
        _suite.write_episodes_jsonl(records, tmp_dir / "episodes.jsonl")
        (tmp_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        client.log_artifacts(run_id, tmp, artifact_path=f"eval/{step}")

    logging.info(
        "Logged step %d: success=%.1f%% grasped=%.1f%% "
        "(dr success=%.1f%% | nodr success=%.1f%%)",
        step,
        flat.get("eval/success_rate", 0.0) * 100,
        flat.get("eval/funnel/grasped_rate", 0.0) * 100,
        flat.get("eval/dr/success_rate", 0.0) * 100,
        flat.get("eval/nodr/success_rate", 0.0) * 100,
    )


def run_worker(args: WorkerArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    base = _resolve_checkpoint_dir(args)
    logging.info(f"Eval worker watching {base}")

    # When co-launched with training, the run-id file may not exist yet. Wait for
    # it (unless an explicit id was passed) so the worker can start in the same
    # command as training without a race.
    if not args.mlflow_run_id:
        run_id_path = base / _RUN_ID_FILE
        while not run_id_path.exists():
            logging.info(f"Waiting for {run_id_path} (training to start)...")
            time.sleep(args.poll_interval_s)

    # Fail fast: the run must exist in the resolved tracking store, otherwise the
    # metrics would silently go nowhere (e.g. wrong tracking URI / file store CWD).
    run_id = _resolve_run_id(args, base)
    client = mlflow.tracking.MlflowClient()
    client.get_run(run_id)  # raises if the run id is unknown to this store
    logging.info(
        f"Eval worker logging to mlflow run {run_id} "
        f"(tracking_uri={mlflow.get_tracking_uri()})"
    )
    _log_worker_args(client, run_id, args)

    evaluated = _load_evaluated(base)
    while True:
        pending = [s for s in _discover_checkpoints(base) if s not in evaluated]
        for step in pending:
            try:
                _eval_checkpoint(args, run_id, step, base / str(step))
                _mark_evaluated(base, step)
                evaluated.add(step)
            except Exception:
                # Don't mark done on failure (e.g. a half-written checkpoint);
                # it will be retried on the next poll.
                logging.exception(f"Eval failed for step {step}; will retry")

        if args.once:
            break
        time.sleep(args.poll_interval_s)


if __name__ == "__main__":
    run_worker(tyro.cli(WorkerArgs))
