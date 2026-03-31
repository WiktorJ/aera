#!/usr/bin/env python3
"""
IK parameter sweep script for finding sensible noise ranges.

Runs one-parameter-at-a-time sweeps over IK solver config values, holding all
other parameters at their defaults. For each value, N pick-and-place trials are
executed with domain randomization. Results are printed as a summary table and
written to CSV.

When --baseline-trials > 0 (default 50), the script also:
  - Runs a baseline block with the current IKConfig defaults before the sweep.
  - After the sweep, builds a "new defaults" config using the best value found
    for each swept parameter, and runs the same number of trials.
  - Prints a comparison summary and includes both blocks in the CSV.

Outputs are written to a data/ subdirectory by default (git-ignored).

Usage:
    # Sweep all parameters defined in my_grid.json, 5 trials each
    python sweep_ik_params.py --grid-path my_grid.json

    # Sweep a single parameter with 2 trials (quick sanity check, no baseline)
    python sweep_ik_params.py --grid-path my_grid.json --params pos_gain --trials-per-config 2 --baseline-trials 0

    # Custom output directory
    python sweep_ik_params.py --grid-path my_grid.json --csv-path /tmp/sweep.csv --json-path /tmp/sweep.json
"""

import csv
import dataclasses
import itertools
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tqdm import tqdm

import numpy as np
import tyro
from geometry_msgs.msg import Point, Pose, Quaternion

from aera.autonomous.envs.ar4_mk3_config import Ar4Mk3EnvConfig
from aera.autonomous.envs.ar4_mk3_pick_and_place import Ar4Mk3PickAndPlaceEnv
from aera_semi_autonomous.control.ar4_mk3_interface_config import (
    Ar4Mk3InterfaceConfig,
    IKConfig,
)
from aera_semi_autonomous.control.ar4_mk3_robot_interface import Ar4Mk3RobotInterface
from aera_semi_autonomous.data.domain_rand_config_generator import (
    generate_random_domain_rand_config,
)
from aera_semi_autonomous.data.pick_and_place_helpers import get_object_pose
from aera_semi_autonomous.data.trajectory_perturbation import (
    IKNoisePerturbation,
    perturb_ik_config,
)

T = np.array([0.6233588611899381, 0.05979687559388906, 0.7537742046170788])
Q = np.array(
    [
        -0.36336720179946663,
        -0.8203835174702869,
        0.22865474664402222,
        0.37769321910336584,
    ]
)

# Parameters that are allowed to be swept. Sweep values must be provided via --grid-path.
SWEEP_PARAMS: set = {
    "pos_gain",
    "orientation_gain",
    "integration_dt",
    "max_update_norm",
    "regularization_strength",
    "joints_update_scaling_0",
    "joints_update_scaling_1",
    "joints_update_scaling_2",
    "joints_update_scaling_3",
    "joints_update_scaling_4",
    "joints_update_scaling_5",
}

_DEFAULT_DATA_DIR = "data"


class _Tee:
    """Mirrors writes to both stdout and a file."""

    def __init__(self, path: str) -> None:
        self._file = open(path, "w")

    def write(self, msg: str) -> None:
        sys.__stdout__.write(msg)
        self._file.write(msg)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


@dataclass
class SweepConfig:
    trials_per_config: int = 5
    """Number of pick-and-place trials per (parameter, value) combination."""
    baseline_trials: int = 50
    """Trials to run with current defaults before sweep and with new best values after.
    Set to 0 to disable baseline comparison."""
    render: bool = False
    """Enable MuJoCo rendering (slow)."""
    seed: int = 42
    """Base random seed. Trial i uses seed + i for reproducibility."""
    params: List[str] = field(default_factory=list)
    """Parameters to sweep. Empty list = sweep all. E.g. --params pos_gain integration_dt"""
    csv_path: str = f"{_DEFAULT_DATA_DIR}/ik_sweep_results.csv"
    """Path for the CSV output file."""
    grid_path: Optional[str] = None
    """Path to a JSON file defining the sweep grid. Overrides the hardcoded SWEEP_GRID when provided."""
    summary_path: str = f"{_DEFAULT_DATA_DIR}/ik_sweep_summary.txt"
    """Path for the human-readable summary file (mirrors stdout)."""
    json_path: Optional[str] = f"{_DEFAULT_DATA_DIR}/ik_sweep_summary.json"
    """Path for the machine-readable JSON summary. Set to empty string to disable."""
    grid_search: bool = False
    """Run a full grid search over all (param, value) combinations instead of one-at-a-time sweeps.
    Only practical with 2-3 params × 3-4 values each. Use --params and --grid-path to keep it small."""
    noise_sweep: bool = False
    """Run noise tolerance sweep: for each fraction in --noise-fractions, run N trials with
    multiplicative IK noise at that level, then compare against a baseline block."""
    noise_fractions: List[float] = field(default_factory=list)
    """List of global noise fractions to test (e.g. --noise-fractions 0.05 0.1 0.15 0.2).
    Each fraction is applied uniformly to all IK parameters via IKNoisePerturbation.default_fraction."""
    debug: bool = False
    """Enable debug logging."""


@dataclass
class TrialResult:
    param_name: str
    param_value: float
    trial_idx: int
    pick_success: bool
    place_success: bool
    full_success: bool


def make_ik_config(param_name: str, value: float) -> IKConfig:
    """Return an IKConfig with one parameter overridden, all others at default."""
    base = IKConfig()
    if param_name.startswith("joints_update_scaling_"):
        idx = int(param_name.split("_")[-1])
        scaling = list(base.joints_update_scaling)
        scaling[idx] = value
        return dataclasses.replace(base, joints_update_scaling=scaling)
    return dataclasses.replace(base, **{param_name: value})


def best_value_for_param(param_name: str, results: List[TrialResult]) -> float:
    """Return the param value with the highest full-success rate among sweep results."""
    param_results = [r for r in results if r.param_name == param_name]
    values = sorted(set(r.param_value for r in param_results))
    best_val, best_rate = values[0], -1.0
    for v in values:
        trials = [r for r in param_results if r.param_value == v]
        rate = sum(r.full_success for r in trials) / len(trials)
        if rate > best_rate:
            best_rate, best_val = rate, v
    return best_val


def build_new_defaults_config(
    all_results: List[TrialResult], params_swept: List[str]
) -> IKConfig:
    """Build an IKConfig using the best value found for each swept parameter."""
    base = IKConfig()
    scaling = list(base.joints_update_scaling)
    kwargs = {}
    for param_name in params_swept:
        best = best_value_for_param(param_name, all_results)
        if param_name.startswith("joints_update_scaling_"):
            idx = int(param_name.split("_")[-1])
            scaling[idx] = best
        else:
            kwargs[param_name] = best
    return dataclasses.replace(base, joints_update_scaling=scaling, **kwargs)


def run_trial(
    model_path: str,
    ik_config: IKConfig,
    render: bool,
    logger: logging.Logger,
) -> TrialResult:
    """Run one pick-and-place episode and return action-level success flags."""
    env = None
    pick_success = False
    place_success = False
    try:
        domain_rand_config, _, _ = generate_random_domain_rand_config()
        env_config = Ar4Mk3EnvConfig(
            model_path=model_path,
            reward_type="sparse",
            use_eef_control=False,
            translation=T,
            quaterion=Q,
            distance_multiplier=1.2,
            z_offset=0.3,
            domain_rand=domain_rand_config,
        )
        env = Ar4Mk3PickAndPlaceEnv(
            render_mode="human" if render else None,
            config=env_config,
        )
        env.reset()

        interface_config = Ar4Mk3InterfaceConfig(render_steps=render, ik=ik_config)
        robot = Ar4Mk3RobotInterface(env, config=interface_config)

        if not robot.go_home():
            return TrialResult("", 0.0, 0, False, False, False)

        object_pose = get_object_pose(env, logger)
        if object_pose is None:
            return TrialResult("", 0.0, 0, False, False, False)

        pick_success = robot.grasp_at(object_pose, gripper_pos=0.0)

        if pick_success:
            target_pos = env.goal
            target_pose = Pose()
            target_pose.position = Point(
                x=float(target_pos[0]),
                y=float(target_pos[1]),
                z=float(target_pos[2] + object_pose.position.z),
            )
            target_pose.orientation = Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)
            place_success = robot.release_at(target_pose)

        robot.go_home()
    except Exception as e:
        logger.warning(f"Trial raised exception: {e}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return TrialResult(
        param_name="",
        param_value=0.0,
        trial_idx=0,
        pick_success=pick_success,
        place_success=place_success,
        full_success=pick_success and place_success,
    )


def run_named_block(
    label: str,
    ik_config: IKConfig,
    n_trials: int,
    seed: int,
    model_path: str,
    render: bool,
    logger: logging.Logger,
) -> List[TrialResult]:
    """Run N trials with a fixed IKConfig, tagging results with param_name=label."""
    results = []
    with tqdm(total=n_trials, desc=label, unit="trial", leave=True) as pbar:
        for i in range(n_trials):
            np.random.seed(seed + i)
            result = run_trial(model_path, ik_config, render, logger)
            result.param_name = label
            result.param_value = 0.0
            result.trial_idx = i
            results.append(result)
            n_ok = sum(r.full_success for r in results)
            pbar.set_postfix(success=f"{n_ok}/{i + 1}")
            pbar.update(1)
    return results


def run_sweep(
    param_name: str,
    values: list,
    cfg: SweepConfig,
    model_path: str,
    logger: logging.Logger,
) -> List[TrialResult]:
    results = []
    total = len(values) * cfg.trials_per_config
    with tqdm(total=total, desc=param_name, unit="trial", leave=True) as pbar:
        for value in values:
            ik_config = make_ik_config(param_name, value)
            value_results = []
            for trial_idx in range(cfg.trials_per_config):
                np.random.seed(cfg.seed + trial_idx)
                result = run_trial(model_path, ik_config, cfg.render, logger)
                result.param_name = param_name
                result.param_value = value
                result.trial_idx = trial_idx
                results.append(result)
                value_results.append(result)
                n_ok = sum(r.full_success for r in value_results)
                pbar.set_postfix(
                    value=f"{value:.4g}", success=f"{n_ok}/{len(value_results)}"
                )
                pbar.update(1)
    return results


# ---------------------------------------------------------------------------
# JSON summary builders
# ---------------------------------------------------------------------------


def build_sweep_summary(
    param_name: str, values: list, results: List[TrialResult]
) -> dict:
    """Build a JSON-serializable dict summarising one parameter's sweep."""
    if param_name.startswith("joints_update_scaling_"):
        idx = int(param_name.split("_")[-1])
        default_value = IKConfig().joints_update_scaling[idx]
    else:
        default_value = getattr(IKConfig(), param_name)

    value_rows = []
    for value in values:
        trial_results = [r for r in results if r.param_value == value]
        n = len(trial_results)
        pick_ok = sum(r.pick_success for r in trial_results)
        place_ok = sum(r.place_success for r in trial_results)
        full_ok = sum(r.full_success for r in trial_results)
        value_rows.append(
            {
                "value": value,
                "trials": n,
                "pick_ok": pick_ok,
                "place_ok": place_ok,
                "full_ok": full_ok,
                "success_pct": round(100.0 * full_ok / n, 2) if n > 0 else 0.0,
                "is_default": abs(value - default_value) < 1e-9,
            }
        )

    return {
        "param": param_name,
        "default_value": default_value,
        "values": value_rows,
    }


def build_comparison_summary(
    baseline: List[TrialResult],
    new_defaults: List[TrialResult],
    new_config: IKConfig,
) -> dict:
    """Build a JSON-serializable dict for the baseline vs new-defaults comparison."""

    def stats(results: List[TrialResult]) -> dict:
        n = len(results)
        pick_ok = sum(r.pick_success for r in results)
        place_ok = sum(r.place_success for r in results)
        full_ok = sum(r.full_success for r in results)
        return {
            "trials": n,
            "pick_ok": pick_ok,
            "place_ok": place_ok,
            "full_ok": full_ok,
            "success_pct": round(100.0 * full_ok / n, 2) if n else 0.0,
        }

    baseline_stats = stats(baseline)
    new_stats = stats(new_defaults)
    return {
        "baseline": baseline_stats,
        "new_defaults": new_stats,
        "delta_pct": round(new_stats["success_pct"] - baseline_stats["success_pct"], 2),
        "new_config": dataclasses.asdict(new_config),
    }


def build_grid_search_summary(results: List[TrialResult]) -> dict:
    """Build a JSON-serializable dict for grid-search results."""
    labels = sorted(
        set(r.param_name for r in results),
        key=lambda l: min(r.param_value for r in results if r.param_name == l),
    )
    rows = []
    for label in labels:
        trial_results = [r for r in results if r.param_name == label]
        n = len(trial_results)
        full_ok = sum(r.full_success for r in trial_results)
        rows.append(
            {
                "config": label,
                "trials": n,
                "full_ok": full_ok,
                "success_pct": round(100.0 * full_ok / n, 2),
            }
        )
    rows.sort(key=lambda r: r["success_pct"], reverse=True)
    return {"configs": rows}


# ---------------------------------------------------------------------------
# Human-friendly printers (derived from the JSON dicts)
# ---------------------------------------------------------------------------


def print_sweep_summary(data: dict) -> None:
    param_name = data["param"]
    default_value = data["default_value"]
    print(f"\n=== Sweeping {param_name} (default={default_value}) ===")
    header = f"  {'Value':>10} | {'Trials':>6} | {'Pick OK':>7} | {'Place OK':>8} | {'Full OK':>7} | {'Success%':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in data["values"]:
        marker = "  <- default" if row["is_default"] else ""
        print(
            f"  {row['value']:>10.4g} | {row['trials']:>6} | {row['pick_ok']:>7} | "
            f"{row['place_ok']:>8} | {row['full_ok']:>7} | {row['success_pct']:>8.1f}%{marker}"
        )


def print_comparison_summary(data: dict) -> None:
    b = data["baseline"]
    n = data["new_defaults"]
    delta = data["delta_pct"]

    print("\n" + "=" * 62)
    print("=== Baseline vs. New Defaults Comparison ===")
    print("=" * 62)
    print(
        f"  {'Config':<20} | {'Trials':>6} | {'Pick OK':>7} | {'Place OK':>8} | {'Full OK':>7} | {'Success%':>9}"
    )
    print("  " + "-" * 60)
    print(
        f"  {'current defaults':<20} | {b['trials']:>6} | {b['pick_ok']:>7} | "
        f"{b['place_ok']:>8} | {b['full_ok']:>7} | {b['success_pct']:>8.1f}%"
    )
    print(
        f"  {'new best values':<20} | {n['trials']:>6} | {n['pick_ok']:>7} | "
        f"{n['place_ok']:>8} | {n['full_ok']:>7} | {n['success_pct']:>8.1f}%"
    )
    print("  " + "-" * 60)
    print(f"  Delta: {delta:+.1f}%")
    print(f"\n  New config: {data['new_config']}")
    print("=" * 62)


def print_grid_search_summary(data: dict) -> None:
    rows = data["configs"]
    print("\n" + "=" * 80)
    print("=== Grid Search Results (sorted by success rate) ===")
    print("=" * 80)
    print(f"  {'Configuration':<50} | {'Trials':>6} | {'Full OK':>7} | {'Success%':>9}")
    print("  " + "-" * 78)
    for row in rows:
        print(
            f"  {row['config']:<50} | {row['trials']:>6} | {row['full_ok']:>7} | "
            f"{row['success_pct']:>8.1f}%"
        )
    print("=" * 80)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_csv(results: List[TrialResult], csv_path: str) -> None:
    _ensure_dir(csv_path)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "param_name",
                "param_value",
                "trial_idx",
                "pick_success",
                "place_success",
                "full_success",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    print(f"\nResults written to {csv_path}")


def write_json_summary(summary: dict, json_path: str) -> None:
    _ensure_dir(json_path)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary written to {json_path}")


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def find_model_path(script_dir: str) -> Optional[str]:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    candidates = [
        os.path.join(
            project_root,
            "aera",
            "autonomous",
            "simulation",
            "mujoco",
            "ar4_mk3",
            "scene.xml",
        ),
        "aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml",
        "../aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml",
        "../../aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml",
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def make_ik_config_multi(overrides: Dict[str, float]) -> IKConfig:
    """Return an IKConfig with multiple parameters overridden."""
    base = IKConfig()
    scaling = list(base.joints_update_scaling)
    kwargs = {}
    for param_name, value in overrides.items():
        if param_name.startswith("joints_update_scaling_"):
            idx = int(param_name.split("_")[-1])
            scaling[idx] = value
        else:
            kwargs[param_name] = value
    return dataclasses.replace(base, joints_update_scaling=scaling, **kwargs)


def run_grid_search(
    param_grid: Dict[str, List[float]],
    cfg: SweepConfig,
    model_path: str,
    logger: logging.Logger,
) -> List[TrialResult]:
    """Run a full grid search over all combinations of param values."""
    param_names = list(param_grid.keys())
    value_lists = [param_grid[p] for p in param_names]
    combos = list(itertools.product(*value_lists))
    total = len(combos) * cfg.trials_per_config
    logger.info(
        f"Grid search: {len(combos)} combinations × {cfg.trials_per_config} trials = {total} total"
    )

    all_results: List[TrialResult] = []
    with tqdm(total=total, desc="grid search", unit="trial", leave=True) as pbar:
        for combo_idx, values in enumerate(combos):
            overrides = dict(zip(param_names, values))
            ik_config = make_ik_config_multi(overrides)
            label = " | ".join(f"{p}={v}" for p, v in overrides.items())

            combo_results = []
            for trial_idx in range(cfg.trials_per_config):
                np.random.seed(cfg.seed + trial_idx)
                result = run_trial(model_path, ik_config, cfg.render, logger)
                result.param_name = label
                result.param_value = combo_idx
                result.trial_idx = trial_idx
                combo_results.append(result)
                pbar.set_postfix(combo=f"{combo_idx + 1}/{len(combos)}")
                pbar.update(1)
            all_results.extend(combo_results)

            n = len(combo_results)
            full_ok = sum(r.full_success for r in combo_results)
            pct = 100.0 * full_ok / n
            logger.info(f"  [{combo_idx + 1}/{len(combos)}] {label} → {pct:.0f}%")

    return all_results


def run_noise_block(
    noise_fraction: float,
    n_trials: int,
    seed: int,
    model_path: str,
    render: bool,
    logger: logging.Logger,
) -> List[TrialResult]:
    """Run N trials where each trial gets a fresh noisy IKConfig.

    On every trial the base IKConfig is perturbed with multiplicative noise
    at the given global fraction, so each trial sees a different IK config.
    """
    noise_cfg = IKNoisePerturbation(default_fraction=noise_fraction)
    label = f"noise_{noise_fraction:.3f}"
    results = []
    with tqdm(total=n_trials, desc=label, unit="trial", leave=True) as pbar:
        for i in range(n_trials):
            np.random.seed(seed + i)
            ik_config = perturb_ik_config(IKConfig(), noise_cfg)
            result = run_trial(model_path, ik_config, render, logger)
            result.param_name = label
            result.param_value = noise_fraction
            result.trial_idx = i
            results.append(result)
            n_ok = sum(r.full_success for r in results)
            pbar.set_postfix(success=f"{n_ok}/{i + 1}")
            pbar.update(1)
    return results


def build_noise_sweep_summary(
    baseline: List[TrialResult],
    noise_blocks: Dict[float, List[TrialResult]],
) -> dict:
    """Build a JSON-serializable summary for the noise tolerance sweep."""

    def stats(results: List[TrialResult]) -> dict:
        n = len(results)
        pick_ok = sum(r.pick_success for r in results)
        place_ok = sum(r.place_success for r in results)
        full_ok = sum(r.full_success for r in results)
        return {
            "trials": n,
            "pick_ok": pick_ok,
            "place_ok": place_ok,
            "full_ok": full_ok,
            "success_pct": round(100.0 * full_ok / n, 2) if n else 0.0,
        }

    baseline_stats = stats(baseline)
    rows = []
    for frac in sorted(noise_blocks.keys()):
        s = stats(noise_blocks[frac])
        s["noise_fraction"] = frac
        s["delta_pct"] = round(s["success_pct"] - baseline_stats["success_pct"], 2)
        rows.append(s)

    return {"baseline": baseline_stats, "noise_levels": rows}


def print_noise_sweep_summary(data: dict) -> None:
    b = data["baseline"]
    print("\n" + "=" * 80)
    print("=== Noise Tolerance Sweep Results ===")
    print("=" * 80)
    print(
        f"  {'Noise %':<10} | {'Trials':>6} | {'Pick OK':>7} | {'Place OK':>8} "
        f"| {'Full OK':>7} | {'Success%':>9} | {'Delta':>7}"
    )
    print("  " + "-" * 78)
    print(
        f"  {'baseline':<10} | {b['trials']:>6} | {b['pick_ok']:>7} | "
        f"{b['place_ok']:>8} | {b['full_ok']:>7} | {b['success_pct']:>8.1f}% | {'—':>7}"
    )
    for row in data["noise_levels"]:
        pct_label = f"{row['noise_fraction'] * 100:.1f}%"
        print(
            f"  {pct_label:<10} | {row['trials']:>6} | {row['pick_ok']:>7} | "
            f"{row['place_ok']:>8} | {row['full_ok']:>7} | {row['success_pct']:>8.1f}% "
            f"| {row['delta_pct']:>+6.1f}%"
        )
    print("=" * 80)


def main() -> None:
    cfg = tyro.cli(SweepConfig)

    level = logging.DEBUG if cfg.debug else logging.CRITICAL
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    _ensure_dir(cfg.summary_path)
    tee = _Tee(cfg.summary_path)
    sys.stdout = tee

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = find_model_path(script_dir)
    if model_path is None:
        logger.error("Could not find AR4 MK3 model file.")
        return

    all_results: List[TrialResult] = []
    json_summary: dict = {"sweeps": {}, "comparison": None, "grid_search": None, "noise_sweep": None}

    if cfg.noise_sweep:
        # --- Noise tolerance sweep mode ---
        if not cfg.noise_fractions:
            logger.error("--noise-fractions is required for --noise-sweep mode.")
            return
        if cfg.baseline_trials <= 0:
            logger.error("--baseline-trials must be > 0 for --noise-sweep mode.")
            return

        baseline_results = run_named_block(
            "__baseline__",
            IKConfig(),
            cfg.baseline_trials,
            cfg.seed,
            model_path,
            cfg.render,
            logger,
        )
        all_results.extend(baseline_results)

        noise_blocks: Dict[float, List[TrialResult]] = {}
        for frac in cfg.noise_fractions:
            block = run_noise_block(
                frac,
                cfg.baseline_trials,
                cfg.seed,
                model_path,
                cfg.render,
                logger,
            )
            noise_blocks[frac] = block
            all_results.extend(block)

        ns_data = build_noise_sweep_summary(baseline_results, noise_blocks)
        json_summary["noise_sweep"] = ns_data
        print_noise_sweep_summary(ns_data)

    else:
        # --- Grid / one-at-a-time sweep modes (require --grid-path) ---
        if cfg.grid_path is None:
            logger.error(
                "--grid-path is required. Provide a JSON file mapping parameter names to sweep values."
            )
            return
        if not os.path.exists(cfg.grid_path):
            logger.error(f"Grid file not found: {cfg.grid_path}")
            return
        try:
            with open(cfg.grid_path) as f:
                grid = json.load(f)
            logger.info(f"Loaded sweep grid from {cfg.grid_path}")
        except Exception as e:
            logger.error(f"Failed to parse grid file: {e}")
            return

        params_to_sweep = cfg.params if cfg.params else list(grid.keys())
        unknown = [p for p in params_to_sweep if p not in SWEEP_PARAMS]
        if unknown:
            logger.error(f"Unknown parameters: {unknown}. Valid: {sorted(SWEEP_PARAMS)}")
            return
        missing = [p for p in params_to_sweep if p not in grid]
        if missing:
            logger.error(f"Parameters not in grid file: {missing}")
            return

        if cfg.grid_search:
            # --- Grid search mode ---
            param_grid = {p: grid[p] for p in params_to_sweep}
            combo_count = 1
            for v in param_grid.values():
                combo_count *= len(v)
            total_trials = combo_count * cfg.trials_per_config
            logger.info(
                f"Grid search: {combo_count} combos × {cfg.trials_per_config} trials = {total_trials} total"
            )

            baseline_results: List[TrialResult] = []
            if cfg.baseline_trials > 0:
                logger.info(
                    f"Running {cfg.baseline_trials} baseline trials with current defaults..."
                )
                baseline_results = run_named_block(
                    "__baseline__",
                    IKConfig(),
                    cfg.baseline_trials,
                    cfg.seed,
                    model_path,
                    cfg.render,
                    logger,
                )
                all_results.extend(baseline_results)
                n = len(baseline_results)
                full_ok = sum(r.full_success for r in baseline_results)
                logger.info(f"Baseline: {full_ok}/{n} ({100.0 * full_ok / n:.1f}%) success")

            grid_results = run_grid_search(param_grid, cfg, model_path, logger)
            all_results.extend(grid_results)

            gs_data = build_grid_search_summary(grid_results)
            json_summary["grid_search"] = gs_data
            print_grid_search_summary(gs_data)

            if cfg.baseline_trials > 0 and grid_results:
                best_row = gs_data["configs"][0]
                best_label = best_row["config"]
                overrides = {}
                for part in best_label.split(" | "):
                    k, v = part.split("=")
                    overrides[k] = float(v)
                best_config = make_ik_config_multi(overrides)
                logger.info(
                    f"Running {cfg.baseline_trials} trials with best grid combo: {best_label}"
                )
                new_defaults_results = run_named_block(
                    "__new_defaults__",
                    best_config,
                    cfg.baseline_trials,
                    cfg.seed,
                    model_path,
                    cfg.render,
                    logger,
                )
                all_results.extend(new_defaults_results)
                comp_data = build_comparison_summary(
                    baseline_results, new_defaults_results, best_config
                )
                json_summary["comparison"] = comp_data
                print_comparison_summary(comp_data)
        else:
            # --- One-at-a-time sweep mode ---
            baseline_results: List[TrialResult] = []
            if cfg.baseline_trials > 0:
                logger.info(
                    f"Running {cfg.baseline_trials} baseline trials with current defaults..."
                )
                baseline_results = run_named_block(
                    "__baseline__",
                    IKConfig(),
                    cfg.baseline_trials,
                    cfg.seed,
                    model_path,
                    cfg.render,
                    logger,
                )
                all_results.extend(baseline_results)
                n = len(baseline_results)
                full_ok = sum(r.full_success for r in baseline_results)
                logger.info(f"Baseline: {full_ok}/{n} ({100.0 * full_ok / n:.1f}%) success")

            for param_name in params_to_sweep:
                values = grid[param_name]
                logger.info(
                    f"Sweeping {param_name} over {values} "
                    f"({cfg.trials_per_config} trials each, {len(values) * cfg.trials_per_config} total)"
                )
                results = run_sweep(param_name, values, cfg, model_path, logger)
                all_results.extend(results)
                sweep_data = build_sweep_summary(param_name, values, results)
                json_summary["sweeps"][param_name] = sweep_data
                print_sweep_summary(sweep_data)

            if cfg.baseline_trials > 0:
                new_config = build_new_defaults_config(
                    [
                        r
                        for r in all_results
                        if r.param_name not in ("__baseline__", "__new_defaults__")
                    ],
                    params_to_sweep,
                )
                logger.info(
                    f"Running {cfg.baseline_trials} trials with new best-value defaults..."
                )
                new_defaults_results = run_named_block(
                    "__new_defaults__",
                    new_config,
                    cfg.baseline_trials,
                    cfg.seed,
                    model_path,
                    cfg.render,
                    logger,
                )
                all_results.extend(new_defaults_results)
                comp_data = build_comparison_summary(
                    baseline_results, new_defaults_results, new_config
                )
                json_summary["comparison"] = comp_data
                print_comparison_summary(comp_data)

    write_csv(all_results, cfg.csv_path)

    if cfg.json_path:
        write_json_summary(json_summary, cfg.json_path)

    sys.stdout = sys.__stdout__
    tee.close()
    logger.info(f"Summary written to {cfg.summary_path}")


if __name__ == "__main__":
    main()
