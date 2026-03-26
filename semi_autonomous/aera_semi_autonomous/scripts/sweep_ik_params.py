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

Usage:
    # Sweep all parameters, 5 trials each, with 50-trial baseline comparison
    python sweep_ik_params.py

    # Sweep a single parameter with 2 trials (quick sanity check, no baseline)
    python sweep_ik_params.py --params pos_gain --trials-per-config 2 --baseline-trials 0

    # Custom CSV output path
    python sweep_ik_params.py --csv-path /tmp/sweep.csv
"""

import csv
import dataclasses
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

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

T = np.array([0.6233588611899381, 0.05979687559388906, 0.7537742046170788])
Q = np.array(
    [
        -0.36336720179946663,
        -0.8203835174702869,
        0.22865474664402222,
        0.37769321910336584,
    ]
)

# One-parameter-at-a-time sweep grid.
# Each entry: parameter name → list of absolute values to test.
# Special cases for joints_update_scaling:
#   "joints_update_scaling_N" → varies index N, keeps all other indices at their IKConfig defaults.
# Joints 0-2, 4-5 default to ~1.0 range; joint 3 (wrist) has a much smaller default.
_JOINT_SCALING_FULL = [0.1, 0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
_JOINT_SCALING_WRIST = [0.001, 0.003, 0.005, 0.008, 0.01, 0.02, 0.05, 0.1]
SWEEP_GRID: dict = {
    "pos_gain": [0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.2, 1.5],
    "orientation_gain": [0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.2, 1.5],
    "integration_dt": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
    "max_update_norm": [0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
    "regularization_strength": [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
    "joints_update_scaling_0": _JOINT_SCALING_FULL,
    "joints_update_scaling_1": _JOINT_SCALING_FULL,
    "joints_update_scaling_2": _JOINT_SCALING_FULL,
    "joints_update_scaling_3": _JOINT_SCALING_WRIST,
    "joints_update_scaling_4": _JOINT_SCALING_FULL,
    "joints_update_scaling_5": _JOINT_SCALING_FULL,
}


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
    csv_path: str = "ik_sweep_results.csv"
    """Path for the CSV output file."""
    grid_path: Optional[str] = None
    """Path to a JSON file defining the sweep grid. Overrides the hardcoded SWEEP_GRID when provided."""
    summary_path: str = "ik_sweep_summary.txt"
    """Path for the human-readable summary file (mirrors stdout)."""
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
    for i in range(n_trials):
        np.random.seed(seed + i)
        result = run_trial(model_path, ik_config, render, logger)
        result.param_name = label
        result.param_value = 0.0
        result.trial_idx = i
        results.append(result)
    return results


def run_sweep(
    param_name: str,
    values: list,
    cfg: SweepConfig,
    model_path: str,
    logger: logging.Logger,
) -> List[TrialResult]:
    results = []
    for value in values:
        ik_config = make_ik_config(param_name, value)
        for trial_idx in range(cfg.trials_per_config):
            np.random.seed(cfg.seed + trial_idx)
            result = run_trial(model_path, ik_config, cfg.render, logger)
            result.param_name = param_name
            result.param_value = value
            result.trial_idx = trial_idx
            results.append(result)
    return results


def print_sweep_summary(
    param_name: str, values: list, results: List[TrialResult], trials_per_config: int
) -> None:
    if param_name.startswith("joints_update_scaling_"):
        idx = int(param_name.split("_")[-1])
        default_value = IKConfig().joints_update_scaling[idx]
    else:
        default_value = getattr(IKConfig(), param_name)

    print(f"\n=== Sweeping {param_name} (default={default_value}) ===")
    header = f"  {'Value':>10} | {'Trials':>6} | {'Pick OK':>7} | {'Place OK':>8} | {'Full OK':>7} | {'Success%':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for value in values:
        trial_results = [r for r in results if r.param_value == value]
        n = len(trial_results)
        pick_ok = sum(r.pick_success for r in trial_results)
        place_ok = sum(r.place_success for r in trial_results)
        full_ok = sum(r.full_success for r in trial_results)
        pct = 100.0 * full_ok / n if n > 0 else 0.0
        marker = "  <- default" if abs(value - default_value) < 1e-9 else ""
        print(
            f"  {value:>10.4g} | {n:>6} | {pick_ok:>7} | {place_ok:>8} | {full_ok:>7} | {pct:>8.1f}%{marker}"
        )


def print_comparison_summary(
    baseline: List[TrialResult],
    new_defaults: List[TrialResult],
    new_config: IKConfig,
) -> None:
    def stats(results: List[TrialResult]):
        n = len(results)
        pick_ok = sum(r.pick_success for r in results)
        place_ok = sum(r.place_success for r in results)
        full_ok = sum(r.full_success for r in results)
        return n, pick_ok, place_ok, full_ok, 100.0 * full_ok / n if n else 0.0

    bn, bp, bpl, bf, bpct = stats(baseline)
    nn, np_, npl, nf, npct = stats(new_defaults)
    delta = npct - bpct

    print("\n" + "=" * 62)
    print("=== Baseline vs. New Defaults Comparison ===")
    print("=" * 62)
    print(f"  {'Config':<20} | {'Trials':>6} | {'Pick OK':>7} | {'Place OK':>8} | {'Full OK':>7} | {'Success%':>9}")
    print("  " + "-" * 60)
    print(f"  {'current defaults':<20} | {bn:>6} | {bp:>7} | {bpl:>8} | {bf:>7} | {bpct:>8.1f}%")
    print(f"  {'new best values':<20} | {nn:>6} | {np_:>7} | {npl:>8} | {nf:>7} | {npct:>8.1f}%")
    print("  " + "-" * 60)
    print(f"  Delta: {delta:+.1f}%")
    print(f"\n  New config: {dataclasses.asdict(new_config)}")
    print("=" * 62)


def write_csv(results: List[TrialResult], csv_path: str) -> None:
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


def find_model_path(script_dir: str) -> Optional[str]:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    candidates = [
        os.path.join(
            project_root, "aera", "autonomous", "simulation", "mujoco", "ar4_mk3", "scene.xml"
        ),
        "aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml",
        "../aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml",
        "../../aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml",
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def main() -> None:
    cfg = tyro.cli(SweepConfig)

    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    tee = _Tee(cfg.summary_path)
    sys.stdout = tee

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = find_model_path(script_dir)
    if model_path is None:
        logger.error("Could not find AR4 MK3 model file.")
        return

    grid = SWEEP_GRID
    if cfg.grid_path is not None:
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
    unknown = [p for p in params_to_sweep if p not in grid]
    if unknown:
        logger.error(f"Unknown parameters: {unknown}. Valid: {list(grid.keys())}")
        return

    all_results: List[TrialResult] = []

    # --- Baseline block (current defaults) ---
    baseline_results: List[TrialResult] = []
    if cfg.baseline_trials > 0:
        logger.info(f"Running {cfg.baseline_trials} baseline trials with current defaults...")
        baseline_results = run_named_block(
            "__baseline__", IKConfig(), cfg.baseline_trials,
            cfg.seed, model_path, cfg.render, logger,
        )
        all_results.extend(baseline_results)
        n = len(baseline_results)
        full_ok = sum(r.full_success for r in baseline_results)
        logger.info(f"Baseline: {full_ok}/{n} ({100.0*full_ok/n:.1f}%) success")

    # --- Parameter sweeps ---
    for param_name in params_to_sweep:
        values = grid[param_name]
        logger.info(
            f"Sweeping {param_name} over {values} "
            f"({cfg.trials_per_config} trials each, {len(values) * cfg.trials_per_config} total)"
        )
        results = run_sweep(param_name, values, cfg, model_path, logger)
        all_results.extend(results)
        print_sweep_summary(param_name, values, results, cfg.trials_per_config)

    # --- New-defaults comparison block ---
    new_defaults_results: List[TrialResult] = []
    if cfg.baseline_trials > 0:
        new_config = build_new_defaults_config(
            [r for r in all_results if r.param_name not in ("__baseline__", "__new_defaults__")],
            params_to_sweep,
        )
        logger.info(f"Running {cfg.baseline_trials} trials with new best-value defaults...")
        new_defaults_results = run_named_block(
            "__new_defaults__", new_config, cfg.baseline_trials,
            cfg.seed, model_path, cfg.render, logger,
        )
        all_results.extend(new_defaults_results)
        print_comparison_summary(baseline_results, new_defaults_results, new_config)

    write_csv(all_results, cfg.csv_path)

    sys.stdout = sys.__stdout__
    tee.close()
    logger.info(f"Summary written to {cfg.summary_path}")


if __name__ == "__main__":
    main()
