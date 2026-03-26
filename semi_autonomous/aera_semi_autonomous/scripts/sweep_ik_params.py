#!/usr/bin/env python3
"""
IK parameter sweep script for finding sensible noise ranges.

Runs one-parameter-at-a-time sweeps over IK solver config values, holding all
other parameters at their defaults. For each value, N pick-and-place trials are
executed with domain randomization. Results are printed as a summary table and
written to CSV.

Usage:
    # Sweep all parameters, 5 trials each
    python sweep_ik_params.py

    # Sweep a single parameter with 2 trials (quick sanity check)
    python sweep_ik_params.py --params pos_gain --trials-per-config 2

    # Custom CSV output path
    python sweep_ik_params.py --csv-path /tmp/sweep.csv
"""

import csv
import dataclasses
import logging
import os
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
# The parameter name matches the IKConfig field name, with one special case:
#   "joints_update_scaling_3" → sets joints_update_scaling[3] (the wrist joint).
SWEEP_GRID: dict = {
    "pos_gain": [0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.2, 1.5],
    "orientation_gain": [0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.2, 1.5],
    "integration_dt": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
    "max_update_norm": [0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
    "regularization_strength": [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
    "joints_update_scaling_3": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
}


@dataclass
class SweepConfig:
    trials_per_config: int = 5
    """Number of pick-and-place trials per (parameter, value) combination."""
    render: bool = False
    """Enable MuJoCo rendering (slow)."""
    seed: int = 42
    """Base random seed. Trial i uses seed + i for reproducibility."""
    params: List[str] = field(default_factory=list)
    """Parameters to sweep. Empty list = sweep all. E.g. --params pos_gain integration_dt"""
    csv_path: str = "ik_sweep_results.csv"
    """Path for the CSV output file."""
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
    if param_name == "joints_update_scaling_3":
        scaling = list(base.joints_update_scaling)
        scaling[3] = value
        return dataclasses.replace(base, joints_update_scaling=scaling)
    return dataclasses.replace(base, **{param_name: value})


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
    default_value = getattr(IKConfig(), param_name if param_name != "joints_update_scaling_3" else "joints_update_scaling")
    if param_name == "joints_update_scaling_3":
        default_value = IKConfig().joints_update_scaling[3]

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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = find_model_path(script_dir)
    if model_path is None:
        logger.error("Could not find AR4 MK3 model file.")
        return

    params_to_sweep = cfg.params if cfg.params else list(SWEEP_GRID.keys())
    unknown = [p for p in params_to_sweep if p not in SWEEP_GRID]
    if unknown:
        logger.error(f"Unknown parameters: {unknown}. Valid: {list(SWEEP_GRID.keys())}")
        return

    all_results: List[TrialResult] = []
    for param_name in params_to_sweep:
        values = SWEEP_GRID[param_name]
        logger.info(
            f"Sweeping {param_name} over {values} "
            f"({cfg.trials_per_config} trials each, {len(values) * cfg.trials_per_config} total)"
        )
        results = run_sweep(param_name, values, cfg, model_path, logger)
        all_results.extend(results)
        print_sweep_summary(param_name, values, results, cfg.trials_per_config)

    write_csv(all_results, cfg.csv_path)


if __name__ == "__main__":
    main()
