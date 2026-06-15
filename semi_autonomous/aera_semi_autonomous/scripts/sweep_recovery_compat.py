#!/usr/bin/env python3
"""Recovery-compatibility sweep.

`--perturbation.perturb-recovery` is the one DR lever that leans on raw contact
physics and re-runs the full grasp pipeline three times per episode (wrong
approach, partial-grasp slip + settle, then the real re-grasp through the
demanding alignment gate). That makes it the only perturbation whose success
rate is dragged down by *other* perturbations — anything that erodes IK
end-pose precision or shifts contact dynamics gets paid ~3x.

Visual / camera DR does NOT interact: the scripted expert reads object poses
straight from sim state, never from the rendered images, so colours, textures,
lights, props, wall-art and camera extrinsics can't change recovery success.
This sweep therefore leaves visuals out and toggles only the levers that touch
the control loop, the contacts, or the grasp geometry:

    arm_dynamics   movement DR (per-joint kp/kv/damping/armature/friction/force)
    object_yaw     blocks spawn rotated (extra rotational IK demand)
    hover          per-episode pre-grasp/place hover height
    home           perturbed home start configuration
    speed          per-episode motion-tempo factor
    actuation      latency / command-lag / step-jitter
    ik_noise       multiplicative noise on the IK solver config
    offset_approach extra approach waypoints on the pick/place

For each scenario we run N paired trials (the same trial index uses the same
seed across scenarios, so the scene + object spawn match and the success-rate
delta isolates the toggled lever) and record WHICH STAGE failed:

    home    couldn't reach (perturbed) home
    locate  couldn't read the object pose
    grasp   grasp_at() returned False (IK couldn't reach the grasp pose)
    held    grasp_at() returned True but the alignment gate rejected the lock
            (the re-grasp after the slip landed outside the 7mm pinch envelope)
    place   release_at() returned False
    settle  everything ran but the object ended too far from the target
    none    success

Two recovery=False reference scenarios (clean baseline, everything-no-recovery)
bracket the result so you can see how much recovery itself costs vs the axes.

Usage:
    # Default matrix, 30 paired trials each
    python sweep_recovery_compat.py --trials-per-scenario 30

    # Just a few scenarios, watch it run
    python sweep_recovery_compat.py --scenarios recovery_only rec+arm_dynamics --render

    # Crank the ik_noise level used by the rec+ik_noise / rec+everything combos
    python sweep_recovery_compat.py --ik-noise-fraction 0.05
"""

import csv
import dataclasses
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np
import tyro
from geometry_msgs.msg import Point, Pose, Quaternion
from tqdm import tqdm

from aera.autonomous.envs.ar4_mk3_config import Ar4Mk3EnvConfig
from aera.autonomous.envs.ar4_mk3_pick_and_place import Ar4Mk3PickAndPlaceEnv
from aera_semi_autonomous.control.ar4_mk3_interface_config import (
    Ar4Mk3InterfaceConfig,
)
from aera_semi_autonomous.control.ar4_mk3_robot_interface import Ar4Mk3RobotInterface
from aera_semi_autonomous.data.domain_rand_config_generator import (
    generate_random_domain_rand_config,
)
from aera_semi_autonomous.data.pick_and_place_helpers import (
    get_object_grasp_gripper_pos,
    get_object_pose,
    inject_partial_grasp,
    inject_wrong_approach,
)
from aera_semi_autonomous.data.trajectory_perturbation import (
    ActuationPerturbation,
    HoverHeightPerturbation,
    HomeOffsetPerturbation,
    IKNoisePerturbation,
    PerturbationConfig,
    RecoveryPerturbation,
    SpeedPerturbation,
    apply_hover_height_perturbation,
    apply_speed_perturbation,
    generate_waypoints,
    go_home_perturbed,
    perturb_ik_config,
    sample_actuation_config,
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

_DEFAULT_DATA_DIR = "data"


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """One row of the sweep: recovery (usually) on + zero or more levers."""

    name: str
    recovery: bool = True
    randomize_arm_dynamics: bool = False
    randomize_object_yaw: bool = False
    perturb_home: bool = False
    perturb_hover_height: bool = False
    perturb_speed: bool = False
    perturb_actuation: bool = False
    # Sampling ranges used when perturb_actuation is on. None = the default
    # ActuationPerturbation() (all three knobs active). Set an override to
    # isolate a single knob (the others pinned to their identity).
    actuation: Optional[ActuationPerturbation] = None
    # "none" | "ik_noise" | "offset_approach"
    mode: str = "none"


def default_scenarios() -> List[Scenario]:
    """Reference ceilings + recovery-only baseline + one-lever-at-a-time +
    the recommended bundle + everything-on."""
    return [
        # --- reference ceilings (recovery OFF) ---
        Scenario("clean_baseline", recovery=False),
        Scenario(
            "everything_no_recovery",
            recovery=False,
            randomize_arm_dynamics=True,
            randomize_object_yaw=True,
            perturb_home=True,
            perturb_hover_height=True,
            perturb_speed=True,
            perturb_actuation=True,
            mode="ik_noise",
        ),
        # --- recovery baseline ---
        Scenario("recovery_only"),
        # --- recovery + one lever ---
        Scenario("rec+arm_dynamics", randomize_arm_dynamics=True),
        Scenario("rec+object_yaw", randomize_object_yaw=True),
        Scenario("rec+hover", perturb_hover_height=True),
        Scenario("rec+home", perturb_home=True),
        Scenario("rec+speed", perturb_speed=True),
        Scenario("rec+actuation", perturb_actuation=True),
        # --- actuation sub-knob isolation (each pins the other two to identity) ---
        Scenario(
            "rec+act_latency",
            perturb_actuation=True,
            actuation=ActuationPerturbation(
                latency_steps_range=(0, 4),
                command_lag_alpha_range=(1.0, 1.0),
                step_jitter_prob_range=(0.0, 0.0),
            ),
        ),
        Scenario(
            "rec+act_lag",
            perturb_actuation=True,
            actuation=ActuationPerturbation(
                latency_steps_range=(0, 0),
                command_lag_alpha_range=(0.2, 0.8),
                step_jitter_prob_range=(0.0, 0.0),
            ),
        ),
        Scenario(
            "rec+act_jitter",
            perturb_actuation=True,
            actuation=ActuationPerturbation(
                latency_steps_range=(0, 0),
                command_lag_alpha_range=(1.0, 1.0),
                step_jitter_prob_range=(0.0, 0.1),
            ),
        ),
        Scenario("rec+ik_noise", mode="ik_noise"),
        Scenario("rec+offset_approach", mode="offset_approach"),
        # --- recommended bundle (Tier 1 + Tier 2) ---
        Scenario(
            "rec+recommended",
            randomize_arm_dynamics=True,
            randomize_object_yaw=True,
            perturb_hover_height=True,
            perturb_home=True,
        ),
        # --- everything non-visual ---
        Scenario(
            "rec+everything",
            randomize_arm_dynamics=True,
            randomize_object_yaw=True,
            perturb_home=True,
            perturb_hover_height=True,
            perturb_speed=True,
            perturb_actuation=True,
            mode="ik_noise",
        ),
    ]


@dataclass
class TrialResult:
    scenario: str
    trial_idx: int
    located: bool
    grasp_returned: bool
    object_held: bool
    place_returned: bool
    full_success: bool
    fail_stage: str  # home|locate|grasp|held|place|settle|exception|none


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------


def build_interface_config(
    scn: Scenario, render: bool, ik_noise_fraction: float
) -> Ar4Mk3InterfaceConfig:
    """Mirror collect_trajectories' interface assembly for the given scenario."""
    cfg = Ar4Mk3InterfaceConfig(render_steps=render)
    if scn.mode == "ik_noise":
        cfg = replace(
            cfg,
            ik=perturb_ik_config(
                cfg.ik, IKNoisePerturbation(default_fraction=ik_noise_fraction)
            ),
        )
    if scn.perturb_actuation:
        cfg = replace(
            cfg,
            actuation=sample_actuation_config(
                scn.actuation or ActuationPerturbation()
            ),
        )
    if scn.perturb_speed:
        cfg = apply_speed_perturbation(cfg, SpeedPerturbation())
    if scn.perturb_hover_height:
        cfg = apply_hover_height_perturbation(cfg, HoverHeightPerturbation())
    return cfg


def build_perturbation_config(
    scn: Scenario, recovery: RecoveryPerturbation
) -> PerturbationConfig:
    """PerturbationConfig matching the scenario. perturb_pick/place only do
    something when mode == 'offset_approach' (generate_waypoints is a no-op
    otherwise), so they're left at their default True."""
    return PerturbationConfig(
        mode=scn.mode,
        perturb_home=scn.perturb_home,
        home_offset=HomeOffsetPerturbation(),
        perturb_recovery=scn.recovery,
        recovery=recovery,
    )


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------


def run_trial(
    scn: Scenario,
    trial_idx: int,
    seed: int,
    model_path: str,
    recovery: RecoveryPerturbation,
    ik_noise_fraction: float,
    render: bool,
    logger: logging.Logger,
) -> TrialResult:
    """One paired pick-and-place episode. The same (trial_idx, seed) reproduces
    the same scene/object spawn across scenarios, so the only difference is the
    toggled lever."""
    np.random.seed(seed)
    located = grasp_returned = object_held = place_returned = full_success = False
    fail_stage = "exception"
    env = None
    try:
        dr_config, _, _ = generate_random_domain_rand_config(
            randomize_cameras=False,
            randomize_arm_dynamics=scn.randomize_arm_dynamics,
        )
        env_config = Ar4Mk3EnvConfig(
            model_path=model_path,
            reward_type="sparse",
            use_eef_control=False,
            translation=T,
            quaterion=Q,
            distance_multiplier=1.2,
            z_offset=0.3,
            use_geometric_lookat=True,
            domain_rand=dr_config,
            randomize_object_yaw=scn.randomize_object_yaw,
        )
        env = Ar4Mk3PickAndPlaceEnv(
            render_mode="human" if render else None, config=env_config
        )
        env.reset(seed=seed)

        interface_config = build_interface_config(scn, render, ik_noise_fraction)
        pcfg = build_perturbation_config(scn, recovery)
        robot = Ar4Mk3RobotInterface(env, config=interface_config)

        if not go_home_perturbed(robot, pcfg):
            return TrialResult(
                scn.name, trial_idx, False, False, False, False, False, "home"
            )

        object_pose = get_object_pose(env, logger)
        if object_pose is None:
            return TrialResult(
                scn.name, trial_idx, False, False, False, False, False, "locate"
            )
        located = True

        grasp_gripper_pos = get_object_grasp_gripper_pos(env, logger=logger)

        if pcfg.perturb_recovery:
            rec = pcfg.recovery
            if rec.wrong_approach:
                inject_wrong_approach(robot, object_pose, rec, logger)
            if rec.partial_grasp:
                object_pose = inject_partial_grasp(
                    robot, env, object_pose, grasp_gripper_pos, rec, logger
                )
                grasp_gripper_pos = get_object_grasp_gripper_pos(env, logger=logger)

        if pcfg.perturb_pick:
            for wp in generate_waypoints(object_pose, pcfg):
                robot.move_to(wp)

        grasp_returned = robot.grasp_at(object_pose, gripper_pos=grasp_gripper_pos)
        object_held = robot.is_object_held()
        if not grasp_returned:
            return TrialResult(
                scn.name, trial_idx, located, False, object_held, False, False, "grasp"
            )
        if not object_held:
            # grasp_at returns True even when the alignment gate refuses to lock,
            # so this is the signature of a re-grasp that missed the pinch envelope.
            return TrialResult(
                scn.name, trial_idx, located, True, False, False, False, "held"
            )

        target_pos = env.goal
        target_pose = Pose()
        target_pose.position = Point(
            x=float(target_pos[0]),
            y=float(target_pos[1]),
            z=float(target_pos[2] + object_pose.position.z),
        )
        target_pose.orientation = Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)
        if pcfg.perturb_place:
            for wp in generate_waypoints(target_pose, pcfg):
                robot.move_to(wp)
        place_returned = robot.release_at(target_pose)
        robot.go_home()

        final_pos = env._utils.get_site_xpos(env.model, env.data, "object0")
        distance = float(np.linalg.norm(final_pos - target_pos))
        full_success = distance < env.distance_threshold

        if not place_returned:
            fail_stage = "place"
        elif not full_success:
            fail_stage = "settle"
        else:
            fail_stage = "none"
    except Exception as e:
        logger.warning(f"[{scn.name} #{trial_idx}] trial raised: {e}")
        fail_stage = "exception"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return TrialResult(
        scn.name,
        trial_idx,
        located,
        grasp_returned,
        object_held,
        place_returned,
        full_success,
        fail_stage,
    )


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------

_STAGES = ["home", "locate", "grasp", "held", "place", "settle", "exception"]


def summarize(scn_name: str, results: List[TrialResult]) -> dict:
    n = len(results)
    held = sum(r.object_held for r in results)
    full = sum(r.full_success for r in results)
    stages = Counter(r.fail_stage for r in results)
    return {
        "scenario": scn_name,
        "trials": n,
        "pick_ok": held,
        "pick_pct": round(100.0 * held / n, 1) if n else 0.0,
        "full_ok": full,
        "success_pct": round(100.0 * full / n, 1) if n else 0.0,
        "fail_stages": {s: stages.get(s, 0) for s in _STAGES if stages.get(s, 0)},
    }


def print_summary(rows: List[dict], baseline_name: str) -> None:
    base = next((r for r in rows if r["scenario"] == baseline_name), None)
    base_pct = base["success_pct"] if base else None

    print("\n" + "=" * 92)
    print("=== Recovery-compatibility sweep ===")
    print("=" * 92)
    header = (
        f"  {'Scenario':<24} | {'Trials':>6} | {'Pick%':>6} | {'Full%':>6} "
        f"| {'vs base':>7} | Dominant failure stages"
    )
    print(header)
    print("  " + "-" * (len(header) + 6))
    for r in rows:
        delta = ""
        if base_pct is not None and r["scenario"] != baseline_name:
            delta = f"{r['success_pct'] - base_pct:+.1f}"
        stages = ", ".join(
            f"{s}:{c}" for s, c in sorted(
                r["fail_stages"].items(), key=lambda kv: -kv[1]
            )
        ) or "—"
        print(
            f"  {r['scenario']:<24} | {r['trials']:>6} | {r['pick_pct']:>5.1f}% "
            f"| {r['success_pct']:>5.1f}% | {delta:>7} | {stages}"
        )
    print("=" * 92)


def print_focus(rows: List[dict], baseline_name: str) -> None:
    """Rank the recovery+lever scenarios by how much they cost vs recovery_only,
    and name the failure stage that dominates each costly combo."""
    base = next((r for r in rows if r["scenario"] == baseline_name), None)
    if base is None:
        return
    base_pct = base["success_pct"]
    combos = [
        r
        for r in rows
        if r["scenario"].startswith("rec+") and r["scenario"] != baseline_name
    ]
    combos.sort(key=lambda r: r["success_pct"])
    print("\n=== Where to focus (sorted by cost vs recovery_only) ===")
    print(f"  recovery_only baseline: {base_pct:.1f}%")
    for r in combos:
        drop = base_pct - r["success_pct"]
        dom = max(r["fail_stages"].items(), key=lambda kv: kv[1])[0] if r[
            "fail_stages"
        ] else "none"
        verdict = "OK (>=50%)" if r["success_pct"] >= 50 else "BELOW 50%"
        print(
            f"  {r['scenario']:<24} {r['success_pct']:>5.1f}%  "
            f"(−{drop:>4.1f})  dominant={dom:<7}  {verdict}"
        )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def write_csv(results: List[TrialResult], path: str) -> None:
    _ensure_dir(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dataclasses.asdict(results[0])))
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    print(f"\nPer-trial results written to {path}")


def write_json(rows: List[dict], path: str) -> None:
    _ensure_dir(path)
    with open(path, "w") as f:
        json.dump({"scenarios": rows}, f, indent=2)
    print(f"Summary written to {path}")


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
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    trials_per_scenario: int = 30
    """Paired trials per scenario (same seed sequence across scenarios)."""
    seed: int = 1000
    """Base seed. Trial i of every scenario uses seed + i."""
    scenarios: List[str] = field(default_factory=list)
    """Subset of scenario names to run (empty = the full default matrix)."""
    ik_noise_fraction: float = 0.1
    """Multiplicative IK-noise level for the rec+ik_noise / rec+everything combos."""
    render: bool = False
    """Render the sim (slow; for eyeballing what a combo actually does)."""
    recovery: RecoveryPerturbation = field(default_factory=RecoveryPerturbation)
    """Recovery sub-mode config (e.g. --recovery.no-wrong-approach to isolate slip)."""
    csv_path: str = f"{_DEFAULT_DATA_DIR}/recovery_compat_trials.csv"
    summary_path: str = f"{_DEFAULT_DATA_DIR}/recovery_compat_summary.json"
    debug: bool = False


def main() -> None:
    cfg = tyro.cli(SweepConfig)
    logging.basicConfig(
        level=logging.DEBUG if cfg.debug else logging.CRITICAL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = find_model_path(script_dir)
    if model_path is None:
        print("Could not find AR4 MK3 scene.xml", file=sys.stderr)
        return

    scenarios = default_scenarios()
    if cfg.scenarios:
        wanted = set(cfg.scenarios)
        unknown = wanted - {s.name for s in scenarios}
        if unknown:
            print(
                f"Unknown scenarios: {sorted(unknown)}. "
                f"Valid: {[s.name for s in scenarios]}",
                file=sys.stderr,
            )
            return
        scenarios = [s for s in scenarios if s.name in wanted]

    all_results: List[TrialResult] = []
    rows: List[dict] = []
    for scn in scenarios:
        scn_results: List[TrialResult] = []
        with tqdm(
            total=cfg.trials_per_scenario, desc=scn.name, unit="trial", leave=True
        ) as pbar:
            for i in range(cfg.trials_per_scenario):
                result = run_trial(
                    scn,
                    i,
                    cfg.seed + i,
                    model_path,
                    cfg.recovery,
                    cfg.ik_noise_fraction,
                    cfg.render,
                    logger,
                )
                scn_results.append(result)
                all_results.append(result)
                n_ok = sum(r.full_success for r in scn_results)
                pbar.set_postfix(full=f"{n_ok}/{i + 1}")
                pbar.update(1)
        rows.append(summarize(scn.name, scn_results))

    print_summary(rows, baseline_name="recovery_only")
    print_focus(rows, baseline_name="recovery_only")

    if all_results:
        write_csv(all_results, cfg.csv_path)
    write_json(rows, cfg.summary_path)


if __name__ == "__main__":
    main()
