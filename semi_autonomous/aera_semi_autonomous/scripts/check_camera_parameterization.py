#!/usr/bin/env python3
"""Do collection and eval render from the same camera distribution?

Collection builds the scene camera as (T_GEOMETRIC, Q_GEOMETRIC,
use_geometric_lookat=True); eval's `_build_env` leaves the env defaults
(T, Q, False). With no camera DR the two are identical by construction —
T_GEOMETRIC was calibrated to reproduce the legacy view exactly — which is why
the mismatch went unnoticed. But `_apply_camera_offset` ADDS the sampled DR
offset to the base translation, and those offsets were validated in collection's
parameterization, so on the eval base they land somewhere else entirely
(measured: azimuth 209 deg vs 264 deg, elevation -31 vs -8, distance 1.11 vs
0.86 — the camera ends up on the other side of the scene).

=> turning on `randomize_cameras` at eval WITHOUT `use_geometric_lookat` samples
an out-of-distribution camera. Both flags or neither.

Since the shared env-config factory landed, both paths build their config from
`build_task_env_config`, so this now reads the LIVE configs and asserts they
agree — it is a regression check, not just a demonstration. `--show-legacy`
re-prints the historical (T, Q, False) eval parameterization alongside, which is
what the divergence above was measured on.

Exits 1 if collection and eval disagree.

Usage:
    python -m ...scripts.check_camera_parameterization --samples 3
"""

import argparse

import numpy as np

from aera.autonomous.envs.ar4_mk3_base import Ar4Mk3Env
from aera.autonomous.envs.ar4_mk3_config import Q, T
from aera.autonomous.envs.task_env_factory import build_task_env_config
from aera_semi_autonomous.data.domain_rand_config_generator import (
    _sample_scene_camera_pose,
)

Z_OFFSET = 0.3
DISTANCE_MULTIPLIER = 1.2
_MODEL_PATH = "/tmp/scene.xml"  # never loaded; only the camera fields are read


class _CameraOffset:
    """Minimal stand-in for CameraConfig's two offset fields."""

    def __init__(self, pos_offset, rot_offset_euler):
        self.pos_offset = pos_offset
        self.rot_offset_euler = rot_offset_euler


def _camera_view(translation, quaternion, geometric, offset) -> dict:
    t, q = Ar4Mk3Env._apply_camera_offset(translation, quaternion, offset)
    return Ar4Mk3Env._calculate_camera_config_from_transform(
        None, t, q, Z_OFFSET, DISTANCE_MULTIPLIER, geometric
    )


def _fmt(view: dict) -> str:
    return (
        f"lookat={np.round(view['lookat'], 3)} azim={view['azimuth']:7.1f} "
        f"elev={view['elevation']:6.1f} dist={view['distance']:.3f}"
    )


def _view_from_config(config, offset) -> dict:
    return _camera_view(
        config.translation, config.quaterion, config.use_geometric_lookat, offset
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=3, help="DR camera offsets to draw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--show-legacy",
        action="store_true",
        help="also print the pre-factory eval parameterization (T, Q, False)",
    )
    args = parser.parse_args()

    # The two configs as the code actually builds them today.
    collection_cfg = build_task_env_config(_MODEL_PATH, domain_rand=None)
    eval_cfg = build_task_env_config(
        _MODEL_PATH,
        domain_rand=None,
        eval_overrides={"n_substeps": 10, "kinematic_grasp": True},
    )

    failures = 0
    offsets = [None]
    np.random.seed(args.seed)
    for _ in range(args.samples):
        pos_offset, rot_offset = _sample_scene_camera_pose()
        offsets.append(_CameraOffset(pos_offset, rot_offset))

    for i, offset in enumerate(offsets):
        label = (
            "no camera DR"
            if offset is None
            else f"DR sample {i - 1} pos_offset={np.round(offset.pos_offset, 3)}"
        )
        collection_view = _view_from_config(collection_cfg, offset)
        eval_view = _view_from_config(eval_cfg, offset)
        agree = _fmt(collection_view) == _fmt(eval_view)
        failures += not agree
        print(f"\n{label}  [{'OK' if agree else 'MISMATCH'}]")
        print("  collection:", _fmt(collection_view))
        print("  eval      :", _fmt(eval_view))
        if args.show_legacy:
            print("  eval (pre-factory):", _fmt(_camera_view(T, Q, False, offset)))

    print(
        f"\n{'PASS' if not failures else 'FAIL'}: "
        f"{len(offsets) - failures}/{len(offsets)} camera poses agree."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
