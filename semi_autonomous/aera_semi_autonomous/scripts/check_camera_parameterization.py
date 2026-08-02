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

This script prints both parameterizations side by side, with and without DR
offsets. Use it to confirm the shared env-config factory keeps eval == collection.

Usage:
    python -m ...scripts.check_camera_parameterization --samples 3
"""

import argparse

import numpy as np

from aera.autonomous.envs.ar4_mk3_base import Ar4Mk3Env
from aera.autonomous.envs.ar4_mk3_config import Q, Q_GEOMETRIC, T, T_GEOMETRIC
from aera_semi_autonomous.data.domain_rand_config_generator import (
    _sample_scene_camera_pose,
)

Z_OFFSET = 0.3
DISTANCE_MULTIPLIER = 1.2


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=3, help="DR camera offsets to draw")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("no camera DR (identical by construction — T_GEOMETRIC is calibrated for this):")
    print("  collection:", _fmt(_camera_view(T_GEOMETRIC, Q_GEOMETRIC, True, None)))
    print("  eval      :", _fmt(_camera_view(T, Q, False, None)))

    np.random.seed(args.seed)
    for i in range(args.samples):
        pos_offset, rot_offset = _sample_scene_camera_pose()
        offset = _CameraOffset(pos_offset, rot_offset)
        print(f"\nDR sample {i} pos_offset={np.round(pos_offset, 3)}")
        print("  collection:", _fmt(_camera_view(T_GEOMETRIC, Q_GEOMETRIC, True, offset)))
        print("  eval      :", _fmt(_camera_view(T, Q, False, offset)))


if __name__ == "__main__":
    main()
