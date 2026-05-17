"""Interactive helper to dial in scene-camera randomization ranges.

Run this, drag the free camera around with the mouse, and the script prints
both the raw MuJoCo free-cam state (lookat/azim/elev/distance) and the
equivalent (pos_offset, rot_offset_euler) values consumable by
`_apply_camera_offset` / `_generate_camera_configs`.

Move to the extreme poses you want to allow, record values, then plug the
mins/maxes into `domain_rand_config_generator._generate_camera_configs`.

Controls (MuJoCo viewer):
  - left-drag:        orbit
  - right-drag:       pan
  - scroll:           zoom
  - double-click:     set lookat
  - Esc:              quit
"""

import os
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from aera.autonomous.envs.ar4_mk3_config import (
    Q as Q_LEGACY,
    Q_GEOMETRIC,
    T as T_LEGACY,
    T_GEOMETRIC,
)

# Defaults the env applies on top of the raw T,Q extrinsics (see
# Ar4Mk3Config.z_offset / .distance_multiplier in ar4_mk3_config.py).
Z_OFFSET = 0.3
DISTANCE_MULTIPLIER = 1.2
# Must match Ar4Mk3Config.use_geometric_lookat. With the legacy (False) value
# the (T, Q) -> view map is not surjective and many viewer poses are
# unreachable; offsets derived here will not match the env's renders. Run with
# True only when the env is configured with use_geometric_lookat=True.
USE_GEOMETRIC_LOOKAT = True

# Pick the T,Q baseline matching the env's auto-swap in Ar4Mk3Config.__post_init__.
T = T_GEOMETRIC if USE_GEOMETRIC_LOOKAT else T_LEGACY
Q = Q_GEOMETRIC if USE_GEOMETRIC_LOOKAT else Q_LEGACY

SCENE_XML = os.path.join(
    os.path.dirname(__file__), "..", "mujoco", "ar4_mk3", "scene.xml"
)

PRINT_EVERY_S = 0.25
POSE_EPS = 1e-3  # only reprint when something actually moved


def extrinsics_to_freecam(translation, quat_xyzw, z_offset, distance_multiplier):
    """Mirror Ar4Mk3RobotEnv._calculate_camera_config_from_transform.

    Returns the (lookat, azim, elev, distance) the env hands to the renderer.
    """
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    look_dir_in_base = R @ np.array([0.0, 0.0, -1.0])
    cam_pos_in_base = -R.T @ translation
    ray_origin = cam_pos_in_base if USE_GEOMETRIC_LOOKAT else translation
    lookat_x = ray_origin[0] + (z_offset - ray_origin[2]) * (
        look_dir_in_base[0] / look_dir_in_base[2]
    )
    lookat_y = ray_origin[1] + (z_offset - ray_origin[2]) * (
        look_dir_in_base[1] / look_dir_in_base[2]
    )
    lookat = np.array([lookat_x, lookat_y, z_offset])

    dx, dy, dz = cam_pos_in_base - lookat
    distance = float(np.linalg.norm([dx, dy, dz]))
    azimuth = float(np.degrees(np.arctan2(dy, dx)) + 90)
    elevation = float(np.degrees(np.arcsin(dz / distance)) - 90)
    return lookat, azimuth, elevation, distance_multiplier * distance


def _view_to_campos(lookat, az_deg, el_deg, dist):
    """World-space camera position implied by an env view tuple.

    Inverts the (azim, elev, dist) extraction in extrinsics_to_freecam so we
    can compare camera positions directly (smoother than comparing angles).
    `dist` includes the distance_multiplier, so we divide it out.
    """
    az_std = np.deg2rad(az_deg - 90.0)
    el_std = np.deg2rad(el_deg + 90.0)
    v = np.array(
        [
            np.cos(el_std) * np.cos(az_std),
            np.cos(el_std) * np.sin(az_std),
            np.sin(el_std),
        ]
    )
    return np.asarray(lookat) + (dist / DISTANCE_MULTIPLIER) * v


def solve_offsets(lookat_d, az_d, el_d, dist_d, x0=None):
    """Numerically find (pos_offset, rot_offset_euler) such that applying them
    to (T, Q) via _apply_camera_offset and then running
    _calculate_camera_config_from_transform reproduces the input view.
    """
    cam_pos_target = _view_to_campos(lookat_d, az_d, el_d, dist_d)

    # Parametrize rotation as a rotation vector (axis-angle) instead of Euler
    # to avoid gimbal-lock-induced local minima. We convert back at the end.
    def residuals(x):
        pos_offset = x[:3]
        rot_offset_rotvec = x[3:]
        T_new = T + pos_offset
        Q_new = (
            Rotation.from_quat(Q) * Rotation.from_rotvec(rot_offset_rotvec)
        ).as_quat()
        lookat, az, el, dist = extrinsics_to_freecam(
            T_new, Q_new, Z_OFFSET, DISTANCE_MULTIPLIER
        )
        cam_pos = _view_to_campos(lookat, az, el, dist)
        return [
            lookat[0] - lookat_d[0],
            lookat[1] - lookat_d[1],
            cam_pos[0] - cam_pos_target[0],
            cam_pos[1] - cam_pos_target[1],
            cam_pos[2] - cam_pos_target[2],
        ]

    if x0 is None:
        x0 = np.zeros(6)
    else:
        # Convert prior Euler warm-start to rotvec for parametrization.
        x0 = np.concatenate(
            [x0[:3], Rotation.from_euler("xyz", x0[3:]).as_rotvec()]
        )

    def _attempt(start):
        return least_squares(
            residuals,
            start,
            method="trf",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=400,
        )

    rng = np.random.default_rng(0)
    candidates = [x0, np.zeros(6)]
    # Add diverse restarts spanning ±pos and ±rotvec ranges.
    for scale_pos, scale_rot in [(0.2, 0.5), (0.5, 1.0), (1.0, 2.0)]:
        for _ in range(4):
            candidates.append(
                np.concatenate(
                    [scale_pos * rng.standard_normal(3), scale_rot * rng.standard_normal(3)]
                )
            )

    best = None
    for c in candidates:
        r = _attempt(c)
        if best is None or r.cost < best.cost:
            best = r
            if best.cost < 1e-12:
                break

    pos_offset = best.x[:3]
    rot_offset_euler = Rotation.from_rotvec(best.x[3:]).as_euler("xyz")
    return pos_offset, rot_offset_euler, float(best.cost)


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    last_print = 0.0
    last_state = None
    last_x = np.zeros(6)  # warm-start for the solver

    init_lookat, init_az, init_el, init_dist = extrinsics_to_freecam(
        T, Q, Z_OFFSET, DISTANCE_MULTIPLIER
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Match what the env's renderer actually uses (T,Q + z_offset + distance_multiplier).
        viewer.cam.lookat[:] = init_lookat
        viewer.cam.azimuth = init_az
        viewer.cam.elevation = init_el
        viewer.cam.distance = init_dist

        print("Base T:", T.tolist())
        print("Base Q (xyzw):", Q.tolist())
        print(
            f"Initial free-cam: lookat={init_lookat.round(3).tolist()} "
            f"azim={init_az:.2f} elev={init_el:.2f} dist={init_dist:.3f}"
        )
        print("Drag the camera; values print when they change.\n")

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            now = time.time()
            if now - last_print >= PRINT_EVERY_S:
                lookat = np.array(viewer.cam.lookat)
                az = float(viewer.cam.azimuth)
                el = float(viewer.cam.elevation)
                dist = float(viewer.cam.distance)
                state = np.array([*lookat, az, el, dist])

                if last_state is None or np.any(np.abs(state - last_state) > POSE_EPS):
                    pos_offset, rot_offset_euler, cost = solve_offsets(
                        lookat, az, el, dist, x0=last_x
                    )
                    last_x = np.concatenate([pos_offset, rot_offset_euler])

                    # Round-trip: what view will the env render with these offsets?
                    T_rt = T + pos_offset
                    Q_rt = (
                        Rotation.from_quat(Q)
                        * Rotation.from_euler("xyz", rot_offset_euler)
                    ).as_quat()
                    lookat_rt, az_rt, el_rt, dist_rt = extrinsics_to_freecam(
                        T_rt, Q_rt, Z_OFFSET, DISTANCE_MULTIPLIER
                    )

                    warn = ""
                    if abs(lookat[2] - Z_OFFSET) > 1e-3:
                        warn += "  [warn: lookat.z != z_offset]"
                    if cost > 1e-6:
                        warn += f"  [warn: residual {cost:.2e}]"
                    print(
                        f"lookat={lookat.round(3).tolist()}  "
                        f"azim={az:.2f}  elev={el:.2f}  dist={dist:.3f}{warn}"
                    )
                    print(
                        f"  pos_offset={pos_offset.round(4).tolist()}  "
                        f"rot_offset_euler={rot_offset_euler.round(4).tolist()}"
                    )
                    print(
                        f"  env round-trip: lookat={lookat_rt.round(3).tolist()} "
                        f"azim={az_rt:.2f} elev={el_rt:.2f} dist={dist_rt:.3f}"
                    )
                    print(f"""
pos_offset = [
        float(np.random.uniform({pos_offset.round(4)[0]}, {pos_offset.round(4)[0]})),
        float(np.random.uniform({pos_offset.round(4)[1]}, {pos_offset.round(4)[1]})),
        float(np.random.uniform({pos_offset.round(4)[2]}, {pos_offset.round(4)[2]})),
]
rot_offset_euler = [
        float(np.random.uniform({rot_offset_euler.round(4)[0]}, {rot_offset_euler.round(4)[0]})),
        float(np.random.uniform({rot_offset_euler.round(4)[1]}, {rot_offset_euler.round(4)[1]})),
        float(np.random.uniform({rot_offset_euler.round(4)[2]}, {rot_offset_euler.round(4)[2]})),
]
                        """)
                    last_state = state
                last_print = now

            sleep_t = model.opt.timestep - (time.time() - step_start)
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    main()
