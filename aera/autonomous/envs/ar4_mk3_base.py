import json
from pathlib import Path
from typing import Optional, Sequence

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import numpy as np
import os
import mujoco

from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations
from scipy.spatial.transform import Rotation

from aera.autonomous.envs.ar4_mk3_config import Ar4Mk3EnvConfig, PLA_BLOCK_PRESETS
from aera.autonomous.envs.kinematic_grasp import KinematicGraspLock
from aera.autonomous.obs_augmentation import augment_image, sample_camera_profile


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def _quat_z_to(direction) -> np.ndarray:
    """MuJoCo (w, x, y, z) quaternion rotating the capsule's local +Z axis onto
    `direction`. Used to orient an arched cable's sub-capsules along each chord
    segment."""
    d = np.array(direction, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    d /= n
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-9:
        return np.array([0.0, 1.0, 0.0, 0.0])  # 180° about X
    axis = np.cross(z, d)
    axis /= np.linalg.norm(axis)
    ang = np.arccos(c)
    s = np.sin(ang / 2.0)
    return np.array([np.cos(ang / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


class BaseRender(MujocoRenderer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def render(
        self,
        render_mode: str | None,
        camera_id: int | None = None,
        camera_name: str | None = None,
    ):
        if render_mode == "human":
            super().render(render_mode)
            return
        if camera_id is None and camera_name is None:
            camera_id = self.camera_id
        elif camera_id is None:
            camera_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
            )
        return self._get_viewer(render_mode).render(
            render_mode=render_mode, camera_id=camera_id
        )


class BaseEnv(MujocoRobotEnv):
    """Superclass for all Fetch environments."""

    def __init__(
        self,
        config: Ar4Mk3EnvConfig,
        default_camera_config: dict | None = None,
        **kwargs,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range float: range of a uniform distribution for sampling initial object positions
            target_range (float): range o)f a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            object_size (float or array-like with 3 elements): size of the object (box half-lengths).
        """
        self.config = config
        # Per-block resting half-height, set by the block-preset DR pass so the
        # spawn pose in _reset_sim places each (possibly differently sized) block
        # on the table. Defaults to object_size when DR is off.
        self._block_half_size: dict[str, float] = {}
        # Compiled positions of the arm-cable tube geoms, cached on first DR
        # pass. The cable pos jitter is additive and self.model persists across
        # episodes, so we add the offset to this cached base rather than to the
        # current (already-jittered) geom_pos — otherwise the routing drifts.
        self._cable_base_pos: dict[int, np.ndarray] = {}
        self.gripper_extra_height = config.gripper_extra_height
        self.block_gripper = config.block_gripper
        self.has_object = config.has_object
        self.target_in_the_air = config.target_in_the_air
        self.target_offset = config.target_offset
        self.obj_range = config.obj_range
        self.obj_offset = config.obj_offset
        self.target_range = config.target_range
        self.distance_threshold = config.distance_threshold
        self.reward_type = config.reward_type
        self.absolute_state_actions = config.absolute_state_actions
        self.mujoco_renderer = None
        if isinstance(config.object_size, (float, int)):
            self.object_size = np.array([config.object_size] * 3)
        else:
            self.object_size = np.array(config.object_size)
        assert self.object_size.shape == (3,), (
            "object_size must be a float or a 3-element array-like"
        )

        self.use_eef_control = config.use_eef_control
        current_dir = os.path.dirname(__file__)
        env_base_path = os.path.join(
            current_dir, "..", "simulation", "mujoco", "ar4_mk3"
        )
        if config.model_path is not None:
            self.model_path = config.model_path
        elif self.use_eef_control:
            self.model_path = os.path.join(env_base_path, "scene_eef.xml")
        else:
            self.model_path = os.path.join(env_base_path, "scene.xml")

        super().__init__(
            model_path=self.model_path,
            n_substeps=config.n_substeps,
            n_actions=4 if config.use_eef_control else 7,
            width=config.image_width,
            height=config.image_height,
            **kwargs,
        )

        self.mujoco_renderer = BaseRender(
            model=self.model,
            data=self.data,
            default_cam_config=default_camera_config,
            width=config.image_width,
            height=config.image_height,
        )

    def _initialize_simulation(self):
        super()._initialize_simulation()
        # The base class asserts render_fps == 1/dt with a class-level metadata
        # of 25 fps, which only holds for n_substeps=20. Recompute it from the
        # actual dt (timestep * n_substeps) so any n_substeps passes; instance
        # copy so the shared class dict isn't mutated.
        self.metadata = {
            **self.metadata,
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _get_obs(self):
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
            default_img,
            gripper_img,
        ) = self.generate_mujoco_observations()

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        obs_vec = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ]
        )

        obs = {
            "observation": obs_vec.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

        if self.config.include_images_in_obs:
            obs["default_camera_image"] = default_img
            obs["gripper_camera_image"] = gripper_img

        return obs

    def generate_mujoco_observations(self):
        raise NotImplementedError

    def _get_gripper_xpos(self):
        raise NotImplementedError

    def _sample_goal(self):
        raise NotImplementedError

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)


class Ar4Mk3Env(BaseEnv):
    # Objects the eval grasp lock may attach (mirrors the interface's set).
    _GRASP_OBJECT_NAMES = ("object0", "object_distractor1", "object_distractor2")
    # Gripper-target thresholds (ctrl units, range [-0.014 open, 0 closed]) used
    # to infer engage/release from the policy's gripper command. Engage is
    # two-staged: this floor ("clearly inside full-open", kept above the release
    # threshold for hysteresis) plus the lock's per-object close-depth gate,
    # which requires the command to reach the candidate block's own surface
    # (collection closes to -(half_width - 0.5mm), so grasp depth scales with
    # block size: -0.009 for the 19 mm preset down to -0.0125 for the 24 mm
    # graspable max). A policy sweeping past a block with a barely-closing
    # command welds nothing. The grasp target is always <= 24 mm
    # (_GRASPABLE_BLOCK_PRESETS), so the floor only binds for wrong-object
    # grabs of the larger distractor-only presets (27/30 mm), whose deeper
    # close commands sit at or below it — those cannot weld, which is fine.
    _GRASP_ENGAGE_CTRL = -0.013
    _GRASP_RELEASE_CTRL = -0.0135

    def __init__(
        self,
        config: Ar4Mk3EnvConfig,
        **kwargs,
    ):
        # Set before super().__init__ so an early step/reset never sees it unset.
        self._grasp_lock = None
        self._gripper_act_ids = None
        # Eval-time image sensor-realism: per-episode profile (resampled on
        # reset) + a persistent rng for per-frame noise. None = disabled.
        self._eval_cam_profile = None
        self._obs_aug_rng = np.random.default_rng()
        default_camera_config = config.default_camera_config
        if config.translation is not None and config.quaterion is not None:
            cam_cfg = (
                config.domain_rand.default_camera if config.domain_rand else None
            )
            translation, quaterion = self._apply_camera_offset(
                config.translation, config.quaterion, cam_cfg
            )
            default_camera_config = self._calculate_camera_config_from_transform(
                translation,
                quaterion,
                config.z_offset,
                config.distance_multiplier,
                config.use_geometric_lookat,
            )
        super().__init__(
            config=config, default_camera_config=default_camera_config, **kwargs
        )

        if config.kinematic_grasp:
            self._grasp_lock = KinematicGraspLock(
                self.model,
                self.data,
                self._GRASP_OBJECT_NAMES,
                engage_config=config.grasp_engage,
            )
            self._gripper_act_ids = np.array(
                [self.model.actuator("act8").id, self.model.actuator("act9").id]
            )

    def _mujoco_step(self, action):
        """Step the sim, enforcing the kinematic grasp lock between substeps.

        The default base implementation does a single mj_step(nstep=n_substeps);
        with the lock active we instead step one substep at a time and re-pin the
        held object after each, so it can't drift across the ~40 ms control
        interval. No-op fallback to the base behavior when the lock is off."""
        if self._grasp_lock is None:
            super()._mujoco_step(action)
            return
        self._update_grasp_engagement()
        for _ in range(self.n_substeps):
            self._mujoco.mj_step(self.model, self.data, nstep=1)
            self._grasp_lock.enforce()

    def _update_grasp_engagement(self):
        """Engage/release the lock from the policy's gripper command.

        Engage the closest in-range object once the gripper is commanded to
        grasp depth (gated on the close command so we don't glue an object
        during the open-gripper approach); release when it's commanded back
        open. The jaws are NOT frozen at engage time — the command fires before
        they have physically travelled, so they're left under actuator control
        to close onto the welded object and pinned once they settle. This keeps
        eval's held frames (jaws closed on the block) matching the collection
        demos, where engage happens after a completed scripted close."""
        target = float(self.data.ctrl[self._gripper_act_ids].mean())
        if self._grasp_lock.is_held:
            if target <= self._GRASP_RELEASE_CTRL:
                self._grasp_lock.release()
            else:
                self._grasp_lock.maybe_pin_jaws()
        elif target >= self._GRASP_ENGAGE_CTRL:
            self._grasp_lock.engage(pin_jaws=False, close_ctrl_target=target)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # gymnasium.Env.reset
        super(MujocoRobotEnv, self).reset(seed=seed)

        if self._grasp_lock is not None:
            self._grasp_lock.release()

        if self.config.obs_image_aug:
            self._eval_cam_profile = sample_camera_profile(
                self._obs_aug_rng, strength=self.config.obs_image_aug_strength
            )

        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal()
        self._visualize_target()
        self._reset_distractors()
        self._mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, {}

    def _visualize_target(self):
        # Visualize target.
        # Move the visual element on the floor
        visual_body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, "target_visual_body"
        )
        target_visual_pos = self.goal.copy()
        target_visual_pos[2] = 0.0
        self.model.body_pos[visual_body_id] = target_visual_pos

        # Move the target site to the goal position
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal

    @staticmethod
    def _apply_camera_offset(translation, quaterion, cam_cfg):
        translation = np.array(translation, dtype=float)
        quaterion = np.array(quaterion, dtype=float)
        if cam_cfg is None:
            return translation, quaterion
        if cam_cfg.pos_offset is not None:
            translation = translation + np.array(cam_cfg.pos_offset)
        if cam_cfg.rot_offset_euler is not None:
            offset_rot = Rotation.from_euler("xyz", cam_cfg.rot_offset_euler)
            quaterion = (Rotation.from_quat(quaterion) * offset_rot).as_quat()
        return translation, quaterion

    def _randomize_body_pose(self, body_name: str, cam_cfg) -> None:
        body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id == -1:
            return
        if cam_cfg.pos_offset is not None:
            self.model.body_pos[body_id] = self.model.body_pos[body_id] + np.array(
                cam_cfg.pos_offset
            )
        if cam_cfg.rot_offset_euler is not None:
            cur_wxyz = self.model.body_quat[body_id]
            cur_xyzw = [cur_wxyz[1], cur_wxyz[2], cur_wxyz[3], cur_wxyz[0]]
            offset_rot = Rotation.from_euler("xyz", cam_cfg.rot_offset_euler)
            new_xyzw = (Rotation.from_quat(cur_xyzw) * offset_rot).as_quat()
            self.model.body_quat[body_id] = [
                new_xyzw[3],
                new_xyzw[0],
                new_xyzw[1],
                new_xyzw[2],
            ]

    def _calculate_camera_config_from_transform(
        self,
        translation,
        quatertion,
        z_offset=0.0,
        distance_multiplier=1.0,
        use_geometric_lookat=False,
    ):
        R_cam_to_base = Rotation.from_quat(quatertion).as_matrix()

        # --- Step 1: Find the Camera's Pose (Position and Orientation) in the Base Frame ---
        # The camera's position in the base frame is the translation part of the inverse transform.
        cam_pos_in_base = -R_cam_to_base.T @ translation

        # --- Step 2: Calculate the 'look_at' point by casting a ray from the camera ---
        # The camera's viewing direction is its local -Z axis.
        # We transform this direction vector from the camera's frame to the base frame.
        local_cam_view_dir = np.array([0, 0, -1])
        look_dir_in_base = R_cam_to_base @ local_cam_view_dir

        # Intersect this viewing ray with a horizontal plane (e.g., the tabletop at z=0).
        # Ray: P(t) = ray_origin + t * look_dir_in_base
        # With use_geometric_lookat=True the ray starts from the camera's true
        # world position (geometrically correct). Legacy behavior uses the raw
        # extrinsic translation as the ray origin (preserved for backward
        # compatibility with existing datasets/policies).
        ray_origin = cam_pos_in_base if use_geometric_lookat else translation
        lookat_x = ray_origin[0] + (z_offset - ray_origin[2]) * (
            look_dir_in_base[0] / look_dir_in_base[2]
        )
        lookat_y = ray_origin[1] + (z_offset - ray_origin[2]) * (
            look_dir_in_base[1] / look_dir_in_base[2]
        )
        lookat = np.array([lookat_x, lookat_y, z_offset])

        # --- Step 3: Calculate the final simulation parameters relative to the look_at point ---
        # Vector from the new look_at point TO the camera
        vec_from_lookat_to_cam = cam_pos_in_base - lookat
        # vec_from_lookat_to_cam = cam_pos_in_base
        dx, dy, dz = vec_from_lookat_to_cam

        # Distance is the length of this vector
        # distance = np.linalg.norm(cam_pos_in_base - lookat)
        distance = np.linalg.norm(vec_from_lookat_to_cam)

        # Azimuth is the angle in the XY-plane from the X-axis
        azimuth = np.degrees(np.arctan2(dy, dx)) + 90
        # Elevation is the angle from the XY-plane
        elevation = np.degrees(np.arcsin(dz / distance)) - 90

        return {
            "distance": distance_multiplier * distance,
            "azimuth": azimuth,
            "elevation": elevation,
            "lookat": lookat,
        }

    def _step_callback(self):
        if self.block_gripper:
            self._utils.set_joint_qpos(self.model, self.data, "gripper_jaw1_joint", 0.0)
            self._utils.set_joint_qpos(self.model, self.data, "gripper_jaw2_joint", 0.0)
            self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action):
        if self.use_eef_control:
            assert action.shape == (4,)
            action = (
                action.copy()
            )  # ensure that we don't change the action outside of this scope
            pos_ctrl, gripper_ctrl = action[:3], action[3]

            pos_ctrl *= 0.05  # limit maximum change in position

            body_id = self._model_names.body_name2id["robot0:mocap"]
            mocap_id = self.model.body_mocapid[body_id]
            current_mocap_pos = self.data.mocap_pos[mocap_id]
            new_mocap_pos = current_mocap_pos + pos_ctrl
            self._utils.set_mocap_pos(
                self.model, self.data, "robot0:mocap", new_mocap_pos
            )

            # Absolute position control for the gripper (-1 closed, +1 open)
            gripper_target_pos = -0.014 * (gripper_ctrl + 1.0) / 2.0
            gripper_action = np.array([gripper_target_pos, gripper_target_pos])

            if self.block_gripper:
                gripper_action = np.zeros_like(gripper_action)

            # Apply action to simulation
            self.data.ctrl[-2:] = gripper_action
        else:
            assert action.shape == (7,)
            action = action.copy()

            if self.absolute_state_actions:
                new_target_arm_qpos = action[:6]
            else:
                # Relative position control for the arm: action[:6] is scaled and
                # added to the current joint qpos. The scale matches the policy's
                # arm-action units (see Ar4Mk3EnvConfig.relative_action_scale).
                arm_joint_deltas = action[:6] * self.config.relative_action_scale
                arm_joint_names = [f"joint_{i + 1}" for i in range(6)]
                current_arm_qpos = np.array(
                    [
                        self._utils.get_joint_qpos(self.model, self.data, name)[0]
                        for name in arm_joint_names
                    ]
                )
                new_target_arm_qpos = current_arm_qpos + arm_joint_deltas

            # Absolute position control for the gripper (-1 closed, +1 open)
            gripper_ctrl = action[6]
            gripper_target_pos = -0.014 * (gripper_ctrl + 1.0) / 2.0

            # Construct and clip the full control vector
            control_signal = np.concatenate(
                [new_target_arm_qpos, [gripper_target_pos, gripper_target_pos]]
            )
            ctrlrange = self.model.actuator_ctrlrange
            control_signal = np.clip(control_signal, ctrlrange[:, 0], ctrlrange[:, 1])

            # Apply action to simulation
            self.data.ctrl[:] = control_signal

    def generate_mujoco_observations(self):
        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = self._utils.get_site_xvelp(self.model, self.data, "grip") * dt

        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        if self.has_object:
            object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            # rotations
            object_rot = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, "object0")
            )
            # velocities
            object_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "object0") * dt
            )
            object_velr = (
                self._utils.get_site_xvelr(self.model, self.data, "object0") * dt
            )
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = (
                np.zeros(0)
            )
        gripper_state = robot_qpos[-2:]

        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        if self.config.include_images_in_obs and self.mujoco_renderer:
            default_img = self.mujoco_renderer.render(render_mode="rgb_array")
            gripper_img = self.mujoco_renderer.render(
                render_mode="rgb_array", camera_name="gripper_camera"
            )
            # Eval sensor-realism: same shared augmentation the training data
            # uses, so sim-eval images match the policy's training distribution.
            if self._eval_cam_profile is not None:
                default_img = augment_image(
                    np.asarray(default_img), self._eval_cam_profile, self._obs_aug_rng
                )
                gripper_img = augment_image(
                    np.asarray(gripper_img), self._eval_cam_profile, self._obs_aug_rng
                )
        else:
            default_img, gripper_img = None, None

        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
            default_img,
            gripper_img,
        )

    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["gripper_link"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        self._visualize_target()

        # Display grip position
        if (
            self.render_mode == "human"
            and self.mujoco_renderer.viewer
            and self.config.show_grip_overlay
        ):
            grip_pos = self._utils.get_site_xpos(self.model, self.data, "grip")
            self.mujoco_renderer.viewer.add_overlay(
                self._mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "Grip Position (x, y, z)",
                f"{grip_pos[0]:.3f}, {grip_pos[1]:.3f}, {grip_pos[2]:.3f}",
            )
            if self.has_object:
                object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
                object_rot = self._utils.get_site_xmat(
                    self.model, self.data, "object0"
                ).reshape(3, 3)
                top_offset_local = np.array([0, 0, self.object_size[2]])
                top_pos_world = object_pos + object_rot @ top_offset_local
                self.mujoco_renderer.viewer.add_overlay(
                    self._mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Object Top (x, y, z)",
                    f"{top_pos_world[0]:.3f}, {top_pos_world[1]:.3f}, {top_pos_world[2]:.3f}",
                )

        self._mujoco.mj_forward(self.model, self.data)

    _PROP_ASSET_IDS_CACHE: list[str] | None = None

    @classmethod
    def _load_prop_asset_ids(cls) -> list[str]:
        """List of asset_ids that the current props.xml has bodies for. The
        runtime iterates this to hide every body, then makes a subset visible.
        Empty list (no sidecar / fresh clone) makes the prop pass a no-op."""
        if cls._PROP_ASSET_IDS_CACHE is not None:
            return cls._PROP_ASSET_IDS_CACHE
        sidecar_path = (
            Path(__file__).resolve().parents[2]
            / "autonomous" / "simulation" / "props" / "_scene_assets.json"
        )
        if not sidecar_path.exists():
            cls._PROP_ASSET_IDS_CACHE = []
            return cls._PROP_ASSET_IDS_CACHE
        raw = json.loads(sidecar_path.read_text())
        cls._PROP_ASSET_IDS_CACHE = list(raw.get("asset_ids", []))
        return cls._PROP_ASSET_IDS_CACHE

    def _apply_domain_randomization(self):
        """Applies domain randomization settings from the config to the Mujoco model."""
        if not self.config.domain_rand:
            return

        dr_config = self.config.domain_rand

        # --- Apply Material Properties ---
        materials_map = {
            "object_material": "object_mat",
            "target_material": "target_mat",
            "distractor1_material": "distractor1_mat",
            "distractor2_material": "distractor2_mat",
            "object_distractor1_material": "object_distractor1_mat",
            "object_distractor2_material": "object_distractor2_mat",
            "floor_material": "groundplane",
            "wall_material": "wallmaterial",
            "table_material": "tablemat",
            "base_link_material": "base_link_mat",
            "link_1_material": "link_1_mat",
            "link_2_material": "link_2_mat",
            "link_3_material": "link_3_mat",
            "link_4_material": "link_4_mat",
            "link_5_material": "link_5_mat",
            "link_6_material": "link_6_mat",
            "gripper_base_link_material": "gripper_base_link_mat",
            "gripper_jaw1_material": "gripper_jaw1_mat",
            "gripper_jaw2_material": "gripper_jaw2_mat",
        }
        for config_key, mat_name in materials_map.items():
            mat_config = getattr(dr_config, config_key)
            if mat_config:
                mat_id = self._mujoco.mj_name2id(
                    self.model, self._mujoco.mjtObj.mjOBJ_MATERIAL, mat_name
                )
                if mat_id != -1:
                    texture_name_to_apply = None
                    if mat_config.texture_name:
                        if isinstance(mat_config.texture_name, str):
                            texture_name_to_apply = mat_config.texture_name
                        else:  # It's a sequence
                            texture_name_to_apply = np.random.choice(
                                mat_config.texture_name
                            )

                    if texture_name_to_apply:
                        tex_id = self._mujoco.mj_name2id(
                            self.model,
                            self._mujoco.mjtObj.mjOBJ_TEXTURE,
                            texture_name_to_apply,
                        )
                        if tex_id != -1:
                            self.model.mat_texid[mat_id] = tex_id
                            # When using a texture, we can also apply a color tint.
                            # If no color is specified, default to white to show original texture colors.
                            if mat_config.rgba is not None:
                                self.model.mat_rgba[mat_id] = mat_config.rgba
                            else:
                                self.model.mat_rgba[mat_id] = [1.0, 1.0, 1.0, 1.0]
                        else:
                            print(
                                f"Warning: Texture '{texture_name_to_apply}' not found in model."
                            )
                    elif mat_config.rgba is not None:
                        self.model.mat_rgba[mat_id] = mat_config.rgba

                    if mat_config.specular is not None:
                        self.model.mat_specular[mat_id] = mat_config.specular
                    if mat_config.shininess is not None:
                        self.model.mat_shininess[mat_id] = mat_config.shininess
                    if mat_config.reflectance is not None:
                        self.model.mat_reflectance[mat_id] = mat_config.reflectance
                    if mat_config.texrepeat is not None:
                        self.model.mat_texrepeat[mat_id] = mat_config.texrepeat

        # --- Apply Light Properties ---
        lights_map = {
            "top_light": "top",
            "scene_light": "scene_light",
        }
        for config_key, light_name in lights_map.items():
            light_config = getattr(dr_config, config_key)
            if light_config:
                light_id = self._mujoco.mj_name2id(
                    self.model, self._mujoco.mjtObj.mjOBJ_LIGHT, light_name
                )
                if light_id != -1:
                    if light_config.active is not None:
                        self.model.light_active[light_id] = int(light_config.active)
                    if light_config.pos is not None:
                        self.model.light_pos[light_id] = light_config.pos
                    if light_config.dir is not None:
                        self.model.light_dir[light_id] = light_config.dir
                    if light_config.diffuse is not None:
                        self.model.light_diffuse[light_id] = light_config.diffuse
                    if light_config.ambient is not None:
                        self.model.light_ambient[light_id] = light_config.ambient
                    if light_config.specular is not None:
                        self.model.light_specular[light_id] = light_config.specular

        if dr_config.headlight:
            if dr_config.headlight.diffuse is not None:
                self.model.vis.headlight.diffuse = dr_config.headlight.diffuse
            if dr_config.headlight.ambient is not None:
                self.model.vis.headlight.ambient = dr_config.headlight.ambient
            if dr_config.headlight.specular is not None:
                self.model.vis.headlight.specular = dr_config.headlight.specular

        # --- Apply Table Geometry ---
        # Visual `floor` slab gets the sampled top size and centerline. The
        # invisible `table_collision` box mirrors the xy footprint of the
        # visual top so the table-edge silhouette matches what's rendered, but
        # keeps its deep z-extent (set in XML) — objects can't tunnel into a
        # thick box because the contact MTV always resolves toward the +z face.
        if dr_config.table:
            table_cfg = dr_config.table
            if table_cfg.top_half_size is not None:
                top_id = self._mujoco.mj_name2id(
                    self.model, self._mujoco.mjtObj.mjOBJ_GEOM, "floor"
                )
                if top_id != -1:
                    self.model.geom_size[top_id] = table_cfg.top_half_size
                    if table_cfg.top_pos is not None:
                        self.model.geom_pos[top_id] = table_cfg.top_pos
                    else:
                        self.model.geom_pos[top_id] = [
                            0.0,
                            0.0,
                            -table_cfg.top_half_size[2],
                        ]
                col_id = self._mujoco.mj_name2id(
                    self.model, self._mujoco.mjtObj.mjOBJ_GEOM, "table_collision"
                )
                if col_id != -1:
                    col_size = self.model.geom_size[col_id].copy()
                    col_size[0] = table_cfg.top_half_size[0]
                    col_size[1] = table_cfg.top_half_size[1]
                    self.model.geom_size[col_id] = col_size
                    if table_cfg.top_pos is not None:
                        col_pos = self.model.geom_pos[col_id].copy()
                        col_pos[0] = table_cfg.top_pos[0]
                        col_pos[1] = table_cfg.top_pos[1]
                        self.model.geom_pos[col_id] = col_pos
            if table_cfg.pedestal_half_size is not None:
                ped_id = self._mujoco.mj_name2id(
                    self.model, self._mujoco.mjtObj.mjOBJ_GEOM, "table_pedestal"
                )
                if ped_id != -1:
                    self.model.geom_size[ped_id] = table_cfg.pedestal_half_size
                    if table_cfg.pedestal_pos is not None:
                        self.model.geom_pos[ped_id] = table_cfg.pedestal_pos

        # --- Apply Gripper Camera Pose ---
        if dr_config.gripper_camera:
            self._randomize_body_pose("gripper_camera_body", dr_config.gripper_camera)

        # --- Apply Background Props ---
        # props.xml declares one body+geom per asset (e.g. prop_body_ycb_025_mug
        # paired with prop_geom_ycb_025_mug). Each geom is compile-time bound to
        # its specific mesh, which is the only way MuJoCo's mesh-specific
        # geom_pos/quat end up correct — runtime writes to those arrays are
        # silently ignored for mesh geoms. So the runtime here only toggles
        # body pose + geom alpha; it never swaps geom_dataid.
        if dr_config.props:
            # Index the per-scene config by asset for O(1) lookup. The same
            # asset can't appear twice (one body per asset), so the sampler
            # already deduplicates.
            active_by_aid = {
                p.asset_id: p for p in dr_config.props
                if p.active and p.asset_id is not None
            }
            for aid in self._load_prop_asset_ids():
                body_id = self._mujoco.mj_name2id(
                    self.model, self._mujoco.mjtObj.mjOBJ_BODY, f"prop_body_{aid}"
                )
                geom_id = self._mujoco.mj_name2id(
                    self.model, self._mujoco.mjtObj.mjOBJ_GEOM, f"prop_geom_{aid}"
                )
                if body_id == -1 or geom_id == -1:
                    continue
                prop = active_by_aid.get(aid)
                if prop is None:
                    self.model.geom_rgba[geom_id, 3] = 0.0
                    continue
                self.model.body_pos[body_id] = prop.pos
                self.model.body_quat[body_id] = prop.quat
                self.model.geom_rgba[geom_id] = [1.0, 1.0, 1.0, 1.0]

        # --- Apply Wall Art ---
        # scene.xml declares two materials for the wall_art geom: wallartmat
        # (texture-bound, used for paintings) and wallboardmat (untextured,
        # used for solid-color boards). The runtime swaps geom_matid between
        # them — this is cleaner than rebinding textures on a single material
        # because a material compiled without `texture=` won't render
        # textures even if mat_texid is written at runtime.
        if dr_config.wall_art:
            geom_id = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_GEOM, "wall_art"
            )
            art_mat_id = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_MATERIAL, "wallartmat"
            )
            board_mat_id = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_MATERIAL, "wallboardmat"
            )
            if geom_id != -1 and art_mat_id != -1 and board_mat_id != -1:
                wa = dr_config.wall_art
                if not wa.active:
                    self.model.geom_rgba[geom_id, 3] = 0.0
                else:
                    if wa.pos is not None:
                        self.model.geom_pos[geom_id] = wa.pos
                    if wa.half_size is not None:
                        # WallArtConfig.half_size carries (thickness, wy, wz)
                        # in world frame. The plane's quat rotates local
                        # (X, Y, Z) → world (Y, Z, X), so local x = world-y
                        # half-extent and local y = world-z half-extent.
                        # size[2] is the plane's infinite-grid spacing.
                        self.model.geom_size[geom_id] = [
                            wa.half_size[1], wa.half_size[2], 1.0
                        ]
                    if wa.texture_name is not None:
                        tex_id = self._mujoco.mj_name2id(
                            self.model,
                            self._mujoco.mjtObj.mjOBJ_TEXTURE,
                            wa.texture_name,
                        )
                        if tex_id != -1:
                            self.model.mat_texid[art_mat_id] = tex_id
                        self.model.geom_matid[geom_id] = art_mat_id
                        self.model.mat_rgba[art_mat_id] = [1.0, 1.0, 1.0, 1.0]
                        self.model.geom_rgba[geom_id] = [1.0, 1.0, 1.0, 1.0]
                    else:
                        self.model.geom_matid[geom_id] = board_mat_id
                        if wa.rgba is not None:
                            self.model.mat_rgba[board_mat_id] = wa.rgba
                            self.model.geom_rgba[geom_id] = wa.rgba

        # --- Apply Block Visual Preset (edge shape + size) ---
        # Each pickable block carries one visual mesh geom per PLA_BLOCK_PRESETS
        # entry. Show the selected one, hide the rest, and scale the collision
        # BOX to the preset's half-size (mesh geoms can't be scaled at runtime, so
        # the visual size is the discrete preset and the box is matched to it).
        # Showing a geom = reset geom_rgba to the default sentinel
        # [0.5,0.5,0.5,1] so the block's material (DR color + texture) takes
        # precedence; hiding = alpha 0. Any other geom_rgba value would override
        # the material color, so the sentinel is load-bearing here. The resting
        # half-height is recorded for the spawn pose in _reset_sim.
        block_variant_map = {
            "object0": dr_config.object_block_variant,
            "object_distractor1": dr_config.object_distractor1_block_variant,
            "object_distractor2": dr_config.object_distractor2_block_variant,
        }
        for base, selected in block_variant_map.items():
            if selected is None:
                continue
            for vi in range(len(PLA_BLOCK_PRESETS)):
                geom_id = self._mujoco.mj_name2id(
                    self.model,
                    self._mujoco.mjtObj.mjOBJ_GEOM,
                    f"{base}_visual_p{vi}",
                )
                if geom_id == -1:
                    continue
                self.model.geom_rgba[geom_id] = (
                    [0.5, 0.5, 0.5, 1.0] if vi == selected
                    else [0.5, 0.5, 0.5, 0.0]
                )
            half = float(PLA_BLOCK_PRESETS[selected][1])
            box_id = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_GEOM, base
            )
            if box_id != -1:
                self.model.geom_size[box_id] = [half, half, half]
                # geom_size alone is not enough: the broadphase uses the
                # compile-time geom_aabb / geom_rbound, which don't track a
                # runtime size change. In the full scene (AABB-tree broadphase)
                # a stale-small AABB clips the box so it collides at the OLD
                # size (the block sinks into the table). Refresh both to match.
                self.model.geom_aabb[box_id] = [0.0, 0.0, 0.0, half, half, half]
                self.model.geom_rbound[box_id] = half * np.sqrt(3.0)
            self._block_half_size[base] = half

        # --- Apply Arm Cables (geometry / silhouette DR) ---
        # Visual-only capsule "tubes" parented to the link bodies — the only DR
        # axis that changes the arm's shape rather than its appearance. These
        # geoms are non-colliding (contype/conaffinity=0), so unlike the block
        # collision box we can rewrite geom_size/geom_pos freely without
        # refreshing the broadphase AABB. Base positions are cached once because
        # the pos offset is additive and the model persists across episodes.
        if dr_config.arm_cables:
            for seg in dr_config.arm_cables.segments:
                geom_id = self._mujoco.mj_name2id(
                    self.model, self._mujoco.mjtObj.mjOBJ_GEOM, seg.geom_name
                )
                if geom_id == -1:
                    continue
                if geom_id not in self._cable_base_pos:
                    self._cable_base_pos[geom_id] = self.model.geom_pos[
                        geom_id
                    ].copy()
                if not seg.active:
                    self.model.geom_rgba[geom_id, 3] = 0.0
                    continue
                if seg.rgba is not None:
                    self.model.geom_rgba[geom_id] = seg.rgba
                if seg.radius is not None:
                    # size[0] is the capsule radius; size[1] (fromto half-length)
                    # is left as compiled so the tube still spans the same joints.
                    self.model.geom_size[geom_id, 0] = seg.radius
                base = self._cable_base_pos[geom_id]
                if seg.pos_offset is not None:
                    self.model.geom_pos[geom_id] = base + np.array(seg.pos_offset)
                else:
                    self.model.geom_pos[geom_id] = base

            # Arched cables: bend each chain of sub-capsules into a quadratic
            # arc through start -> (midpoint + apex_offset) -> end so the cable
            # bows off the link. All sub-geom poses are written absolutely
            # (recomputed from the config), so unlike the straight segments
            # there's no accumulation across episodes and no base-pos cache.
            for arc in dr_config.arm_cables.arcs:
                start = np.array(arc.start, dtype=float)
                end = np.array(arc.end, dtype=float)
                ctrl = (start + end) / 2.0 + 2.0 * np.array(
                    arc.apex_offset, dtype=float
                )
                ts = np.linspace(0.0, 1.0, arc.n_segments + 1)
                pts = [
                    (1 - t) ** 2 * start + 2 * (1 - t) * t * ctrl + t**2 * end
                    for t in ts
                ]
                for i in range(arc.n_segments):
                    geom_id = self._mujoco.mj_name2id(
                        self.model,
                        self._mujoco.mjtObj.mjOBJ_GEOM,
                        f"{arc.base_name}_s{i}",
                    )
                    if geom_id == -1:
                        continue
                    if not arc.active:
                        self.model.geom_rgba[geom_id, 3] = 0.0
                        continue
                    p0, p1 = pts[i], pts[i + 1]
                    seg_vec = p1 - p0
                    half_len = float(np.linalg.norm(seg_vec)) / 2.0
                    self.model.geom_pos[geom_id] = (p0 + p1) / 2.0
                    self.model.geom_quat[geom_id] = _quat_z_to(seg_vec)
                    # size[1] is the capsule half-length; size[0] the radius.
                    self.model.geom_size[geom_id, 1] = half_len
                    if arc.radius is not None:
                        self.model.geom_size[geom_id, 0] = arc.radius
                    if arc.rgba is not None:
                        self.model.geom_rgba[geom_id] = arc.rgba

        # --- Apply Arm Actuator / Joint Physics (movement DR) ---
        # The only DR axis that changes how the arm tracks commands rather than
        # how it looks. Values are absolute per-joint (resolved by the sampler),
        # so each field is written directly with no base caching. A MuJoCo
        # position actuator computes force = kp*(ctrl - qpos) - kv*qvel, encoded
        # as gainprm[0]=kp and biasprm=[0, -kp, -kv] — so kp must be written to
        # BOTH gainprm and biasprm[1] to stay consistent.
        if dr_config.arm_dynamics:
            ad = dr_config.arm_dynamics
            for i in range(6):
                joint_id = self._mujoco.mj_name2id(
                    self.model, self._mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i + 1}"
                )
                if joint_id != -1:
                    dof_adr = self.model.jnt_dofadr[joint_id]
                    if ad.damping is not None:
                        self.model.dof_damping[dof_adr] = ad.damping[i]
                    if ad.armature is not None:
                        self.model.dof_armature[dof_adr] = ad.armature[i]
                    if ad.frictionloss is not None:
                        self.model.dof_frictionloss[dof_adr] = ad.frictionloss[i]
                act_id = self._mujoco.mj_name2id(
                    self.model, self._mujoco.mjtObj.mjOBJ_ACTUATOR, f"act{i + 1}"
                )
                if act_id != -1:
                    if ad.kp is not None:
                        self.model.actuator_gainprm[act_id, 0] = ad.kp[i]
                        self.model.actuator_biasprm[act_id, 1] = -ad.kp[i]
                    if ad.kv is not None:
                        self.model.actuator_biasprm[act_id, 2] = -ad.kv[i]
                    if ad.force_limit is not None:
                        f = ad.force_limit[i]
                        self.model.actuator_forcerange[act_id] = [-f, f]

        # --- Apply Dynamics Properties ---
        dynamics_map = {
            "object_dynamics": "object0",
            "object_distractor1_dynamics": "object_distractor1",
            "object_distractor2_dynamics": "object_distractor2",
        }
        for config_key, object_name in dynamics_map.items():
            dyn_config = getattr(dr_config, config_key)
            if dyn_config:
                if dyn_config.mass is not None:
                    body_id = self._mujoco.mj_name2id(
                        self.model, self._mujoco.mjtObj.mjOBJ_BODY, object_name
                    )
                    if body_id != -1:
                        self.model.body_mass[body_id] = dyn_config.mass
                if dyn_config.damping is not None:
                    joint_id = self._mujoco.mj_name2id(
                        self.model,
                        self._mujoco.mjtObj.mjOBJ_JOINT,
                        f"{object_name}:joint",
                    )
                    if joint_id != -1:
                        dof_adr = self.model.jnt_dofadr[joint_id]
                        # A free joint has 6 DoFs
                        self.model.dof_damping[dof_adr : dof_adr + 6] = (
                            dyn_config.damping
                        )
                if dyn_config.friction is not None:
                    geom_id = self._mujoco.mj_name2id(
                        self.model, self._mujoco.mjtObj.mjOBJ_GEOM, object_name
                    )
                    if geom_id != -1:
                        self.model.geom_friction[geom_id] = dyn_config.friction
                if dyn_config.size is not None:
                    geom_id = self._mujoco.mj_name2id(
                        self.model, self._mujoco.mjtObj.mjOBJ_GEOM, object_name
                    )
                    if geom_id != -1:
                        self.model.geom_size[geom_id] = dyn_config.size

    def _reset_distractors(self):
        """Randomize start position of distractors, ensuring they don't overlap."""
        distractor_body_names = [
            "target_distractor1_visual_body",
            "target_distractor2_visual_body",
        ]
        placed_positions_2d = []
        min_dist = 0.06  # Minimum distance between centers of objects

        # Ensure distractors don't overlap with the target visual
        placed_positions_2d.append(self.goal[:2])

        if self.has_object:
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            placed_positions_2d.append(object_qpos[:2])

        for body_name in distractor_body_names:
            body_id = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_BODY, body_name
            )

            while True:
                distractor_pos_2d = self.initial_gripper_xpos[
                    :2
                ] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=2
                )

                if all(
                    np.linalg.norm(distractor_pos_2d - pos) >= min_dist
                    for pos in placed_positions_2d
                ):
                    placed_positions_2d.append(distractor_pos_2d)
                    distractor_pos_3d = np.array(
                        [distractor_pos_2d[0], distractor_pos_2d[1], 0.0]
                    )
                    self.model.body_pos[body_id] = distractor_pos_3d
                    break

    def _apply_spawn_yaw(self, qpos):
        """Write a random yaw (about +Z) into a free-joint qpos[3:7] when
        randomize_object_yaw is on, so blocks spawn rotated and the gripper
        practices non-parallel grasps. No-op (orientation left as-is) when off,
        so default behavior is unchanged."""
        if not self.config.randomize_object_yaw:
            return qpos
        r = self.config.object_yaw_range
        yaw = self.np_random.uniform(-r, r)
        qpos[3:7] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        return qpos

    def _reset_sim(self):
        # Reset buffers for joint states, actuators, warm-start, control buffers etc.
        self._mujoco.mj_resetData(self.model, self.data)

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None
        if self.use_eef_control:
            self._mujoco.mj_forward(self.model, self.data)
            gripper_body_id = self._model_names.body_name2id["gripper_base_link"]
            mocap_pos = self.data.xpos[gripper_body_id]
            mocap_quat = self.data.xquat[gripper_body_id]
            self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", mocap_pos)
            self._utils.set_mocap_quat(
                self.model, self.data, "robot0:mocap", mocap_quat
            )

        # Randomize start position of object.
        if self.has_object:
            placed_object_positions_2d = []
            min_dist = 0.06  # Minimum distance between centers of objects

            # Position for object0
            object_xpos = self.initial_gripper_xpos[:2].copy()
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.05:
                object_xpos[0] = (
                    self.initial_gripper_xpos[0]
                    + self.np_random.uniform(-self.obj_range[0], self.obj_range[0])
                    + self.obj_offset[0]
                )
                object_xpos[1] = (
                    self.initial_gripper_xpos[1]
                    + self.np_random.uniform(-self.obj_range[1], self.obj_range[1])
                    + self.obj_offset[1]
                )
            placed_object_positions_2d.append(object_xpos)

            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            object_qpos[2] = self._block_half_size.get(
                "object0", self.object_size[2]
            )
            self._apply_spawn_yaw(object_qpos)
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

            # Now position distractors
            object_distractor_joint_names = [
                "object_distractor1:joint",
                "object_distractor2:joint",
            ]
            for joint_name in object_distractor_joint_names:
                while True:
                    distractor_pos_2d = self.initial_gripper_xpos[
                        :2
                    ] + self.np_random.uniform(
                        -self.target_range, self.target_range, size=2
                    )
                    if all(
                        np.linalg.norm(distractor_pos_2d - pos) >= min_dist
                        for pos in placed_object_positions_2d
                    ):
                        placed_object_positions_2d.append(distractor_pos_2d)
                        distractor_qpos = self._utils.get_joint_qpos(
                            self.model, self.data, joint_name
                        )
                        distractor_qpos[:2] = distractor_pos_2d
                        distractor_qpos[2] = self._block_half_size.get(
                            joint_name.split(":")[0], self.object_size[2]
                        )
                        self._apply_spawn_yaw(distractor_qpos)
                        self._utils.set_joint_qpos(
                            self.model, self.data, joint_name, distractor_qpos
                        )
                        break

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)

        if self.use_eef_control:
            self._utils.reset_mocap_welds(self.model, self.data)
            self._mujoco.mj_forward(self.model, self.data)
            gripper_body_id = self._model_names.body_name2id["gripper_base_link"]
            mocap_pos = self.data.xpos[gripper_body_id]
            mocap_quat = self.data.xquat[gripper_body_id]
            self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", mocap_pos)
            self._utils.set_mocap_quat(
                self.model, self.data, "robot0:mocap", mocap_quat
            )

        if self.has_object:
            geom_id = self._model_names.geom_name2id["object0"]
            self.model.geom_size[geom_id] = self.object_size
            distractor1_geom_id = self._model_names.geom_name2id["object_distractor1"]
            self.model.geom_size[distractor1_geom_id] = self.object_size
            distractor2_geom_id = self._model_names.geom_name2id["object_distractor2"]
            self.model.geom_size[distractor2_geom_id] = self.object_size

        self._apply_domain_randomization()

        self._mujoco.mj_forward(self.model, self.data)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "grip"
        ).copy()

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal[:2] += (
                self.target_offset
                if isinstance(self.target_offset, float)
                else self.target_offset[:2]
            )
            goal[2] = (
                self.target_offset
                if isinstance(self.target_offset, float)
                else self.target_offset[2]
            )
            if self.target_in_the_air:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
        return goal.copy()