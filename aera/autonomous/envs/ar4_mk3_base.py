import numpy as np

from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations
from scipy.spatial.transform import Rotation

from aera.autonomous.envs.ar4_mk3_config import Ar4Mk3EnvConfig


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class BaseEnv(MujocoRobotEnv):
    """Superclass for all Fetch environments."""

    def __init__(
        self,
        config: Ar4Mk3EnvConfig,
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
        if isinstance(config.object_size, (float, int)):
            self.object_size = np.array([config.object_size] * 3)
        else:
            self.object_size = np.array(config.object_size)
        assert self.object_size.shape == (3,), (
            "object_size must be a float or a 3-element array-like"
        )
        self.use_eef_control = config.use_eef_control

        super().__init__(
            model_path=config.model_path,
            n_substeps=config.n_substeps,
            n_actions=4 if config.use_eef_control else 7,
            **kwargs,
        )

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
        ) = self.generate_mujoco_observations()

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate(
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

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

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
    def __init__(
        self,
        config: Ar4Mk3EnvConfig,
        **kwargs,
    ):
        default_camera_config = config.default_camera_config
        if config.translation is not None and config.quaterion is not None:
            default_camera_config = self._calculate_camera_config_from_transform(
                config.translation,
                config.quaterion,
                config.z_offset,
                config.distance_multiplier,
            )
        super().__init__(
            config=config, default_camera_config=default_camera_config, **kwargs
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # gymnasium.Env.reset
        super(MujocoRobotEnv, self).reset(seed=seed)

        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal()
        self._reset_distractors()
        self._mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, {}

    def _calculate_camera_config_from_transform(
        self, translation, quatertion, z_offset=0.0, distance_multiplier=1.0
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
        # Ray: P(t) = cam_pos_in_base + t * look_dir_in_base
        # Plane: P.z = intersection_plane_z

        # P_origin.z + t * LookDir.z = plane_z  =>  t = (plane_z - P_origin.z) / LookDir.z
        lookat_x = translation[0] + (z_offset - translation[2]) * (
            look_dir_in_base[0] / look_dir_in_base[2]
        )
        lookat_y = translation[1] + (z_offset - translation[2]) * (
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

            # Absolute position control for the gripper (+1 closed, -1 open)
            gripper_target_pos = -0.014 * (gripper_ctrl + 1.0) / 2.0
            gripper_action = np.array([gripper_target_pos, gripper_target_pos])

            if self.block_gripper:
                gripper_action = np.zeros_like(gripper_action)

            # Apply action to simulation
            self.data.ctrl[-2:] = gripper_action
        else:
            assert action.shape == (7,)
            action = action.copy()

            # Relative position control for the arm
            arm_joint_deltas = action[:6] * 0.05  # Max 0.05 rad change per step
            arm_joint_names = [f"joint_{i + 1}" for i in range(6)]
            current_arm_qpos = np.array(
                [
                    self._utils.get_joint_qpos(self.model, self.data, name)[0]
                    for name in arm_joint_names
                ]
            )
            new_target_arm_qpos = current_arm_qpos + arm_joint_deltas

            # Absolute position control for the gripper (+1 closed, -1 open)
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
        )

    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["gripper_link"]
        return self.data.xpos[body_id]

    def _render_callback(self):
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

        # Display grip position
        if self.render_mode == "human" and self.mujoco_renderer.viewer:
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

    def _reset_distractors(self):
        """Randomize start position of distractors, ensuring they don't overlap."""
        distractor_body_names = ["distractor1_visual_body", "distractor2_visual_body"]
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
            object_xpos = self.initial_gripper_xpos[:2].copy()
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.05:
                object_xpos[0] = (
                    self.initial_gripper_xpos[0]
                    + self.np_random.uniform(
                        -self.obj_range[0], self.obj_range[0], size=1
                    )
                    + self.obj_offset[0]
                )
                object_xpos[1] = (
                    self.initial_gripper_xpos[1]
                    + self.np_random.uniform(
                        -self.obj_range[1], self.obj_range[1], size=1
                    )
                    + self.obj_offset[1]
                )

            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            object_qpos[2] = self.object_size[2]
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

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
