import gymnasium as gym
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# Import to register the environments
import aera.autonomous.envs as aera_envs

T = np.array([0.6233588611899381, 0.05979687559388906, 0.7537742046170788])
Q = (
    -0.36336720179946663,
    -0.8203835174702869,
    0.22865474664402222,
    0.37769321910336584,
)


def display_video(frames, framerate=30, filename="animated_arm.mp4"):
    """Fast video rendering using OpenCV - much faster than matplotlib."""
    print(f"Starting OpenCV video rendering with {len(frames)} frames...")
    start_time = time.time()

    if not frames:
        print("No frames to render!")
        return

    # Convert frames to proper format (0-255 uint8)
    height, width, channels = frames[0].shape

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, framerate, (width, height))

    write_start_time = time.time()
    for i, frame in enumerate(frames):
        # Convert from RGB to BGR (OpenCV uses BGR)
        frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    write_end_time = time.time()

    total_time = write_end_time - start_time
    write_time = write_end_time - write_start_time

    print(f"video rendering completed in {total_time:.2f}s (write: {write_time:.2f}s)")


R_cam_to_base = Rotation.from_quat(Q).as_matrix()

# --- Step 1: Find the Camera's Pose (Position and Orientation) in the Base Frame ---
# The camera's position in the base frame is the translation part of the inverse transform.
cam_pos_in_base = -R_cam_to_base.T @ T

# --- Step 2: Calculate the 'look_at' point by casting a ray from the camera ---
# The camera's viewing direction is its local -Z axis.
# We transform this direction vector from the camera's frame to the base frame.
local_cam_view_dir = np.array([0, 0, -1])
look_dir_in_base = R_cam_to_base @ local_cam_view_dir

# Intersect this viewing ray with a horizontal plane (e.g., the tabletop at z=0).
# Ray: P(t) = cam_pos_in_base + t * look_dir_in_base
# Plane: P.z = intersection_plane_z

intersection_plane_z = 0

# P_origin.z + t * LookDir.z = plane_z  =>  t = (plane_z - P_origin.z) / LookDir.z
t = (intersection_plane_z - cam_pos_in_base[2]) / look_dir_in_base[2]
lookat_z = 0.3
P = T
lookat_x = P[0] + (lookat_z - P[2]) * (look_dir_in_base[0] / look_dir_in_base[2])
lookat_y = P[1] + (lookat_z - P[2]) * (look_dir_in_base[1] / look_dir_in_base[2])
lookat = (1) * np.array([lookat_x, lookat_y, lookat_z])

# We only want to look "in front" of the camera, so t should be positive.
# If t is negative, the plane is behind the camera. The math still works perfectly.
# lookat = cam_pos_in_base + t * look_dir_in_base

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


camera_config = {
    "distance": 1.2 * distance,
    "azimuth": azimuth,
    "elevation": elevation,
    "lookat": lookat,
}

env = gym.make(
    "Ar4Mk3PickAndPlaceEefDenseEnv-v1",
    render_mode="rgb_array",
    width=1920,
    height=1080,
    max_episode_steps=1000,
    default_camera_config=camera_config,
)
env.reset()
terminated, truncated = False, False
ep_lens = 0
frames = []
frames.append(env.render() / 255)

print("Starting frame collection...")
frame_collection_start = time.time()

while not (terminated or truncated) and ep_lens < 100:
    # action = [5, 0, 0, 0, 0, 0, 0]
    action = env.action_space.sample()
    # print(action)
    state, reward, terminated, truncated, info = env.step(action)
    ep_lens += 1
    observation = state["observation"]

    frame_start = time.time()
    frames.append(env.render() / 255)
    frame_end = time.time()

frame_collection_end = time.time()
print(
    f"Frame collection completed in {frame_collection_end - frame_collection_start:.2f}s"
)

# Test both methods for comparison
print("\n=== Testing OpenCV method ===")
display_video(frames, filename="animated_arm.mp4")

# For fastest testing, you can skip video creation entirely:
# print("Skipping video creation for speed testing")
