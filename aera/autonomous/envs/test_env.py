import gymnasium as gym
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np

# Import to register the environments
import aera.autonomous.envs as aera_envs


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
        if i % 10 == 0:  # Log progress every 10 frames
            print(f"  Written frame {i + 1}/{len(frames)}")

    out.release()
    write_end_time = time.time()

    total_time = write_end_time - start_time
    write_time = write_end_time - write_start_time

    print(f"video rendering completed in {total_time:.2f}s (write: {write_time:.2f}s)")


env = gym.make(
    "Ar4Mk3PickAndPlaceDenseEnv-v1", render_mode="rgb_array", max_episode_steps=1000
)
env.reset()
terminated, truncated = False, False
ep_lens = 0
frames = []

print("Starting frame collection...")
frame_collection_start = time.time()

while not (terminated or truncated) and ep_lens < 10:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    ep_lens += 1
    if hasattr(state, "__getitem__") and "observation" in state:
        print("A")
        observation = state["observation"]
    else:
        print("B")
        observation = state

    frame_start = time.time()
    frames.append(env.render() / 255)
    frame_end = time.time()
    print(f"Frame {ep_lens} rendered in {(frame_end - frame_start) * 1000:.1f}ms")

frame_collection_end = time.time()
print(
    f"Frame collection completed in {frame_collection_end - frame_collection_start:.2f}s"
)

# Test both methods for comparison
print("\n=== Testing OpenCV method ===")
display_video(frames)

# For fastest testing, you can skip video creation entirely:
# print("Skipping video creation for speed testing")
