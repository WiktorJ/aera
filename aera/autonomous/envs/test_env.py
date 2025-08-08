import gymnasium as gym
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import time

# Import to register the environments
import aera.autonomous.envs as aera_envs


def display_video(frames, framerate=30):
    print(f"Starting video rendering with {len(frames)} frames...")
    start_time = time.time()

    height, width, _ = frames[0].shape
    dpi = 32
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False
    )
    # plt.show()
    # anim.save('animated_arm.gif', fps=24, writer='imagemagick')

    save_start_time = time.time()
    anim.save("animated_arm.mp4", fps=24, extra_args=["-vcodec", "libx264"])
    save_end_time = time.time()

    total_time = save_end_time - start_time
    save_time = save_end_time - save_start_time

    print(f"Video rendering completed in {total_time:.2f}s (save: {save_time:.2f}s)")
    # return HTML(anim.to_html5_video())


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

frame_collection_end = time.time()
print(
    f"Frame collection completed in {frame_collection_end - frame_collection_start:.2f}s"
)

display_video(frames)
