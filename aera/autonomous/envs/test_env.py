import gymnasium as gym
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Import to register the environments
import aera.autonomous.envs


def display_video(frames, framerate=30):
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
    plt.show()
    # anim.save('animated_arm.gif', fps=24, writer='imagemagick')
    anim.save("animated_arm.mp4", fps=24, extra_args=["-vcodec", "libx264"])
    # return HTML(anim.to_html5_video())


env = gym.make(
    "Ar4Mk3PickAndPlaceDenseEnv-v1", render_mode="rgb_array", max_episode_steps=1000
)
env.reset()
terminated, truncated = False, False
ep_lens = 0
frames = []
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
    frames.append(env.render() / 255)

display_video(frames)
