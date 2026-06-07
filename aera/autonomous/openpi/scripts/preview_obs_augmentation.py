#!/usr/bin/env python3
"""Visual check for the observation augmentation (aera.autonomous.obs_augmentation).

Renders a grid of ``original | aug | aug | ...`` columns, one block of rows per
sampled per-episode CameraProfile, so you can eyeball whether the sensor-realism
transforms (noise / blur / vignette / white-balance / jpeg / grayscale ...) look
plausible and tune magnitudes. Also prints the sampled state-noise so you can
sanity-check proprioception jitter/bias.

Frame source (in priority order):
  --from-sim         render a real frame from the env (default + gripper cams),
                     with domain randomization on. Needs a GL context.
  --image PATH       load a frame from disk.
  (neither)          a synthetic gradient + shapes, so the tool runs anywhere.

Examples:
    # Live sim frames, 4 augmented samples across 3 camera profiles:
    python -m aera.autonomous.openpi.scripts.preview_obs_augmentation \
        --from-sim --n-samples 4 --n-profiles 3 --out /tmp/aug_preview.png

    # From a saved image, stronger augmentation:
    python -m aera.autonomous.openpi.scripts.preview_obs_augmentation \
        --image frame.png --strength 1.0 --show
"""

import argparse
import logging
import os

import cv2
import numpy as np

from aera.autonomous.obs_augmentation import (
    augment_image,
    sample_camera_profile,
    sample_state_noise_profile,
    apply_state_noise,
)


def _synthetic_frame(size: int = 224) -> np.ndarray:
    """A colored gradient with a few shapes — enough to read blur/noise/WB."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.stack(
        [xx * 255, yy * 255, (1 - xx) * 255], axis=-1
    ).astype(np.uint8)
    cv2.circle(img, (size // 3, size // 2), size // 6, (240, 240, 30), -1)
    cv2.rectangle(img, (size // 2, size // 4), (size - 20, size // 2), (30, 30, 220), -1)
    cv2.putText(img, "AERA", (10, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2)
    return img


def _sim_frames(model_path: str, size: int) -> dict[str, np.ndarray]:
    """Render default + gripper frames from a domain-randomized env."""
    from aera.autonomous.envs.ar4_mk3_config import Ar4Mk3EnvConfig
    from aera.autonomous.envs.ar4_mk3_pick_and_place import Ar4Mk3PickAndPlaceEnv
    from aera_semi_autonomous.data.domain_rand_config_generator import (
        generate_random_domain_rand_config,
    )

    dr, _, _ = generate_random_domain_rand_config()
    cfg = Ar4Mk3EnvConfig(
        model_path=model_path, domain_rand=dr,
        image_width=size, image_height=size, include_images_in_obs=True,
    )
    env = Ar4Mk3PickAndPlaceEnv(render_mode="rgb_array", config=cfg)
    env.reset()
    frames = {
        "default": np.asarray(env.mujoco_renderer.render("rgb_array", camera_id=-1)),
        "gripper": np.asarray(
            env.mujoco_renderer.render("rgb_array", camera_name="gripper_camera")
        ),
    }
    env.close()
    return frames


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(out, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _build_grid(frame: np.ndarray, n_samples: int, n_profiles: int,
                strength: float, rng: np.random.Generator) -> np.ndarray:
    """One row per profile: [original | sample | sample | ...]."""
    rows = []
    for pi in range(n_profiles):
        profile = sample_camera_profile(rng, strength=strength)
        cells = [_label(frame, "original")]
        for si in range(n_samples):
            aug = augment_image(frame, profile, rng)
            cells.append(_label(aug, f"p{pi} s{si}"))
        rows.append(np.hstack(cells))
    return np.vstack(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-sim", action="store_true", help="render frames from the env")
    ap.add_argument("--image", type=str, default=None, help="load a frame from disk")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--n-profiles", type=int, default=3)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="/tmp/obs_aug_preview.png")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--model-path", type=str, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Gather base frame(s).
    if args.from_sim:
        model_path = args.model_path or os.path.abspath(
            "aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml"
        )
        frames = _sim_frames(model_path, args.size)
    elif args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"could not read image: {args.image}")
        frames = {"image": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)}
    else:
        logging.info("No source given; using a synthetic frame.")
        frames = {"synthetic": _synthetic_frame(args.size)}

    # Build a grid per camera, stacked vertically with a separator.
    blocks = []
    for name, frame in frames.items():
        grid = _build_grid(frame, args.n_samples, args.n_profiles, args.strength, rng)
        blocks.append(_label(grid, f"=== {name} ==="))
    full = np.vstack(blocks)

    cv2.imwrite(args.out, cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
    logging.info(f"Wrote preview grid: {args.out}  ({full.shape[1]}x{full.shape[0]})")

    # Sample + report a state-noise profile too.
    sp = sample_state_noise_profile(state_dim=7, rng=rng, strength=args.strength)
    clean = np.array([0.1, -0.4, 0.3, 0.0, 0.9, -0.2, -0.0115], dtype=np.float32)
    noisy = apply_state_noise(clean, sp, rng)
    logging.info("State noise (per-episode bias, per-frame jitter std):")
    logging.info(f"  bias       = {np.round(sp.bias, 5)}")
    logging.info(f"  jitter_std = {np.round(sp.jitter_std, 5)}")
    logging.info(f"  clean      = {np.round(clean, 5)}")
    logging.info(f"  noisy      = {np.round(noisy, 5)}")

    if args.show:
        cv2.imshow("obs augmentation preview", cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
