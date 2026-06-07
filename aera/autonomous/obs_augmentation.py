"""Sensor-realism augmentation for the policy's observations (images + state).

The MuJoCo renders are pristine and perfectly sharp; a real camera adds noise,
blur, vignetting, white-balance drift, compression artifacts, and the occasional
dropped/frozen frame. Real joint encoders add jitter and a slow per-boot bias.
This module makes the *sim observation distribution* look more camera/encoder-
like so the policy learns invariance to it.

Single shared implementation used by:
  - the offline dataset-prep stage (transform_skip_dataset / dataset_transforms)
    — bakes train-only augmentation into the training dataset;
  - eval (`run_policy_on_env`, behind a flag) — so sim-eval isn't pristine while
    training is noisy;
  - the preview tool (`preview_obs_augmentation.py`) — visual verification.

Design:
  - A *per-episode* ``CameraProfile`` fixes the persistent camera character
    (white balance, vignette, gamma, blur, base noise, jpeg quality, grayscale)
    so a clip is internally consistent; per-frame calls add the stochastic
    sensor noise / motion blur on top.
  - openpi already applies resampled RandomCrop / Rotate / ColorJitter at train
    time, so we deliberately do NOT duplicate those geometric/color-jitter ops
    here — this layer is the *sensor* gap only.
  - Temporal effects (dropped / frozen / duplicated frames) are sequence-level,
    so they live in the caller; ``CameraProfile.frame_drop_prob`` /
    ``frame_freeze_prob`` are provided for it to use.

All image ops are uint8 HWC RGB in, uint8 HWC RGB out.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np


# --- Image: per-episode camera profile -------------------------------------

@dataclasses.dataclass(frozen=True)
class CameraProfile:
    """Persistent per-episode camera characteristics. Sample once per episode
    with :func:`sample_camera_profile`, then pass to :func:`augment_image` for
    every frame of that episode."""

    # White-balance per-channel gains (multiplicative, ~1.0 = neutral).
    wb_gain: tuple = (1.0, 1.0, 1.0)
    # Tone.
    brightness: float = 1.0      # multiplicative
    contrast: float = 1.0        # around mid-gray
    gamma: float = 1.0           # >1 darker mids, <1 brighter
    hue_shift_deg: float = 0.0   # OpenCV hue is [0,180); shift in degrees/2
    saturation: float = 1.0
    # Optics / sensor.
    blur_sigma: float = 0.0      # gaussian defocus blur
    vignette: float = 0.0        # 0 = none, 1 = strong corner darkening
    noise_sigma: float = 0.0     # additive gaussian, in 0-255 units
    shot_noise: float = 0.0      # poisson-like, scales with intensity (0 = off)
    jpeg_quality: int = 100      # 100 = lossless-ish; lower = more artifacts
    grayscale: bool = False      # monochrome sensor / desaturated feed
    # Per-frame motion blur magnitude ceiling (pixels). The per-frame call
    # samples a random length/direction up to this (or uses a supplied
    # velocity).
    motion_blur_max: float = 0.0
    # Sequence-level (used by the caller, not augment_image).
    frame_drop_prob: float = 0.0
    frame_freeze_prob: float = 0.0


# Recommended default ranges (decision: "full set", recommended magnitudes).
# Conservative enough to stay realistic; wide enough to force invariance.
_DEFAULTS = {
    "wb_gain": (0.92, 1.08),       # per-channel
    "brightness": (0.8, 1.2),
    "contrast": (0.85, 1.2),
    "gamma": (0.8, 1.25),
    "hue_shift_deg": (-8.0, 8.0),
    "saturation": (0.8, 1.2),
    "blur_sigma": (0.0, 1.2),
    "vignette": (0.0, 0.45),
    "noise_sigma": (1.0, 6.0),     # 0-255 units
    "shot_noise": (0.0, 0.5),
    "jpeg_quality": (35, 95),
    "p_grayscale": 0.05,
    "motion_blur_max": (0.0, 6.0),
    "frame_drop_prob": (0.0, 0.03),
    "frame_freeze_prob": (0.0, 0.05),
}


def sample_camera_profile(
    rng: np.random.Generator | None = None, strength: float = 1.0
) -> CameraProfile:
    """Draw a per-episode camera profile. ``strength`` in [0,1] scales how far
    each parameter is pushed away from neutral (1.0 = full default ranges)."""
    r = rng or np.random.default_rng()

    def lerp(neutral, lo, hi):
        # Sample in [lo, hi] then pull toward `neutral` by (1 - strength).
        v = r.uniform(lo, hi)
        return neutral + strength * (v - neutral)

    d = _DEFAULTS
    wb = tuple(float(lerp(1.0, *d["wb_gain"])) for _ in range(3))
    return CameraProfile(
        wb_gain=wb,
        brightness=float(lerp(1.0, *d["brightness"])),
        contrast=float(lerp(1.0, *d["contrast"])),
        gamma=float(lerp(1.0, *d["gamma"])),
        hue_shift_deg=float(strength * r.uniform(*d["hue_shift_deg"])),
        saturation=float(lerp(1.0, *d["saturation"])),
        blur_sigma=float(strength * r.uniform(*d["blur_sigma"])),
        vignette=float(strength * r.uniform(*d["vignette"])),
        noise_sigma=float(strength * r.uniform(*d["noise_sigma"])),
        shot_noise=float(strength * r.uniform(*d["shot_noise"])),
        jpeg_quality=int(round(lerp(100, *d["jpeg_quality"]))),
        grayscale=bool(r.random() < strength * d["p_grayscale"]),
        motion_blur_max=float(strength * r.uniform(*d["motion_blur_max"])),
        frame_drop_prob=float(strength * r.uniform(*d["frame_drop_prob"])),
        frame_freeze_prob=float(strength * r.uniform(*d["frame_freeze_prob"])),
    )


# --- Image: per-frame application -------------------------------------------

def _apply_white_balance_tone(img: np.ndarray, p: CameraProfile) -> np.ndarray:
    out = img.astype(np.float32)
    out *= np.asarray(p.wb_gain, dtype=np.float32) * p.brightness
    # Contrast around mid-gray.
    out = (out - 127.5) * p.contrast + 127.5
    out = np.clip(out, 0, 255)
    # Gamma.
    if abs(p.gamma - 1.0) > 1e-3:
        out = 255.0 * np.power(out / 255.0, p.gamma)
    return out


def _apply_hue_sat(img: np.ndarray, p: CameraProfile) -> np.ndarray:
    if abs(p.hue_shift_deg) < 1e-3 and abs(p.saturation - 1.0) < 1e-3:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + p.hue_shift_deg / 2.0) % 180.0  # OpenCV hue 0-180
    hsv[..., 1] = np.clip(hsv[..., 1] * p.saturation, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _apply_vignette(img: np.ndarray, p: CameraProfile) -> np.ndarray:
    if p.vignette <= 1e-3:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = ((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2
    mask = (1.0 - p.vignette * np.clip(r2, 0, 1)).astype(np.float32)
    return img.astype(np.float32) * mask[..., None]


def _apply_motion_blur(
    img: np.ndarray, length: float, angle_deg: float
) -> np.ndarray:
    k = int(round(length))
    if k < 2:
        return img
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s = kernel.sum()
    if s <= 0:
        return img
    kernel /= s
    return cv2.filter2D(img, -1, kernel)


def augment_image(
    img: np.ndarray,
    profile: CameraProfile,
    rng: np.random.Generator | None = None,
    motion_vec: tuple | None = None,
) -> np.ndarray:
    """Apply the sensor-realism pipeline to one uint8 HWC RGB frame.

    ``motion_vec`` optionally supplies a (dx, dy) image-plane motion (pixels)
    to make motion blur track real motion; otherwise a random blur up to
    ``profile.motion_blur_max`` is used. Per-frame stochastic parts
    (noise realization, motion-blur sample) use ``rng``.
    """
    r = rng or np.random.default_rng()
    out = img

    # Persistent camera tone / color (per-episode constants).
    out = _apply_white_balance_tone(out, profile).astype(np.uint8)
    out = _apply_hue_sat(out, profile)
    if profile.grayscale:
        g = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

    # Optics: defocus + motion blur.
    if profile.blur_sigma > 1e-3:
        out = cv2.GaussianBlur(out, (0, 0), profile.blur_sigma)
    if motion_vec is not None:
        dx, dy = motion_vec
        length = float(np.hypot(dx, dy))
        if length >= 2.0:
            out = _apply_motion_blur(out, length, np.degrees(np.arctan2(dy, dx)))
    elif profile.motion_blur_max >= 2.0:
        length = float(r.uniform(0, profile.motion_blur_max))
        out = _apply_motion_blur(out, length, float(r.uniform(0, 180)))

    out = _apply_vignette(out, profile).astype(np.float32)

    # Sensor noise: shot (intensity-dependent) + additive gaussian.
    if profile.shot_noise > 1e-3:
        out = out + r.normal(0.0, 1.0, out.shape).astype(np.float32) * (
            profile.shot_noise * np.sqrt(np.clip(out, 0, None))
        )
    if profile.noise_sigma > 1e-3:
        out = out + r.normal(0.0, profile.noise_sigma, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Compression artifacts.
    if profile.jpeg_quality < 100:
        ok, enc = cv2.imencode(
            ".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(profile.jpeg_quality)],
        )
        if ok:
            out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    return out


# --- State / proprioception noise ------------------------------------------

@dataclasses.dataclass(frozen=True)
class StateNoiseProfile:
    """Per-episode proprioception error: a constant ``bias`` (miscalibration /
    per-boot offset) plus the std of per-frame Gaussian ``jitter``. Both are
    length-`state_dim` arrays. Applied to the *state input only* — never the
    action target."""

    bias: np.ndarray
    jitter_std: np.ndarray


# Defaults: ~0.7 deg jitter on the 6 arm joints, ~0.2 mm on the gripper; the
# per-episode bias is ~half the jitter scale. Index 6 is the gripper jaw (m).
_ARM_JITTER_RAD = np.deg2rad(0.7)
_GRIPPER_JITTER_M = 0.0002
_BIAS_FRACTION = 0.5


def sample_state_noise_profile(
    state_dim: int = 7,
    rng: np.random.Generator | None = None,
    strength: float = 1.0,
) -> StateNoiseProfile:
    r = rng or np.random.default_rng()
    jitter = np.full(state_dim, _ARM_JITTER_RAD, dtype=np.float32)
    if state_dim >= 1:
        jitter[-1] = _GRIPPER_JITTER_M  # gripper dim is meters
    jitter *= strength
    bias = r.normal(0.0, jitter * _BIAS_FRACTION).astype(np.float32)
    return StateNoiseProfile(bias=bias, jitter_std=jitter)


def apply_state_noise(
    state: np.ndarray,
    profile: StateNoiseProfile,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return ``state`` + per-episode bias + per-frame Gaussian jitter."""
    r = rng or np.random.default_rng()
    state = np.asarray(state, dtype=np.float32)
    noise = r.normal(0.0, 1.0, state.shape).astype(np.float32) * profile.jitter_std
    return state + profile.bias + noise
