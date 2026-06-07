"""Minimal tests for the crucial obs-augmentation invariants."""

import numpy as np

from aera.autonomous.obs_augmentation import (
    StateNoiseProfile,
    apply_state_noise,
    augment_image,
    sample_camera_profile,
    sample_state_noise_profile,
)


def _img() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)


def test_augment_image_preserves_uint8_hwc_contract():
    # Downstream (dataset writer, model) relies on uint8 HWC in [0,255].
    rng = np.random.default_rng(0)
    img = _img()
    out = augment_image(img, sample_camera_profile(rng, strength=1.0), rng)
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert out.min() >= 0 and out.max() <= 255


def test_augment_image_strength_zero_is_identity():
    rng = np.random.default_rng(0)
    img = _img()
    out = augment_image(img, sample_camera_profile(rng, strength=0.0), rng)
    assert np.array_equal(out, img)


def test_apply_state_noise_zero_profile_is_identity_and_keeps_shape():
    state = np.array([0.1, -0.4, 0.3, 0.0, 0.9, -0.2, -0.0115], dtype=np.float32)
    prof = StateNoiseProfile(
        bias=np.zeros(7, np.float32), jitter_std=np.zeros(7, np.float32)
    )
    out = apply_state_noise(state, prof, np.random.default_rng(0))
    assert out.shape == state.shape
    assert np.allclose(out, state)


def test_apply_state_noise_adds_constant_bias_and_varying_jitter():
    state = np.zeros(7, dtype=np.float32)
    prof = StateNoiseProfile(
        bias=np.full(7, 0.05, np.float32), jitter_std=np.full(7, 0.01, np.float32)
    )
    rng = np.random.default_rng(0)
    a = apply_state_noise(state, prof, rng)
    b = apply_state_noise(state, prof, rng)
    # Centered on the (constant) bias, but jitter makes successive draws differ.
    assert np.all(np.abs(a - 0.05) < 0.06)
    assert not np.allclose(a, b)


def test_state_profile_gripper_jitter_smaller_than_arm():
    # The gripper dim is meters; arm dims are radians — must not share scale.
    prof = sample_state_noise_profile(7, np.random.default_rng(0), strength=1.0)
    assert prof.jitter_std[6] < prof.jitter_std[0]
