"""Minimal tests for the crucial fixture-free helpers in transform_skip_dataset."""

import numpy as np

from aera.autonomous.openpi.scripts.transform_skip_dataset import (
    _build_output_repo_id,
    _parse_image_from_sample,
)


def test_build_output_repo_id_naming():
    # A wrong name silently writes/uploads the wrong dataset, so pin the scheme.
    assert _build_output_repo_id("org/name", 5, False, None) == "org/name_skip5"
    assert _build_output_repo_id("org/name", 5, True, None) == "org/name_skip5_delta"
    assert _build_output_repo_id("org/name", 3, True, "custom") == "org/name_custom"


def test_parse_image_float_chw_to_uint8_hwc():
    # LeRobot stores float32 CHW; the pipeline (and obs-aug) needs uint8 HWC.
    chw = np.zeros((3, 2, 2), dtype=np.float32)
    chw[0] = 1.0  # full red channel
    out = _parse_image_from_sample(chw)
    assert out.dtype == np.uint8
    assert out.shape == (2, 2, 3)
    assert (out[..., 0] == 255).all()
    assert (out[..., 1] == 0).all() and (out[..., 2] == 0).all()
