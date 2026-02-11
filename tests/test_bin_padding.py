from __future__ import annotations

import numpy as np
import pytest

from wcv import EstimationOptions, GridSpec, estimate_single_seed_velocity, estimate_velocity_map


def _make_inputs(shape: tuple[int, int] = (396, 640), nt: int = 24) -> tuple[np.ndarray, GridSpec]:
    ny, nx = shape
    rng = np.random.default_rng(0)
    movie = rng.standard_normal((nt, ny, nx), dtype=np.float32)
    grid = GridSpec(patch_px=4, grid_stride_patches=4)  # bin_px=16
    return movie, grid


def test_single_seed_strict_mode_raises_value_error_for_non_divisible_shape() -> None:
    movie, grid = _make_inputs()

    with pytest.raises(ValueError, match=r"superbin bin_px=16 must tile image dimensions \(396,640\) exactly"):
        estimate_single_seed_velocity(
            movie=movie,
            fs=1000.0,
            grid=grid,
            seed_box_px=(16, 32, 16, 32),
            bg_boxes_px=[(32, 80, 32, 80)],
            extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
            dj_mm=1.0,
            shifts=(1,),
            options=EstimationOptions(min_used=1),
            allow_bin_padding=False,
        )


def test_single_seed_padded_mode_runs_and_warns() -> None:
    movie, grid = _make_inputs()

    with pytest.warns(UserWarning, match="edge padding can affect"):
        result = estimate_single_seed_velocity(
            movie=movie,
            fs=1000.0,
            grid=grid,
            seed_box_px=(16, 32, 16, 32),
            bg_boxes_px=[(32, 80, 32, 80)],
            extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
            dj_mm=1.0,
            shifts=(1,),
            options=EstimationOptions(min_used=1),
            allow_bin_padding=True,
        )

    assert result.by == 25
    assert result.bx == 40


def test_velocity_map_strict_mode_raises_value_error_for_non_divisible_shape() -> None:
    movie, grid = _make_inputs()

    with pytest.raises(ValueError, match=r"superbin bin_px=16 must tile image dimensions \(396,640\) exactly"):
        estimate_velocity_map(
            movie=movie,
            fs=1000.0,
            grid=grid,
            bg_boxes_px=[(32, 80, 32, 80)],
            extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
            dj_mm=1.0,
            shifts=(1,),
            options=EstimationOptions(min_used=1),
            allow_bin_padding=False,
            use_shear_mask=False,
        )


def test_velocity_map_padded_mode_runs_and_warns() -> None:
    movie, grid = _make_inputs()

    with pytest.warns(UserWarning, match="edge padding can affect"):
        result = estimate_velocity_map(
            movie=movie,
            fs=1000.0,
            grid=grid,
            bg_boxes_px=[(32, 80, 32, 80)],
            extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
            dj_mm=1.0,
            shifts=(1,),
            options=EstimationOptions(min_used=1),
            allow_bin_padding=True,
            use_shear_mask=False,
        )

    assert result.ux_map.shape == (25, 40)
    assert result.uy_map.shape == (25, 40)
