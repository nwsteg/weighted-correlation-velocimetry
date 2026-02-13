from __future__ import annotations

import numpy as np

from wcv import (
    EstimationOptions,
    GridSpec,
    bootstrap_velocity_map,
    default_block_length,
    moving_block_bootstrap_indices,
)
from wcv.types import VelocityMapResult


def _stub_estimator(movie: np.ndarray, nan_every: int = 2) -> VelocityMapResult:
    n = movie.shape[0]
    ux = np.array([[float(n), np.nan if n % nan_every == 0 else 1.0]], dtype=np.float32)
    uy = np.array([[2.0, np.nan if n % nan_every == 0 else 3.0]], dtype=np.float32)
    return VelocityMapResult(
        ux_map=ux,
        uy_map=uy,
        used_count_map=np.ones_like(ux, dtype=np.int32),
        shear_mask_vec=np.ones(ux.size, dtype=bool),
        x_m=np.array([0.0, 1.0], dtype=np.float32),
        y_m=np.array([0.0, 0.0], dtype=np.float32),
        by=1,
        bx=2,
        corr_by_shift={},
        valid_seed_count=1,
        total_seed_count=2,
    )


def test_moving_block_bootstrap_indices_shape_and_range() -> None:
    rng = np.random.default_rng(7)
    idx = moving_block_bootstrap_indices(100, 10, rng, circular=False)
    assert idx.shape == (100,)
    assert np.all((idx >= 0) & (idx < 100))


def test_bootstrap_is_repeatable_for_fixed_seed() -> None:
    movie = np.random.default_rng(0).standard_normal((100, 4, 4), dtype=np.float32)

    r1 = bootstrap_velocity_map(movie, n_bootstrap=8, estimator=_stub_estimator, seed=123)
    r2 = bootstrap_velocity_map(movie, n_bootstrap=8, estimator=_stub_estimator, seed=123)

    np.testing.assert_allclose(r1.ux.mean, r2.ux.mean, equal_nan=True)
    np.testing.assert_allclose(r1.uy.ci_low, r2.uy.ci_low, equal_nan=True)
    np.testing.assert_allclose(r1.um.std, r2.um.std, equal_nan=True)


def test_bootstrap_handles_low_n_and_min_valid_threshold() -> None:
    movie = np.random.default_rng(3).standard_normal((100, 4, 4), dtype=np.float32)

    result = bootstrap_velocity_map(
        movie,
        n_bootstrap=10,
        estimator=_stub_estimator,
        seed=8,
        min_valid_fraction=0.8,
        estimator_kwargs={"nan_every": 1},
    )

    assert result.block_length == default_block_length(100)
    assert result.sample_shape == (1, 2)
    assert np.isnan(result.ux.mean[0, 1])
    assert result.ux.valid_fraction[0, 1] == 0.0


def test_bootstrap_progress_callback_receives_all_replicates() -> None:
    movie = np.random.default_rng(11).standard_normal((50, 4, 4), dtype=np.float32)
    updates: list[tuple[int, int]] = []

    _ = bootstrap_velocity_map(
        movie,
        n_bootstrap=5,
        estimator=_stub_estimator,
        seed=42,
        progress_callback=lambda done, total: updates.append((done, total)),
    )

    assert updates == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]


def test_bootstrap_integration_with_velocity_estimator_shift1() -> None:
    rng = np.random.default_rng(12)
    movie = rng.standard_normal((100, 16, 16), dtype=np.float32)

    out = bootstrap_velocity_map(
        movie,
        n_bootstrap=3,
        seed=21,
        estimator_kwargs={
            "fs": 1000.0,
            "grid": GridSpec(patch_px=8, grid_stride_patches=1),
            "bg_boxes_px": [(0, 8, 0, 8)],
            "extent_xd_yd": (0.0, 1.0, 0.0, 1.0),
            "dj_mm": 1.0,
            "shifts": (1,),
            "options": EstimationOptions(min_used=1, rmin=0.0),
            "allow_bin_padding": False,
            "use_shear_mask": False,
        },
    )

    assert out.ux.mean.shape == (2, 2)
    assert out.uy.ci_high.shape == (2, 2)
    assert np.all(out.ux.valid_fraction >= 0.0)
    assert np.all(out.ux.valid_fraction <= 1.0)
