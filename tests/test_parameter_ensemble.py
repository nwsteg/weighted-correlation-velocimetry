from __future__ import annotations

import numpy as np

from wcv.parameter_ensemble import run_parameter_ensemble_uncertainty, sample_parameter_ensemble
from wcv.types import EstimationOptions, VelocityMapResult


def _stub_estimator(movie: np.ndarray, options: EstimationOptions) -> VelocityMapResult:
    by, bx = 2, 3
    base = options.rmin + 0.01 * options.min_used + 0.1 * options.weight_power
    ux = np.full((by, bx), base, dtype=np.float32)
    uy = np.full((by, bx), -base, dtype=np.float32)
    ux[0, 0] = np.nan
    uy[0, 0] = np.nan
    return VelocityMapResult(
        ux_map=ux,
        uy_map=uy,
        used_count_map=np.ones((by, bx), dtype=np.int32),
        shear_mask_vec=np.ones(by * bx, dtype=bool),
        x_m=np.arange(by * bx, dtype=np.float32),
        y_m=np.arange(by * bx, dtype=np.float32),
        by=by,
        bx=bx,
        corr_by_shift={},
        valid_seed_count=5,
        total_seed_count=6,
    )


def test_parameter_sampling_is_reproducible() -> None:
    s1 = sample_parameter_ensemble(32, seed=9, edge_clip_none_fraction=0.2)
    s2 = sample_parameter_ensemble(32, seed=9, edge_clip_none_fraction=0.2)

    np.testing.assert_allclose(s1["rmin"], s2["rmin"])
    np.testing.assert_array_equal(s1["min_used"], s2["min_used"])
    np.testing.assert_allclose(s1["weight_power"], s2["weight_power"])
    np.testing.assert_array_equal(s1["edge_clip_is_none"], s2["edge_clip_is_none"])


def test_parameter_sampling_custom_bounds_are_applied() -> None:
    s = sample_parameter_ensemble(
        48,
        seed=4,
        edge_clip_none_fraction=0.0,
        rmin_bounds=(0.25, 0.3),
        min_used_bounds=(12, 13),
        weight_power_bounds=(1.5, 1.7),
        edge_clip_bounds=(0.8, 0.9),
    )

    assert np.all((s["rmin"] >= 0.25) & (s["rmin"] <= 0.3))
    assert np.all((s["min_used"] >= 12) & (s["min_used"] <= 13))
    assert np.all((s["weight_power"] >= 1.5) & (s["weight_power"] <= 1.7))
    assert np.all((s["edge_clip_reject_k"] >= 0.8) & (s["edge_clip_reject_k"] <= 0.9))


def test_ensemble_shapes_nan_handling_and_stats_ordering() -> None:
    movie = np.zeros((5, 8, 8), dtype=np.float32)
    out = run_parameter_ensemble_uncertainty(
        movie,
        estimator_kwargs={},
        baseline_options=EstimationOptions(),
        estimator=_stub_estimator,
        n_param=20,
        seed=11,
        min_valid_fraction=0.6,
    )

    assert out.ux.mean.shape == (2, 3)
    assert out.coverage_map.shape == (2, 3)
    assert np.isnan(out.ux.mean[0, 0])
    assert np.isnan(out.uy.std[0, 0])

    finite = np.isfinite(out.ux.p2_5) & np.isfinite(out.ux.p97_5)
    assert np.all(out.ux.p2_5[finite] <= out.ux.p97_5[finite])
    finite_std = np.isfinite(out.ux.std)
    assert np.all(out.ux.std[finite_std] >= 0.0)


def test_ensemble_progress_callback_receives_all_updates() -> None:
    movie = np.zeros((5, 8, 8), dtype=np.float32)
    updates: list[tuple[int, int]] = []

    _ = run_parameter_ensemble_uncertainty(
        movie,
        estimator_kwargs={},
        baseline_options=EstimationOptions(),
        estimator=_stub_estimator,
        n_param=7,
        seed=3,
        progress_callback=lambda done, total: updates.append((done, total)),
    )

    assert updates == [(1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7)]


def test_ensemble_run_accepts_custom_bounds() -> None:
    movie = np.zeros((5, 8, 8), dtype=np.float32)
    out = run_parameter_ensemble_uncertainty(
        movie,
        estimator_kwargs={},
        baseline_options=EstimationOptions(),
        estimator=_stub_estimator,
        n_param=16,
        seed=8,
        rmin_bounds=(0.31, 0.32),
        min_used_bounds=(14, 18),
        weight_power_bounds=(2.4, 2.5),
        edge_clip_bounds=(1.4, 1.5),
        edge_clip_none_fraction=0.0,
    )

    assert np.all((out.sampled_rmin >= 0.31) & (out.sampled_rmin <= 0.32))
    assert np.all((out.sampled_min_used >= 14) & (out.sampled_min_used <= 18))
    assert np.all((out.sampled_weight_power >= 2.4) & (out.sampled_weight_power <= 2.5))
    assert np.all((out.sampled_edge_clip_reject_k >= 1.4) & (out.sampled_edge_clip_reject_k <= 1.5))
