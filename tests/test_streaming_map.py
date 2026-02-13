from __future__ import annotations

import numpy as np

from wcv import (
    EstimationOptions,
    GridSpec,
    estimate_velocity_map,
    estimate_velocity_map_hybrid,
    estimate_velocity_map_streaming,
)
from wcv.correlation import (
    corr_matrix_positive_shift,
    corr_targets_for_seed_block_positive_shift,
    corr_targets_for_seed_chunk_positive_shift,
    corr_targets_for_seed_positive_shift,
    sparse_useful_corr_row,
)


def test_seedwise_and_chunkwise_positive_shift_correlation_match_full_matrix() -> None:
    rng = np.random.default_rng(11)
    region_res = rng.standard_normal((7, 20), dtype=np.float32)
    shift = 2

    full = corr_matrix_positive_shift(region_res, shift)

    for seed in range(region_res.shape[0]):
        np.testing.assert_allclose(
            corr_targets_for_seed_positive_shift(region_res, seed, shift),
            full[:, seed],
            rtol=1e-6,
            atol=1e-6,
        )

    seed_chunk = np.array([1, 4, 6], dtype=np.int64)
    np.testing.assert_allclose(
        corr_targets_for_seed_chunk_positive_shift(region_res, seed_chunk, shift),
        full[:, seed_chunk],
        rtol=1e-6,
        atol=1e-6,
    )


def test_seedblock_positive_shift_correlation_matches_full_matrix() -> None:
    rng = np.random.default_rng(44)
    region_res = rng.standard_normal((8, 24), dtype=np.float32)
    shift = 3
    seed_block = np.array([0, 3, 5, 7], dtype=np.int64)

    full = corr_matrix_positive_shift(region_res, shift)
    block = corr_targets_for_seed_block_positive_shift(region_res, seed_block, shift)

    assert block.shape == (seed_block.size, region_res.shape[0])
    np.testing.assert_allclose(block, full[:, seed_block].T, rtol=1e-6, atol=1e-6)


def test_seedwise_positive_shift_correlation_matches_with_precomputed_stats() -> None:
    rng = np.random.default_rng(31)
    region_res = rng.standard_normal((9, 22), dtype=np.float32)
    shift = 2

    target_slice = region_res[:, shift:].astype(np.float64, copy=False)
    seed_slice = region_res[:, :-shift].astype(np.float64, copy=False)
    target_row_means = target_slice.mean(axis=1)
    target_row_stds = target_slice.std(axis=1, ddof=1) + 1e-12
    seed_row_means = seed_slice.mean(axis=1)
    seed_row_stds = seed_slice.std(axis=1, ddof=1) + 1e-12
    norm_denom = float(target_slice.shape[1] - 1)

    seed = 3
    baseline = corr_targets_for_seed_positive_shift(region_res, seed, shift)
    precomputed = corr_targets_for_seed_positive_shift(
        region_res,
        seed,
        shift,
        target_row_means=target_row_means,
        target_row_stds=target_row_stds,
        seed_row_means=seed_row_means,
        seed_row_stds=seed_row_stds,
        norm_denom=norm_denom,
    )

    np.testing.assert_allclose(precomputed, baseline, rtol=1e-6, atol=1e-6)


def test_velocity_map_streaming_matches_materialized_and_skips_corr_storage_by_default() -> None:
    rng = np.random.default_rng(12)
    movie = rng.standard_normal((24, 32, 32), dtype=np.float32)
    grid = GridSpec(patch_px=8, grid_stride_patches=1)
    options = EstimationOptions(min_used=1, rmin=0.0)

    common_kwargs = dict(
        movie=movie,
        fs=1000.0,
        grid=grid,
        bg_boxes_px=[(0, 8, 0, 8)],
        extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
        dj_mm=1.0,
        shifts=(1, 2),
        options=options,
        allow_bin_padding=False,
        use_shear_mask=False,
    )

    materialized = estimate_velocity_map(**common_kwargs)
    streaming = estimate_velocity_map_streaming(**common_kwargs)
    streaming_with_corr = estimate_velocity_map_streaming(
        **common_kwargs,
        store_corr_by_shift=True,
    )

    np.testing.assert_allclose(streaming.ux_map, materialized.ux_map, equal_nan=True)
    np.testing.assert_allclose(streaming.uy_map, materialized.uy_map, equal_nan=True)
    np.testing.assert_array_equal(streaming.used_count_map, materialized.used_count_map)
    assert streaming.valid_seed_count == materialized.valid_seed_count
    assert streaming.total_seed_count == materialized.total_seed_count

    assert streaming.corr_by_shift == {}
    assert set(streaming_with_corr.corr_by_shift) == {1, 2}
    np.testing.assert_allclose(
        streaming_with_corr.corr_by_shift[1],
        materialized.corr_by_shift[1],
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )






def test_velocity_map_seed_mask_controls_seed_selection_independently() -> None:
    rng = np.random.default_rng(90)
    movie = rng.standard_normal((24, 32, 32), dtype=np.float32)
    grid = GridSpec(patch_px=8, grid_stride_patches=1)
    options = EstimationOptions(min_used=1, rmin=0.0)

    # 4x4 patch grid for a 32x32 frame with bin_px=8
    seed_mask_px = np.zeros((32, 32), dtype=bool)
    seed_mask_px[8:16, 8:16] = True
    seed_mask_px[16:24, 8:16] = True

    shear_mask_px = np.zeros((32, 32), dtype=bool)
    shear_mask_px[8:24, 8:16] = True
    shear_mask_px[8:24, 16:24] = True

    common_kwargs = dict(
        movie=movie,
        fs=1000.0,
        grid=grid,
        bg_boxes_px=[(0, 8, 0, 8)],
        extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
        dj_mm=1.0,
        shifts=(1, 2),
        options=options,
        allow_bin_padding=False,
        use_shear_mask=True,
        shear_mask_px=shear_mask_px,
        seed_mask_px=seed_mask_px,
    )

    materialized = estimate_velocity_map(**common_kwargs)
    streaming = estimate_velocity_map_streaming(**common_kwargs)
    hybrid = estimate_velocity_map_hybrid(**common_kwargs, seed_chunk_size=2)

    np.testing.assert_allclose(streaming.ux_map, materialized.ux_map, equal_nan=True)
    np.testing.assert_allclose(streaming.uy_map, materialized.uy_map, equal_nan=True)
    np.testing.assert_array_equal(streaming.used_count_map, materialized.used_count_map)
    np.testing.assert_allclose(hybrid.ux_map, materialized.ux_map, equal_nan=True)
    np.testing.assert_allclose(hybrid.uy_map, materialized.uy_map, equal_nan=True)

    assert materialized.total_seed_count == 2
    assert streaming.total_seed_count == 2
    assert hybrid.total_seed_count == 2

    non_seed_indices = np.setdiff1d(np.arange(materialized.ux_map.size), np.array([5, 9]))
    assert np.all(np.isnan(materialized.ux_map.ravel()[non_seed_indices]))
    assert np.all(streaming.used_count_map.ravel()[non_seed_indices] == 0)

def test_velocity_map_streaming_matches_materialized_with_gating_and_weight_options() -> None:
    rng = np.random.default_rng(21)
    movie = rng.standard_normal((30, 40, 40), dtype=np.float32)
    grid = GridSpec(patch_px=8, grid_stride_patches=1)
    options = EstimationOptions(
        min_used=2,
        rmin=0.1,
        require_downstream=True,
        require_dy_positive=True,
        weight_power=1.7,
    )

    common_kwargs = dict(
        movie=movie,
        fs=1200.0,
        grid=grid,
        bg_boxes_px=[(0, 8, 0, 8)],
        extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
        dj_mm=1.0,
        shifts=(1, 2, 3),
        options=options,
        allow_bin_padding=False,
        use_shear_mask=False,
    )

    materialized = estimate_velocity_map(**common_kwargs)
    streaming = estimate_velocity_map_streaming(**common_kwargs)

    np.testing.assert_allclose(streaming.ux_map, materialized.ux_map, equal_nan=True)
    np.testing.assert_allclose(streaming.uy_map, materialized.uy_map, equal_nan=True)
    np.testing.assert_array_equal(streaming.used_count_map, materialized.used_count_map)
    assert streaming.valid_seed_count == materialized.valid_seed_count
    assert streaming.total_seed_count == materialized.total_seed_count


def test_velocity_map_streaming_seed_block_matches_single_seed_path() -> None:
    rng = np.random.default_rng(77)
    movie = rng.standard_normal((30, 40, 40), dtype=np.float32)
    grid = GridSpec(patch_px=8, grid_stride_patches=1)
    options = EstimationOptions(
        min_used=2,
        rmin=0.1,
        require_downstream=True,
        require_dy_positive=True,
        weight_power=1.7,
    )

    common_kwargs = dict(
        movie=movie,
        fs=1200.0,
        grid=grid,
        bg_boxes_px=[(0, 8, 0, 8)],
        extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
        dj_mm=1.0,
        shifts=(1, 2, 3),
        options=options,
        allow_bin_padding=False,
        use_shear_mask=False,
    )

    streaming_single = estimate_velocity_map_streaming(**common_kwargs, seed_chunk_size=1)
    streaming_block = estimate_velocity_map_streaming(**common_kwargs, seed_chunk_size=4)

    np.testing.assert_allclose(streaming_block.ux_map, streaming_single.ux_map, equal_nan=True)
    np.testing.assert_allclose(streaming_block.uy_map, streaming_single.uy_map, equal_nan=True)
    np.testing.assert_array_equal(streaming_block.used_count_map, streaming_single.used_count_map)
    assert streaming_block.valid_seed_count == streaming_single.valid_seed_count
    assert streaming_block.total_seed_count == streaming_single.total_seed_count


def test_velocity_map_streaming_single_shift_single_bin_matches_materialized() -> None:
    rng = np.random.default_rng(32)
    movie = rng.standard_normal((20, 24, 24), dtype=np.float32)
    grid = GridSpec(patch_px=8, grid_stride_patches=1)
    options = EstimationOptions(min_used=1, rmin=0.0)

    common_kwargs = dict(
        movie=movie,
        fs=800.0,
        grid=grid,
        bg_boxes_px=[(0, 8, 0, 8)],
        extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
        dj_mm=1.0,
        shifts=(1,),
        options=options,
        allow_bin_padding=False,
        use_shear_mask=False,
    )

    materialized = estimate_velocity_map(**common_kwargs)
    streaming = estimate_velocity_map_streaming(**common_kwargs)

    np.testing.assert_allclose(streaming.ux_map, materialized.ux_map, rtol=1e-6, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(streaming.uy_map, materialized.uy_map, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_velocity_map_progress_callback_reports_all_stages() -> None:
    rng = np.random.default_rng(13)
    movie = rng.standard_normal((18, 24, 24), dtype=np.float32)
    grid = GridSpec(patch_px=8, grid_stride_patches=1)

    events: list[tuple[int, int, str]] = []

    estimate_velocity_map(
        movie=movie,
        fs=1000.0,
        grid=grid,
        bg_boxes_px=[(0, 8, 0, 8)],
        extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
        dj_mm=1.0,
        shifts=(1, 2, 3),
        options=EstimationOptions(min_used=1, rmin=0.0),
        allow_bin_padding=False,
        use_shear_mask=False,
        on_progress=lambda done, total, stage: events.append((done, total, stage)),
    )

    assert any(stage == "preprocessing" and done == total == 4 for done, total, stage in events)
    assert any(stage == "per-shift processing" and done == total == 3 for done, total, stage in events)
    assert any(stage == "seed loop progress" and done == total == 9 for done, total, stage in events)


def test_streaming_velocity_map_progress_factory_per_stage_callback() -> None:
    rng = np.random.default_rng(14)
    movie = rng.standard_normal((18, 24, 24), dtype=np.float32)
    grid = GridSpec(patch_px=8, grid_stride_patches=1)

    stage_done: dict[str, tuple[int, int]] = {}

    def factory(total: int, stage: str):
        def callback(done: int, total_arg: int, stage_arg: str) -> None:
            stage_done[stage_arg] = (done, total_arg)

        assert total > 0
        assert stage in {"preprocessing", "seed loop progress"}
        return callback

    estimate_velocity_map_streaming(
        movie=movie,
        fs=1000.0,
        grid=grid,
        bg_boxes_px=[(0, 8, 0, 8)],
        extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
        dj_mm=1.0,
        shifts=(1, 2, 3),
        options=EstimationOptions(min_used=1, rmin=0.0),
        allow_bin_padding=False,
        use_shear_mask=False,
        progress_factory=factory,
    )

    assert stage_done["preprocessing"] == (4, 4)
    assert stage_done["seed loop progress"] == (9, 9)


def test_sparse_useful_corr_row_applies_threshold_and_top_k() -> None:
    corr = np.array([0.1, 0.6, np.nan, 0.8, 0.4, -0.2], dtype=np.float32)
    gate = np.array([True, True, True, True, True, False])

    idx_all, vals_all = sparse_useful_corr_row(corr, candidate_mask=gate, rmin=0.3)
    np.testing.assert_array_equal(idx_all, np.array([1, 3, 4], dtype=np.int32))
    np.testing.assert_allclose(vals_all, np.array([0.6, 0.8, 0.4], dtype=np.float32))

    idx_top2, vals_top2 = sparse_useful_corr_row(corr, candidate_mask=gate, rmin=0.3, top_k=2)
    np.testing.assert_array_equal(idx_top2, np.array([1, 3], dtype=np.int32))
    np.testing.assert_allclose(vals_top2, np.array([0.6, 0.8], dtype=np.float32))


def test_velocity_map_sparse_corr_storage_matches_dense_fit() -> None:
    rng = np.random.default_rng(123)
    movie = rng.standard_normal((24, 32, 32), dtype=np.float32)
    grid = GridSpec(patch_px=8, grid_stride_patches=1)
    options = EstimationOptions(min_used=1, rmin=0.05, require_downstream=True, weight_power=1.3)

    common_kwargs = dict(
        movie=movie,
        fs=1000.0,
        grid=grid,
        bg_boxes_px=[(0, 8, 0, 8)],
        extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
        dj_mm=1.0,
        shifts=(1, 2),
        options=options,
        allow_bin_padding=False,
        use_shear_mask=False,
    )

    dense = estimate_velocity_map(**common_kwargs)
    sparse = estimate_velocity_map(**common_kwargs, sparse_corr_storage=True)

    np.testing.assert_allclose(sparse.ux_map, dense.ux_map, equal_nan=True)
    np.testing.assert_allclose(sparse.uy_map, dense.uy_map, equal_nan=True)
    np.testing.assert_array_equal(sparse.used_count_map, dense.used_count_map)

    seed_count = dense.by * dense.bx
    for shift in (1, 2):
        assert isinstance(sparse.corr_by_shift[shift], dict)
        assert len(sparse.corr_by_shift[shift]) == seed_count


def test_velocity_map_sparse_corr_storage_top_k_caps_neighbors() -> None:
    rng = np.random.default_rng(456)
    movie = rng.standard_normal((24, 32, 32), dtype=np.float32)
    grid = GridSpec(patch_px=8, grid_stride_patches=1)
    options = EstimationOptions(min_used=1, rmin=0.0, require_downstream=False)

    sparse = estimate_velocity_map(
        movie=movie,
        fs=1000.0,
        grid=grid,
        bg_boxes_px=[(0, 8, 0, 8)],
        extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
        dj_mm=1.0,
        shifts=(1,),
        options=options,
        allow_bin_padding=False,
        use_shear_mask=False,
        sparse_corr_storage=True,
        sparse_top_k=2,
    )

    sparse_rows = sparse.corr_by_shift[1]
    assert isinstance(sparse_rows, dict)
    for idx, vals in sparse_rows.values():
        assert idx.size <= 2
        assert vals.size <= 2


def test_velocity_map_hybrid_matches_materialized_with_buffer_controls() -> None:
    rng = np.random.default_rng(92)
    movie = rng.standard_normal((28, 32, 32), dtype=np.float32)
    grid = GridSpec(patch_px=8, grid_stride_patches=1)
    options = EstimationOptions(min_used=1, rmin=0.0, require_downstream=True, weight_power=1.5)

    common_kwargs = dict(
        movie=movie,
        fs=1000.0,
        grid=grid,
        bg_boxes_px=[(0, 8, 0, 8)],
        extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
        dj_mm=1.0,
        shifts=(1, 2, 3),
        options=options,
        allow_bin_padding=False,
        use_shear_mask=False,
    )

    materialized = estimate_velocity_map(**common_kwargs)
    hybrid = estimate_velocity_map_hybrid(
        **common_kwargs,
        seed_chunk_size=4,
        shift_chunk_size=2,
        max_corr_buffer_mb=0.001,
    )

    np.testing.assert_allclose(hybrid.ux_map, materialized.ux_map, equal_nan=True)
    np.testing.assert_allclose(hybrid.uy_map, materialized.uy_map, equal_nan=True)
    np.testing.assert_array_equal(hybrid.used_count_map, materialized.used_count_map)


def test_velocity_map_streaming_shift_chunked_matches_unchunked() -> None:
    rng = np.random.default_rng(93)
    movie = rng.standard_normal((26, 32, 32), dtype=np.float32)
    grid = GridSpec(patch_px=8, grid_stride_patches=1)
    options = EstimationOptions(min_used=1, rmin=0.0)

    common_kwargs = dict(
        movie=movie,
        fs=1000.0,
        grid=grid,
        bg_boxes_px=[(0, 8, 0, 8)],
        extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
        dj_mm=1.0,
        shifts=(1, 2, 3),
        options=options,
        allow_bin_padding=False,
        use_shear_mask=False,
    )

    unchunked = estimate_velocity_map_streaming(**common_kwargs, seed_chunk_size=5, shift_chunk_size=None)
    chunked = estimate_velocity_map_streaming(
        **common_kwargs,
        seed_chunk_size=5,
        shift_chunk_size=1,
        max_corr_buffer_mb=0.01,
    )

    np.testing.assert_allclose(chunked.ux_map, unchunked.ux_map, equal_nan=True)
    np.testing.assert_allclose(chunked.uy_map, unchunked.uy_map, equal_nan=True)
    np.testing.assert_array_equal(chunked.used_count_map, unchunked.used_count_map)

def test_velocity_map_edge_clip_filter_reports_and_applies_consistently() -> None:
    rng = np.random.default_rng(14)
    movie = rng.standard_normal((24, 32, 32), dtype=np.float32)
    grid = GridSpec(patch_px=8, grid_stride_patches=1)

    common_kwargs = dict(
        movie=movie,
        fs=1000.0,
        grid=grid,
        bg_boxes_px=[(0, 8, 0, 8)],
        extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
        dj_mm=1.0,
        shifts=(1, 2),
        allow_bin_padding=False,
        use_shear_mask=False,
    )

    materialized = estimate_velocity_map(
        **common_kwargs,
        options=EstimationOptions(min_used=1, rmin=0.0, require_downstream=False, edge_clip_reject_k=1e6),
    )
    streaming = estimate_velocity_map_streaming(
        **common_kwargs,
        options=EstimationOptions(min_used=1, rmin=0.0, require_downstream=False, edge_clip_reject_k=1e6),
    )

    assert materialized.edge_clipped_by_shift is not None
    assert streaming.edge_clipped_by_shift is not None
    for s in (1, 2):
        np.testing.assert_array_equal(materialized.edge_clipped_by_shift[s], streaming.edge_clipped_by_shift[s])
        assert np.any(materialized.edge_clipped_by_shift[s])
