from __future__ import annotations

import numpy as np

from wcv import EstimationOptions, GridSpec, estimate_single_seed_velocity


def test_single_seed_edge_clip_filter_can_disable_shift_fit() -> None:
    rng = np.random.default_rng(4)
    movie = rng.standard_normal((24, 32, 32), dtype=np.float32)
    grid = GridSpec(patch_px=8, grid_stride_patches=1)
    kwargs = dict(
        movie=movie,
        fs=1000.0,
        grid=grid,
        seed_box_px=(0, 8, 0, 8),
        bg_boxes_px=[(0, 8, 0, 8)],
        extent_xd_yd=(0.0, 1.0, 0.0, 1.0),
        dj_mm=1.0,
        shifts=(1, 2),
        allow_bin_padding=False,
    )

    base = estimate_single_seed_velocity(
        **kwargs,
        options=EstimationOptions(min_used=1, rmin=0.0, require_downstream=False, edge_clip_reject_k=None),
    )
    clipped = estimate_single_seed_velocity(
        **kwargs,
        options=EstimationOptions(min_used=1, rmin=0.0, require_downstream=False, edge_clip_reject_k=1e6),
    )

    assert clipped.edge_clipped_by_shift is not None
    assert clipped.edge_distance_by_shift is not None
    assert clipped.support_radius_by_shift is not None
    assert all(clipped.edge_clipped_by_shift[s] for s in (1, 2))
    assert len(clipped.shifts_used_for_fit) <= len(base.shifts_used_for_fit)
