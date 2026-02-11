from __future__ import annotations

import numpy as np

from .estimator_single_seed import estimate_single_seed_velocity
from .geometry import box_px_to_extent_xywh
from .plotting import plot_seed_and_background_boxes
from .types import EstimationOptions, GridSpec


def estimate_velocity_per_shift_framework(
    bgs_f,
    fs,
    PATCH_PX,
    seed_box_px,
    bg_boxes_px,
    extent_xD_yD,
    Dj_mm,
    shifts=(1, 2, 3),
    RMIN=0.2,
    MIN_USED=10,
    REQUIRE_DOWNSTREAM=True,
    REQUIRE_DY_POSITIVE=False,
    origin="upper",
    detrend_type="linear",
    grid_stride_patches=1,
    make_plots=True,
    frame_vis=None,
    **_,
):
    """Compatibility wrapper matching the old notebook API.

    Returns
    -------
    Ux, Uy, diag
        Same tuple pattern as the notebook function.
    """
    opts = EstimationOptions(
        rmin=RMIN,
        min_used=MIN_USED,
        require_downstream=REQUIRE_DOWNSTREAM,
        require_dy_positive=REQUIRE_DY_POSITIVE,
    )
    result = estimate_single_seed_velocity(
        movie=np.asarray(bgs_f),
        fs=fs,
        grid=GridSpec(patch_px=int(PATCH_PX), grid_stride_patches=int(grid_stride_patches)),
        seed_box_px=tuple(seed_box_px),
        bg_boxes_px=[tuple(b) for b in bg_boxes_px],
        extent_xd_yd=tuple(extent_xD_yD),
        dj_mm=Dj_mm,
        shifts=tuple(shifts),
        options=opts,
        origin=origin,
        detrend_type=detrend_type,
    )

    diag = {
        "fs": float(fs),
        "PATCH_PX": int(PATCH_PX),
        "grid_stride_patches": int(grid_stride_patches),
        "BIN_PX": int(PATCH_PX) * int(grid_stride_patches),
        "shifts": list(result.shifts),
        "shifts_used_for_fit": list(result.shifts_used_for_fit),
        "By": int(result.by),
        "Bx": int(result.bx),
        "dx_m": result.dx_m,
        "dy_m": result.dy_m,
        "r_maps": result.corr_by_shift,
        "masks": result.mask_by_shift,
        "dx_bar": result.dx_bar_by_shift,
        "dy_bar": result.dy_bar_by_shift,
        "n_used": result.n_used_by_shift,
        "Ux": float(result.ux),
        "Uy": float(result.uy),
    }

    if make_plots:
        import matplotlib.pyplot as plt

        if frame_vis is None:
            frame_vis = np.median(np.asarray(bgs_f), axis=0)
        plot_seed_and_background_boxes(frame_vis, extent_xD_yD, seed_box_px, bg_boxes_px, origin=origin)
        plt.show()

    return float(result.ux), float(result.uy), diag


def draw_boxes_debug(frame_vis, extent, seed_box_px, bg_boxes_px, BG_ANCHORS=None, origin="upper"):
    """Legacy helper to plot seed and BG boxes on a representative frame."""
    return plot_seed_and_background_boxes(frame_vis, extent, seed_box_px, bg_boxes_px, origin=origin)


__all__ = [
    "estimate_velocity_per_shift_framework",
    "draw_boxes_debug",
    "box_px_to_extent_xywh",
]
