from __future__ import annotations

import numpy as np

from .estimator_single_seed import estimate_single_seed_velocity
from .geometry import box_px_to_extent_xywh
from .plotting import plot_seed_and_background_boxes, plot_single_seed_correlation_maps
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

        # 1) Seed + background clumps overlay
        plot_seed_and_background_boxes(frame_vis, extent_xD_yD, seed_box_px, bg_boxes_px, origin=origin)

        # 2) Per-shift correlation maps with accepted-mask overlay
        plot_single_seed_correlation_maps(result, frame_vis, extent_xD_yD)

        # 3) Centroid displacement vs lag (through-origin fit)
        shifts_fit = [s for s in result.shifts if s != 0]
        taus, dxs, dys = [], [], []
        for s in shifts_fit:
            dx = result.dx_bar_by_shift.get(s, np.nan)
            dy = result.dy_bar_by_shift.get(s, np.nan)
            if np.isfinite(dx) and np.isfinite(dy):
                taus.append(float(s) / float(fs))
                dxs.append(float(dx))
                dys.append(float(dy))

        if taus:
            taus = np.asarray(taus, dtype=float)
            dxs = np.asarray(dxs, dtype=float)
            dys = np.asarray(dys, dtype=float)

            fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)

            axes[0].scatter(taus, dxs, s=50)
            tline = np.linspace(0.0, float(taus.max()) * 1.1, 100)
            axes[0].plot(tline, float(result.ux) * tline, "k-", lw=2)
            axes[0].set_xlabel("tau (s)")
            axes[0].set_ylabel("dx_bar (m)")
            axes[0].set_title(f"dx vs tau (Ux={result.ux:.3g} m/s)")

            axes[1].scatter(taus, dys, s=50)
            axes[1].plot(tline, float(result.uy) * tline, "k-", lw=2)
            axes[1].set_xlabel("tau (s)")
            axes[1].set_ylabel("dy_bar (m)")
            axes[1].set_title(f"dy vs tau (Uy={result.uy:.3g} m/s)")

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
