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
    allow_bin_padding=False,
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
        allow_bin_padding=bool(allow_bin_padding),
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
        plot_single_seed_correlation_maps(result, frame_vis, extent_xD_yD, vmin=-0.25, vmax=0.25)

        # 3) Per-shift scatter cloud of accepted bins: dx vs tau (sizes ~ |r|^2)
        def _sizes_from_weights(w, smin=14.0, smax=120.0):
            w = np.asarray(w, dtype=float)
            if w.size == 0:
                return w
            scale = np.nanpercentile(w, 95) + 1e-12
            w0 = np.clip(w / scale, 0.0, 1.0)
            return smin + (smax - smin) * w0

        shifts_fit = [s for s in result.shifts if s != 0]

        fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        taus_used = []
        for s in shifts_fit:
            m = np.asarray(result.mask_by_shift.get(s, np.zeros_like(result.dx_m, dtype=bool)), dtype=bool)
            if not np.any(m):
                continue
            tau = float(s) / float(fs)
            r = np.asarray(result.corr_by_shift[s], dtype=float)
            w = np.abs(r[m]) ** 2
            sizes = _sizes_from_weights(w)
            ax.scatter(np.full(np.sum(m), tau), result.dx_m[m], s=sizes, alpha=0.35, label=f"shift={s}")

            dxc = result.dx_bar_by_shift.get(s, np.nan)
            if np.isfinite(dxc):
                ax.scatter([tau], [dxc], s=170, marker="X")
                taus_used.append(tau)

        if np.isfinite(result.ux) and len(taus_used):
            tline = np.linspace(0.0, max(taus_used) * 1.1, 200)
            ax.plot(tline, float(result.ux) * tline, "k-", lw=2, label=f"fit: Ux={result.ux:.2f} m/s")

        ax.axhline(0, lw=1)
        ax.axvline(0, lw=1)
        ax.set_xlabel("tau (s) = shift / fs")
        ax.set_ylabel("dx (m)")
        ax.set_title("Per-shift scatter of accepted bins: dx vs tau (sizes ~ |r|^2); X = centroid")
        if ax.has_data():
            ax.legend(fontsize=9)

        # 4) dx vs dy scatter cloud of accepted bins (per shift)
        fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        for s in shifts_fit:
            m = np.asarray(result.mask_by_shift.get(s, np.zeros_like(result.dx_m, dtype=bool)), dtype=bool)
            if not np.any(m):
                continue
            r = np.asarray(result.corr_by_shift[s], dtype=float)
            w = np.abs(r[m]) ** 2
            sizes = _sizes_from_weights(w)
            ax.scatter(result.dx_m[m], result.dy_m[m], s=sizes, alpha=0.35, label=f"shift={s}")

            dxc = result.dx_bar_by_shift.get(s, np.nan)
            dyc = result.dy_bar_by_shift.get(s, np.nan)
            if np.isfinite(dxc) and np.isfinite(dyc):
                ax.scatter([dxc], [dyc], s=180, marker="X")

        if np.isfinite(result.ux) and np.isfinite(result.uy) and abs(float(result.ux)) > 1e-12:
            slope = float(result.uy) / float(result.ux)
            x_min, x_max = np.nanmin(result.dx_m), np.nanmax(result.dx_m)
            xline = np.linspace(float(x_min), float(x_max), 200)
            ax.plot(xline, slope * xline, "k-", lw=2, label=f"direction: Uy/Ux={slope:.3f}")

        ax.axhline(0, lw=1)
        ax.axvline(0, lw=1)
        ax.set_xlabel("dx (m)")
        ax.set_ylabel("dy (m)")
        ax.set_title("Accepted-bin scatter in dx-dy (sizes ~ |r|^2); X = centroid")
        if ax.has_data():
            ax.legend(fontsize=9)

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
