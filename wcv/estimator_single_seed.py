from __future__ import annotations

import warnings

import numpy as np

from .correlation import shifted_corr_regions
from .geometry import patch_centers_dxdy_m, validate_grid
from .preprocess import (
    block_mean_timeseries,
    build_background_regressors,
    detrend_series,
    regress_out_common_mode_2d,
)
from .types import EstimationOptions, GridSpec, SingleSeedResult


def estimate_single_seed_velocity(
    movie: np.ndarray,
    fs: float,
    grid: GridSpec,
    seed_box_px: tuple[int, int, int, int],
    bg_boxes_px: list[tuple[int, int, int, int]],
    extent_xd_yd: tuple[float, float, float, float],
    dj_mm: float,
    shifts: tuple[int, ...] = (1, 2, 3),
    options: EstimationOptions = EstimationOptions(),
    allow_bin_padding: bool = False,
    origin: str = "upper",
    detrend_type: str = "linear",
) -> SingleSeedResult:
    f = np.asarray(movie, dtype=np.float32)
    _, ny, nx = f.shape
    bin_px = grid.bin_px
    if allow_bin_padding and (ny % bin_px or nx % bin_px):
        ny_pad = ((ny + bin_px - 1) // bin_px) * bin_px
        nx_pad = ((nx + bin_px - 1) // bin_px) * bin_px
        f = np.pad(f, ((0, 0), (0, ny_pad - ny), (0, nx_pad - nx)), mode="edge")
        warnings.warn(
            f"Input movie shape ({ny},{nx}) was padded to ({ny_pad},{nx_pad}) to satisfy "
            "bin_px divisibility; edge padding can affect correlation/velocity estimates near "
            "image borders.",
            UserWarning,
            stacklevel=2,
        )

    _, ny, nx = f.shape
    bin_px, by, bx = validate_grid(ny, nx, grid.patch_px, grid.grid_stride_patches)

    sy0, sy1, sx0, sx1 = map(int, seed_box_px)
    if (sy1 - sy0) != bin_px or (sx1 - sx0) != bin_px:
        raise ValueError(f"seed_box_px must be exactly one BIN_PX cell ({bin_px}x{bin_px}).")

    y, by_chk, bx_chk = block_mean_timeseries(f, bin_px=bin_px)
    assert by_chk == by and bx_chk == bx

    y_dt = detrend_series(y, axis=1, detrend_type=detrend_type)
    seed_raw = f[:, sy0:sy1, sx0:sx1].mean(axis=(1, 2))
    seed_dt = detrend_series(seed_raw, detrend_type=detrend_type)

    g = build_background_regressors(f, bg_boxes_px, detrend_type=detrend_type)
    seed_res = regress_out_common_mode_2d(seed_dt[None, :], g)[0]
    region_res = regress_out_common_mode_2d(y_dt, g)

    dx_m, dy_m = patch_centers_dxdy_m(
        ny, nx, by, bx, bin_px, extent_xd_yd, seed_box_px=seed_box_px, dj_mm=dj_mm, origin=origin
    )

    shifts = tuple(int(s) for s in shifts)
    shifts_fit = [s for s in shifts if s != 0]

    corr_by_shift: dict[int, np.ndarray] = {}
    mask_by_shift: dict[int, np.ndarray] = {}
    dx_bar_by_shift: dict[int, float] = {}
    dy_bar_by_shift: dict[int, float] = {}
    n_used_by_shift: dict[int, int] = {}

    for s in shifts:
        r = shifted_corr_regions(region_res, seed_res, s)
        corr_by_shift[s] = r

        if s == 0:
            mask_by_shift[s] = np.zeros_like(r, dtype=bool)
            dx_bar_by_shift[s] = np.nan
            dy_bar_by_shift[s] = np.nan
            n_used_by_shift[s] = 0
            continue

        mask = np.isfinite(r)
        mask &= np.abs(r) >= options.rmin
        mask &= r > 0.0
        if options.require_downstream:
            mask &= dx_m > 0
        if options.require_dy_positive:
            mask &= dy_m > 0

        mask_by_shift[s] = mask
        n_used_by_shift[s] = int(mask.sum())
        if n_used_by_shift[s] < options.min_used:
            dx_bar_by_shift[s] = np.nan
            dy_bar_by_shift[s] = np.nan
            continue

        w = (np.abs(r[mask]) ** options.weight_power).astype(np.float64)
        wsum = w.sum()
        if not np.isfinite(wsum) or wsum <= 0:
            dx_bar_by_shift[s] = np.nan
            dy_bar_by_shift[s] = np.nan
            continue

        dx_bar_by_shift[s] = float(np.sum(w * dx_m[mask]) / wsum)
        dy_bar_by_shift[s] = float(np.sum(w * dy_m[mask]) / wsum)

    taus, dxs, dys, used = [], [], [], []
    for s in shifts_fit:
        dxv = dx_bar_by_shift.get(s, np.nan)
        dyv = dy_bar_by_shift.get(s, np.nan)
        if np.isfinite(dxv) and np.isfinite(dyv):
            taus.append(s / float(fs))
            dxs.append(dxv)
            dys.append(dyv)
            used.append(s)

    if len(taus) == 0:
        ux, uy = np.nan, np.nan
    else:
        taus = np.asarray(taus)
        denom = np.sum(taus * taus)
        ux = float(np.sum(taus * np.asarray(dxs)) / denom) if denom > 0 else np.nan
        uy = float(np.sum(taus * np.asarray(dys)) / denom) if denom > 0 else np.nan

    return SingleSeedResult(
        ux=ux,
        uy=uy,
        shifts=shifts,
        shifts_used_for_fit=used,
        dx_bar_by_shift=dx_bar_by_shift,
        dy_bar_by_shift=dy_bar_by_shift,
        n_used_by_shift=n_used_by_shift,
        corr_by_shift=corr_by_shift,
        mask_by_shift=mask_by_shift,
        dx_m=dx_m,
        dy_m=dy_m,
        by=by,
        bx=bx,
    )
