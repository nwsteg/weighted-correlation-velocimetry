from __future__ import annotations

import numpy as np

from .correlation import corr_matrix_positive_shift
from .geometry import build_shear_mask_patchvec, patch_centers_xy_m, validate_grid
from .preprocess import (
    block_mean_timeseries,
    build_background_regressors,
    detrend_series,
    regress_out_common_mode_2d,
)
from .types import EstimationOptions, GridSpec, VelocityMapResult


def estimate_velocity_map(
    movie: np.ndarray,
    fs: float,
    grid: GridSpec,
    bg_boxes_px: list[tuple[int, int, int, int]],
    extent_xd_yd: tuple[float, float, float, float],
    dj_mm: float,
    shifts: tuple[int, ...] = (1,),
    options: EstimationOptions = EstimationOptions(),
    use_shear_mask: bool = True,
    shear_mask_px: np.ndarray | None = None,
    shear_pctl: float = 50,
    origin: str = "upper",
    detrend_type: str = "linear",
) -> VelocityMapResult:
    f = np.asarray(movie, dtype=np.float32)
    nt, ny, nx = f.shape
    bin_px, by, bx = validate_grid(ny, nx, grid.patch_px, grid.grid_stride_patches)

    region_raw, _, _ = block_mean_timeseries(f, bin_px=bin_px)
    region_dt = detrend_series(region_raw, axis=1, detrend_type=detrend_type)

    g = build_background_regressors(f, bg_boxes_px, detrend_type=detrend_type)
    region_res = regress_out_common_mode_2d(region_dt, g)

    if use_shear_mask:
        shear = build_shear_mask_patchvec(
            f, by=by, bx=bx, bin_px=bin_px, shear_mask_px=shear_mask_px, shear_pctl=shear_pctl
        )
    else:
        shear = np.ones(by * bx, dtype=bool)

    x_m, y_m = patch_centers_xy_m(ny, nx, by, bx, bin_px, extent_xd_yd, dj_mm, origin=origin)

    shifts = tuple(sorted(set(int(s) for s in shifts if int(s) > 0)))
    if not shifts:
        raise ValueError("shifts must include at least one positive integer")

    corr_by_shift = {s: corr_matrix_positive_shift(region_res, s) for s in shifts}

    n_regions = by * bx
    ux = np.full(n_regions, np.nan, dtype=np.float32)
    uy = np.full(n_regions, np.nan, dtype=np.float32)
    used_count = np.zeros(n_regions, dtype=np.int32)

    seeds = np.flatnonzero(shear)
    valid = 0
    for j in seeds:
        dx_all = x_m - x_m[j]
        dy_all = y_m - y_m[j]

        gate = shear.copy()
        gate[j] = False
        if options.require_downstream:
            gate &= dx_all > 0

        taus, dxs, dys = [], [], []
        n_any = 0
        for s in shifts:
            r = corr_by_shift[s][:, j]
            m = gate & np.isfinite(r) & (r > 0) & (r >= options.rmin)
            if options.require_dy_positive:
                m &= dy_all > 0
            n = int(m.sum())
            if n < options.min_used:
                continue

            w = (r[m].astype(np.float64) ** options.weight_power)
            wsum = w.sum()
            if wsum <= 0 or not np.isfinite(wsum):
                continue

            taus.append(s / float(fs))
            dxs.append(float(np.sum(w * dx_all[m]) / wsum))
            dys.append(float(np.sum(w * dy_all[m]) / wsum))
            n_any += n

        used_count[j] = n_any
        if not taus:
            continue

        taus = np.asarray(taus)
        denom = np.sum(taus * taus)
        if denom > 0:
            ux[j] = float(np.sum(taus * np.asarray(dxs)) / denom)
            uy[j] = float(np.sum(taus * np.asarray(dys)) / denom)
            if np.isfinite(ux[j]) and np.isfinite(uy[j]):
                valid += 1

    return VelocityMapResult(
        ux_map=ux.reshape(by, bx),
        uy_map=uy.reshape(by, bx),
        used_count_map=used_count.reshape(by, bx),
        shear_mask_vec=shear,
        x_m=x_m,
        y_m=y_m,
        by=by,
        bx=bx,
        corr_by_shift=corr_by_shift,
        valid_seed_count=valid,
        total_seed_count=int(seeds.size),
    )
