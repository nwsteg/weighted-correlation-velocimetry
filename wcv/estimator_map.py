from __future__ import annotations

from collections.abc import Callable
import warnings

import numpy as np

from .correlation import (
    corr_matrix_positive_shift,
    corr_targets_for_seed_block_positive_shift,
    corr_targets_for_seed_positive_shift,
    sparse_useful_corr_row,
)
from .geometry import (
    build_shear_mask_patchvec,
    compute_bin_aligned_padding,
    patch_centers_xy_m,
    validate_grid,
)
from .preprocess import (
    block_mean_timeseries,
    build_background_regressors,
    detrend_series,
    regress_out_common_mode_2d,
)
from .types import EstimationOptions, GridSpec, VelocityMapResult


ProgressCallback = Callable[[int, int, str], None]


class _StageProgress:
    def __init__(self, stage: str, total: int, on_progress: ProgressCallback | None = None):
        self.stage = stage
        self.total = int(total)
        self.done = 0
        self._on_progress = on_progress

    def advance(self, n: int = 1) -> None:
        self.done = min(self.total, self.done + int(n))
        if self._on_progress is not None:
            self._on_progress(self.done, self.total, self.stage)

    def close(self) -> None:
        if self._on_progress is not None and self.done < self.total:
            self._on_progress(self.total, self.total, self.stage)


def _edge_clip_diagnostic(
    *,
    mask: np.ndarray,
    weights: np.ndarray,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    by: int,
    bx: int,
    options: EstimationOptions,
) -> tuple[bool, float, float]:
    if options.edge_clip_reject_k is None:
        return False, np.nan, np.nan
    if mask.ndim != 1 or mask.size != row_coords.size:
        return False, np.nan, np.nan

    idx = np.flatnonzero(mask)
    if idx.size == 0 or weights.size != idx.size:
        return False, np.nan, np.nan

    wsum = float(weights.sum())
    if wsum <= 0 or not np.isfinite(wsum):
        return False, np.nan, np.nan

    r = row_coords[idx]
    c = col_coords[idx]
    r_bar = float(np.sum(weights * r) / wsum)
    c_bar = float(np.sum(weights * c) / wsum)

    var_r = float(np.sum(weights * (r - r_bar) ** 2) / wsum)
    var_c = float(np.sum(weights * (c - c_bar) ** 2) / wsum)
    support_radius = float(options.edge_clip_sigma_mult) * float(np.sqrt(max(var_r + var_c, 0.0)))

    edge_distance = float(min(r_bar, c_bar, (by - 1) - r_bar, (bx - 1) - c_bar))
    clipped = bool(edge_distance < float(options.edge_clip_reject_k) * support_radius)
    return clipped, edge_distance, support_radius


def _fit_seed_from_corr_vectors(
    *,
    seed_idx: int,
    target_gate_vec: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    by: int,
    bx: int,
    shifts: tuple[int, ...],
    fs: float,
    options: EstimationOptions,
    corr_for_shift: Callable[[int], np.ndarray],
    edge_clipped_by_shift: dict[int, bool] | None = None,
) -> tuple[float, float, int]:
    dx_all = x_m - x_m[seed_idx]
    dy_all = y_m - y_m[seed_idx]

    gate = target_gate_vec.copy()
    gate[seed_idx] = False
    if options.require_downstream:
        gate &= dx_all > 0

    taus, dxs, dys = [], [], []
    n_any = 0
    for s in shifts:
        r = corr_for_shift(int(s))
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

        edge_clipped, _, _ = _edge_clip_diagnostic(
            mask=m,
            weights=w,
            row_coords=row_coords,
            col_coords=col_coords,
            by=by,
            bx=bx,
            options=options,
        )
        if edge_clipped_by_shift is not None:
            edge_clipped_by_shift[int(s)] = edge_clipped
        if edge_clipped:
            continue

        taus.append(s / float(fs))
        dxs.append(float(np.sum(w * dx_all[m]) / wsum))
        dys.append(float(np.sum(w * dy_all[m]) / wsum))
        n_any += n

    if not taus:
        return np.nan, np.nan, n_any

    taus_arr = np.asarray(taus)
    denom = np.sum(taus_arr * taus_arr)
    if denom <= 0:
        return np.nan, np.nan, n_any

    ux_j = float(np.sum(taus_arr * np.asarray(dxs)) / denom)
    uy_j = float(np.sum(taus_arr * np.asarray(dys)) / denom)
    return ux_j, uy_j, n_any


def _fit_seed_from_sparse_vectors(
    *,
    seed_idx: int,
    x_m: np.ndarray,
    y_m: np.ndarray,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    by: int,
    bx: int,
    shifts: tuple[int, ...],
    fs: float,
    options: EstimationOptions,
    sparse_corr_for_shift: Callable[[int], tuple[np.ndarray, np.ndarray]],
    edge_clipped_by_shift: dict[int, bool] | None = None,
) -> tuple[float, float, int]:
    dx_all = x_m - x_m[seed_idx]
    dy_all = y_m - y_m[seed_idx]

    taus, dxs, dys = [], [], []
    n_any = 0
    for s in shifts:
        idx, vals = sparse_corr_for_shift(int(s))
        if idx.size == 0:
            continue

        if options.require_dy_positive:
            keep = dy_all[idx] > 0
            idx = idx[keep]
            vals = vals[keep]
        n = int(idx.size)
        if n < options.min_used:
            continue

        w = (vals.astype(np.float64) ** options.weight_power)
        wsum = w.sum()
        if wsum <= 0 or not np.isfinite(wsum):
            continue

        full_mask = np.zeros_like(dx_all, dtype=bool)
        full_mask[idx] = True
        edge_clipped, _, _ = _edge_clip_diagnostic(
            mask=full_mask,
            weights=w,
            row_coords=row_coords,
            col_coords=col_coords,
            by=by,
            bx=bx,
            options=options,
        )
        if edge_clipped_by_shift is not None:
            edge_clipped_by_shift[int(s)] = edge_clipped
        if edge_clipped:
            continue

        taus.append(s / float(fs))
        dxs.append(float(np.sum(w * dx_all[idx]) / wsum))
        dys.append(float(np.sum(w * dy_all[idx]) / wsum))
        n_any += n

    if not taus:
        return np.nan, np.nan, n_any

    taus_arr = np.asarray(taus)
    denom = np.sum(taus_arr * taus_arr)
    if denom <= 0:
        return np.nan, np.nan, n_any

    ux_j = float(np.sum(taus_arr * np.asarray(dxs)) / denom)
    uy_j = float(np.sum(taus_arr * np.asarray(dys)) / denom)
    return ux_j, uy_j, n_any


def _resolve_progress_callback(
    show_progress: bool,
    progress_factory: Callable[[int, str], object] | None,
    on_progress: ProgressCallback | None,
) -> Callable[[str, int], _StageProgress] | None:
    if not show_progress and progress_factory is None and on_progress is None:
        return None

    user_cb = on_progress
    tqdm_factory = None
    if show_progress and progress_factory is None:
        try:
            from tqdm.auto import tqdm as tqdm_factory  # type: ignore
        except Exception:
            tqdm_factory = None

    def build_stage(stage: str, total: int) -> _StageProgress:
        stage_cb = user_cb
        factory = progress_factory
        bar = None

        if factory is not None:
            instance = factory(int(total), stage)
            if callable(instance):
                stage_cb = instance
            elif hasattr(instance, "update"):
                def _bar_cb(done: int, _total: int, _stage: str) -> None:
                    update = int(done - getattr(_bar_cb, "_last_done", 0))
                    if update > 0:
                        instance.update(update)
                    setattr(_bar_cb, "_last_done", done)

                setattr(_bar_cb, "_last_done", 0)
                stage_cb = _bar_cb
                bar = instance

        elif tqdm_factory is not None:
            bar = tqdm_factory(total=int(total), desc=stage, leave=False)

            def _tqdm_cb(done: int, _total: int, _stage: str) -> None:
                update = int(done - getattr(_tqdm_cb, "_last_done", 0))
                if update > 0:
                    bar.update(update)
                setattr(_tqdm_cb, "_last_done", done)

            setattr(_tqdm_cb, "_last_done", 0)
            stage_cb = _tqdm_cb

        p = _StageProgress(stage=stage, total=int(total), on_progress=stage_cb)
        _base_close = p.close

        def _close() -> None:
            _base_close()
            if bar is not None and hasattr(bar, "close"):
                bar.close()

        p.close = _close  # type: ignore[method-assign]
        return p

    return build_stage



def estimate_velocity_map_streaming(
    movie: np.ndarray,
    fs: float,
    grid: GridSpec,
    bg_boxes_px: list[tuple[int, int, int, int]],
    extent_xd_yd: tuple[float, float, float, float],
    dj_mm: float,
    shifts: tuple[int, ...] = (1,),
    options: EstimationOptions = EstimationOptions(),
    allow_bin_padding: bool = False,
    use_shear_mask: bool = True,
    shear_mask_px: np.ndarray | None = None,
    seed_mask_px: np.ndarray | None = None,
    shear_pctl: float = 50,
    origin: str = "upper",
    detrend_type: str = "linear",
    store_corr_by_shift: bool = False,
    seed_chunk_size: int = 1,
    shift_chunk_size: int | None = None,
    max_corr_buffer_mb: float | None = None,
    show_progress: bool = False,
    progress_factory: Callable[[int, str], object] | None = None,
    on_progress: ProgressCallback | None = None,
) -> VelocityMapResult:
    """Estimate a velocity map by streaming correlations seed-by-seed.

    Unlike :func:`estimate_velocity_map`, this function does not materialize
    full ``(target, seed)`` correlation matrices unless ``store_corr_by_shift``
    is enabled for diagnostics.

    Progress hooks are optional and disabled by default to keep overhead low.
    """
    stage_progress = _resolve_progress_callback(show_progress, progress_factory, on_progress)
    preprocess = stage_progress("preprocessing", 4) if stage_progress is not None else None

    f = np.asarray(movie, dtype=np.float32)
    _, ny, nx = f.shape
    original_shape = (ny, nx)
    padded = False
    bin_px = grid.bin_px
    use_padding = bool(allow_bin_padding or options.allow_bin_padding)
    if use_padding and (ny % bin_px or nx % bin_px):
        ny_pad, nx_pad, pad_y, pad_x = compute_bin_aligned_padding(ny, nx, bin_px)
        f = np.pad(f, ((0, 0), (0, pad_y), (0, pad_x)), mode=options.padding_mode)
        padded = True
        warnings.warn(
            f"Input movie shape ({ny},{nx}) was padded to ({ny_pad},{nx_pad}) to satisfy "
            f"bin_px divisibility using mode={options.padding_mode!r}; edge padding can affect "
            "correlation/velocity estimates near image borders.",
            UserWarning,
            stacklevel=2,
        )

    _, ny, nx = f.shape
    padded_shape = (ny, nx)
    bin_px, by, bx = validate_grid(ny, nx, grid.patch_px, grid.grid_stride_patches)

    region_raw, _, _ = block_mean_timeseries(f, bin_px=bin_px)
    region_dt = detrend_series(region_raw, axis=1, detrend_type=detrend_type)
    if preprocess is not None:
        preprocess.advance()

    g = build_background_regressors(f, bg_boxes_px, detrend_type=detrend_type)
    region_res = regress_out_common_mode_2d(region_dt, g)
    if preprocess is not None:
        preprocess.advance()

    if use_shear_mask:
        target_gate_vec = build_shear_mask_patchvec(
            f, by=by, bx=bx, bin_px=bin_px, shear_mask_px=shear_mask_px, shear_pctl=shear_pctl
        )
    else:
        target_gate_vec = np.ones(by * bx, dtype=bool)

    if seed_mask_px is not None:
        seed_gate_vec = build_shear_mask_patchvec(
            f, by=by, bx=bx, bin_px=bin_px, shear_mask_px=seed_mask_px, shear_pctl=shear_pctl
        )
    else:
        seed_gate_vec = target_gate_vec
    if preprocess is not None:
        preprocess.advance()

    x_m, y_m = patch_centers_xy_m(ny, nx, by, bx, bin_px, extent_xd_yd, dj_mm, origin=origin)
    if preprocess is not None:
        preprocess.advance()
        preprocess.close()

    shifts = tuple(sorted(set(int(s) for s in shifts if int(s) > 0)))
    if not shifts:
        raise ValueError("shifts must include at least one positive integer")

    n_regions = by * bx
    row_coords = np.repeat(np.arange(by, dtype=np.float64), bx)
    col_coords = np.tile(np.arange(bx, dtype=np.float64), by)
    ux = np.full(n_regions, np.nan, dtype=np.float32)
    uy = np.full(n_regions, np.nan, dtype=np.float32)
    used_count = np.zeros(n_regions, dtype=np.int32)

    corr_by_shift = (
        {s: np.full((n_regions, n_regions), np.nan, dtype=np.float32) for s in shifts}
        if store_corr_by_shift
        else {}
    )
    edge_clipped_by_shift = {s: np.zeros(n_regions, dtype=bool) for s in shifts} if options.edge_clip_reject_k is not None else None

    shift_stats: dict[int, dict[str, np.ndarray | float]] = {}
    for s in shifts:
        n1 = region_res.shape[1] - int(s)
        if n1 < 3:
            continue
        target_slice = region_res[:, s:].astype(np.float64, copy=False)
        seed_slice = region_res[:, :-s].astype(np.float64, copy=False)
        shift_stats[s] = {
            "target_row_means": target_slice.mean(axis=1),
            "target_row_stds": target_slice.std(axis=1, ddof=1) + 1e-12,
            "seed_row_means": seed_slice.mean(axis=1),
            "seed_row_stds": seed_slice.std(axis=1, ddof=1) + 1e-12,
            "norm_denom": float(n1 - 1),
        }

    seeds = np.flatnonzero(seed_gate_vec)
    chunk_size = 1 if seed_chunk_size is None else max(1, int(seed_chunk_size))
    shift_chunk = len(shifts) if shift_chunk_size is None else max(1, int(shift_chunk_size))
    shift_chunk = min(shift_chunk, len(shifts))

    if max_corr_buffer_mb is not None:
        max_buffer_bytes = int(float(max_corr_buffer_mb) * 1024 * 1024)
        if max_buffer_bytes <= 0:
            raise ValueError("max_corr_buffer_mb must be > 0 when provided")
        bytes_per_seed_for_shift_chunk = int(shift_chunk * n_regions * np.dtype(np.float32).itemsize)
        if bytes_per_seed_for_shift_chunk <= 0:
            bytes_per_seed_for_shift_chunk = 1
        chunk_size = max(1, min(chunk_size, max_buffer_bytes // bytes_per_seed_for_shift_chunk))

    use_seed_blocks = chunk_size > 1 and seeds.size > chunk_size
    seed_progress = stage_progress("seed loop progress", int(seeds.size)) if stage_progress is not None else None
    valid = 0
    if not use_seed_blocks and shift_chunk == len(shifts):
        for j in seeds:
            corr_cache: dict[int, np.ndarray] = {}
            seed_edge_diag: dict[int, bool] = {}

            def _corr_for_shift(s: int) -> np.ndarray:
                r = corr_cache.get(s)
                if r is None:
                    stats = shift_stats.get(s)
                    if stats is None:
                        r = corr_targets_for_seed_positive_shift(region_res, int(j), int(s))
                    else:
                        r = corr_targets_for_seed_positive_shift(
                            region_res,
                            int(j),
                            int(s),
                            target_row_means=stats["target_row_means"],
                            target_row_stds=stats["target_row_stds"],
                            seed_row_means=stats["seed_row_means"],
                            seed_row_stds=stats["seed_row_stds"],
                            norm_denom=stats["norm_denom"],
                        )
                    corr_cache[s] = r
                    if store_corr_by_shift:
                        corr_by_shift[s][:, j] = r
                return r

            ux_j, uy_j, n_any = _fit_seed_from_corr_vectors(
                seed_idx=int(j),
                target_gate_vec=target_gate_vec,
                x_m=x_m,
                y_m=y_m,
                row_coords=row_coords,
                col_coords=col_coords,
                by=by,
                bx=bx,
                shifts=shifts,
                fs=fs,
                options=options,
                corr_for_shift=_corr_for_shift,
                edge_clipped_by_shift=seed_edge_diag if edge_clipped_by_shift is not None else None,
            )
            used_count[j] = n_any
            if edge_clipped_by_shift is not None:
                for s in shifts:
                    edge_clipped_by_shift[s][j] = bool(seed_edge_diag.get(int(s), False))
            ux[j] = ux_j
            uy[j] = uy_j
            if np.isfinite(ux_j) and np.isfinite(uy_j):
                valid += 1
            if seed_progress is not None:
                seed_progress.advance()
    else:
        tau_by_shift = {s: float(s) / float(fs) for s in shifts}
        for start in range(0, seeds.size, chunk_size):
            seed_chunk = seeds[start : start + chunk_size]
            numer_x = np.zeros(seed_chunk.size, dtype=np.float64)
            numer_y = np.zeros(seed_chunk.size, dtype=np.float64)
            denom = np.zeros(seed_chunk.size, dtype=np.float64)
            used_chunk = np.zeros(seed_chunk.size, dtype=np.int32)

            dx_block = x_m[None, :] - x_m[seed_chunk][:, None]
            dy_block = y_m[None, :] - y_m[seed_chunk][:, None]
            gate_block = np.broadcast_to(target_gate_vec[None, :], (seed_chunk.size, n_regions)).copy()
            gate_block[np.arange(seed_chunk.size), seed_chunk] = False
            if options.require_downstream:
                gate_block &= dx_block > 0

            for shift_start in range(0, len(shifts), shift_chunk):
                shift_slice = shifts[shift_start : shift_start + shift_chunk]
                corr_by_shift_block: dict[int, np.ndarray] = {}

                for s in shift_slice:
                    stats = shift_stats.get(s)
                    if stats is None:
                        block = corr_targets_for_seed_block_positive_shift(region_res, seed_chunk, int(s))
                    else:
                        block = corr_targets_for_seed_block_positive_shift(
                            region_res,
                            seed_chunk,
                            int(s),
                            target_row_means=stats["target_row_means"],
                            target_row_stds=stats["target_row_stds"],
                            seed_row_means=stats["seed_row_means"],
                            seed_row_stds=stats["seed_row_stds"],
                            norm_denom=stats["norm_denom"],
                        )
                    corr_by_shift_block[s] = block
                    if store_corr_by_shift:
                        corr_by_shift[s][:, seed_chunk] = block.T

                for s in shift_slice:
                    tau = tau_by_shift[s]
                    block = corr_by_shift_block[s]
                    keep = gate_block & np.isfinite(block) & (block > 0) & (block >= options.rmin)
                    if options.require_dy_positive:
                        keep &= dy_block > 0

                    for k in range(seed_chunk.size):
                        m = keep[k]
                        n = int(m.sum())
                        if n < options.min_used:
                            continue
                        vals = block[k, m]
                        weights = vals.astype(np.float64) ** options.weight_power
                        wsum = float(weights.sum())
                        if wsum <= 0 or not np.isfinite(wsum):
                            continue

                        edge_clipped, _, _ = _edge_clip_diagnostic(
                            mask=m,
                            weights=weights,
                            row_coords=row_coords,
                            col_coords=col_coords,
                            by=by,
                            bx=bx,
                            options=options,
                        )
                        if edge_clipped_by_shift is not None:
                            edge_clipped_by_shift[s][seed_chunk[k]] = edge_clipped
                        if edge_clipped:
                            continue

                        dxs = float(np.sum(weights * dx_block[k, m]) / wsum)
                        dys = float(np.sum(weights * dy_block[k, m]) / wsum)
                        numer_x[k] += tau * dxs
                        numer_y[k] += tau * dys
                        denom[k] += tau * tau
                        used_chunk[k] += n

            for k, j in enumerate(seed_chunk):
                if denom[k] > 0:
                    ux_j = float(numer_x[k] / denom[k])
                    uy_j = float(numer_y[k] / denom[k])
                else:
                    ux_j, uy_j = np.nan, np.nan

                used_count[j] = used_chunk[k]
                ux[j] = ux_j
                uy[j] = uy_j
                if np.isfinite(ux_j) and np.isfinite(uy_j):
                    valid += 1
                if seed_progress is not None:
                    seed_progress.advance()
    if seed_progress is not None:
        seed_progress.close()

    return VelocityMapResult(
        ux_map=ux.reshape(by, bx),
        uy_map=uy.reshape(by, bx),
        used_count_map=used_count.reshape(by, bx),
        shear_mask_vec=target_gate_vec,
        x_m=x_m,
        y_m=y_m,
        by=by,
        bx=bx,
        corr_by_shift=corr_by_shift,
        valid_seed_count=valid,
        total_seed_count=int(seeds.size),
        padded=padded,
        original_shape=original_shape,
        padded_shape=padded_shape,
        edge_clipped_by_shift=edge_clipped_by_shift,
    )


def estimate_velocity_map_hybrid(*args, **kwargs) -> VelocityMapResult:
    """Backward-compatible alias of :func:`estimate_velocity_map_streaming`."""
    return estimate_velocity_map_streaming(*args, **kwargs)

def estimate_velocity_map(
    movie: np.ndarray,
    fs: float,
    grid: GridSpec,
    bg_boxes_px: list[tuple[int, int, int, int]],
    extent_xd_yd: tuple[float, float, float, float],
    dj_mm: float,
    shifts: tuple[int, ...] = (1,),
    options: EstimationOptions = EstimationOptions(),
    allow_bin_padding: bool = False,
    use_shear_mask: bool = True,
    shear_mask_px: np.ndarray | None = None,
    seed_mask_px: np.ndarray | None = None,
    shear_pctl: float = 50,
    origin: str = "upper",
    detrend_type: str = "linear",
    sparse_corr_storage: bool | None = None,
    sparse_top_k: int | None = None,
    show_progress: bool = False,
    progress_factory: Callable[[int, str], object] | None = None,
    on_progress: ProgressCallback | None = None,
) -> VelocityMapResult:
    """Estimate a velocity map over all seed bins on the analysis grid.

    Padding behavior is controlled by the estimator argument
    ``allow_bin_padding`` (preferred) and, for backward compatibility,
    ``options.allow_bin_padding``. If either is ``True``, non-divisible input
    dimensions are edge-padded to the nearest ``grid.bin_px`` multiple and a
    ``UserWarning`` is emitted once per call.

    Progress hooks are optional and disabled by default to keep overhead low.
    """
    stage_progress = _resolve_progress_callback(show_progress, progress_factory, on_progress)
    preprocess = stage_progress("preprocessing", 4) if stage_progress is not None else None

    f = np.asarray(movie, dtype=np.float32)
    _, ny, nx = f.shape
    original_shape = (ny, nx)
    padded = False
    bin_px = grid.bin_px
    use_padding = bool(allow_bin_padding or options.allow_bin_padding)
    if use_padding and (ny % bin_px or nx % bin_px):
        ny_pad, nx_pad, pad_y, pad_x = compute_bin_aligned_padding(ny, nx, bin_px)
        f = np.pad(f, ((0, 0), (0, pad_y), (0, pad_x)), mode=options.padding_mode)
        padded = True
        warnings.warn(
            f"Input movie shape ({ny},{nx}) was padded to ({ny_pad},{nx_pad}) to satisfy "
            f"bin_px divisibility using mode={options.padding_mode!r}; edge padding can affect "
            "correlation/velocity estimates near image borders.",
            UserWarning,
            stacklevel=2,
        )

    _, ny, nx = f.shape
    padded_shape = (ny, nx)
    bin_px, by, bx = validate_grid(ny, nx, grid.patch_px, grid.grid_stride_patches)

    region_raw, _, _ = block_mean_timeseries(f, bin_px=bin_px)
    region_dt = detrend_series(region_raw, axis=1, detrend_type=detrend_type)
    if preprocess is not None:
        preprocess.advance()

    g = build_background_regressors(f, bg_boxes_px, detrend_type=detrend_type)
    region_res = regress_out_common_mode_2d(region_dt, g)
    if preprocess is not None:
        preprocess.advance()

    if use_shear_mask:
        target_gate_vec = build_shear_mask_patchvec(
            f, by=by, bx=bx, bin_px=bin_px, shear_mask_px=shear_mask_px, shear_pctl=shear_pctl
        )
    else:
        target_gate_vec = np.ones(by * bx, dtype=bool)

    if seed_mask_px is not None:
        seed_gate_vec = build_shear_mask_patchvec(
            f, by=by, bx=bx, bin_px=bin_px, shear_mask_px=seed_mask_px, shear_pctl=shear_pctl
        )
    else:
        seed_gate_vec = target_gate_vec
    if preprocess is not None:
        preprocess.advance()

    x_m, y_m = patch_centers_xy_m(ny, nx, by, bx, bin_px, extent_xd_yd, dj_mm, origin=origin)
    if preprocess is not None:
        preprocess.advance()
        preprocess.close()

    shifts = tuple(sorted(set(int(s) for s in shifts if int(s) > 0)))
    if not shifts:
        raise ValueError("shifts must include at least one positive integer")

    use_sparse_corr_storage = (
        options.sparse_corr_storage if sparse_corr_storage is None else bool(sparse_corr_storage)
    )
    sparse_top_k_value = options.sparse_top_k if sparse_top_k is None else sparse_top_k
    if sparse_top_k_value is not None and int(sparse_top_k_value) <= 0:
        raise ValueError("sparse_top_k must be > 0 when provided")

    n_regions = by * bx
    row_coords = np.repeat(np.arange(by, dtype=np.float64), bx)
    col_coords = np.tile(np.arange(bx, dtype=np.float64), by)
    ux = np.full(n_regions, np.nan, dtype=np.float32)
    uy = np.full(n_regions, np.nan, dtype=np.float32)
    used_count = np.zeros(n_regions, dtype=np.int32)
    edge_clipped_by_shift = {s: np.zeros(n_regions, dtype=bool) for s in shifts} if options.edge_clip_reject_k is not None else None

    seeds = np.flatnonzero(seed_gate_vec)
    seed_progress = stage_progress("seed loop progress", int(seeds.size)) if stage_progress is not None else None
    valid = 0

    if not use_sparse_corr_storage:
        corr_by_shift: dict[int, np.ndarray | dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        shift_progress = stage_progress("per-shift processing", len(shifts)) if stage_progress is not None else None
        for s in shifts:
            corr_by_shift[s] = corr_matrix_positive_shift(region_res, s)
            if shift_progress is not None:
                shift_progress.advance()
        if shift_progress is not None:
            shift_progress.close()

        for j in seeds:
            seed_edge_diag: dict[int, bool] = {}
            ux_j, uy_j, n_any = _fit_seed_from_corr_vectors(
                seed_idx=int(j),
                target_gate_vec=target_gate_vec,
                x_m=x_m,
                y_m=y_m,
                row_coords=row_coords,
                col_coords=col_coords,
                by=by,
                bx=bx,
                shifts=shifts,
                fs=fs,
                options=options,
                corr_for_shift=lambda s, _j=int(j): np.asarray(corr_by_shift[s])[:, _j],
                edge_clipped_by_shift=seed_edge_diag if edge_clipped_by_shift is not None else None,
            )
            used_count[j] = n_any
            if edge_clipped_by_shift is not None:
                for s in shifts:
                    edge_clipped_by_shift[s][j] = bool(seed_edge_diag.get(int(s), False))
            ux[j] = ux_j
            uy[j] = uy_j
            if np.isfinite(ux_j) and np.isfinite(uy_j):
                valid += 1
            if seed_progress is not None:
                seed_progress.advance()
    else:
        corr_by_shift = {s: {} for s in shifts}
        shift_stats: dict[int, dict[str, np.ndarray | float]] = {}
        for s in shifts:
            n1 = region_res.shape[1] - int(s)
            if n1 < 3:
                continue
            target_slice = region_res[:, s:].astype(np.float64, copy=False)
            seed_slice = region_res[:, :-s].astype(np.float64, copy=False)
            shift_stats[s] = {
                "target_row_means": target_slice.mean(axis=1),
                "target_row_stds": target_slice.std(axis=1, ddof=1) + 1e-12,
                "seed_row_means": seed_slice.mean(axis=1),
                "seed_row_stds": seed_slice.std(axis=1, ddof=1) + 1e-12,
                "norm_denom": float(n1 - 1),
            }

        for j in seeds:
            gate = target_gate_vec.copy()
            gate[int(j)] = False
            dx_all = x_m - x_m[int(j)]
            if options.require_downstream:
                gate &= dx_all > 0

            sparse_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

            for s in shifts:
                stats = shift_stats.get(s)
                if stats is None:
                    corr_row = corr_targets_for_seed_positive_shift(region_res, int(j), int(s))
                else:
                    corr_row = corr_targets_for_seed_positive_shift(
                        region_res,
                        int(j),
                        int(s),
                        target_row_means=stats["target_row_means"],
                        target_row_stds=stats["target_row_stds"],
                        seed_row_means=stats["seed_row_means"],
                        seed_row_stds=stats["seed_row_stds"],
                        norm_denom=stats["norm_denom"],
                    )
                sparse_cache[s] = sparse_useful_corr_row(
                    corr_row,
                    candidate_mask=gate,
                    rmin=options.rmin,
                    top_k=sparse_top_k_value,
                )
                corr_by_shift[s][int(j)] = sparse_cache[s]

            seed_edge_diag: dict[int, bool] = {}
            ux_j, uy_j, n_any = _fit_seed_from_sparse_vectors(
                seed_idx=int(j),
                x_m=x_m,
                y_m=y_m,
                row_coords=row_coords,
                col_coords=col_coords,
                by=by,
                bx=bx,
                shifts=shifts,
                fs=fs,
                options=options,
                sparse_corr_for_shift=lambda s: sparse_cache[s],
                edge_clipped_by_shift=seed_edge_diag if edge_clipped_by_shift is not None else None,
            )
            used_count[j] = n_any
            if edge_clipped_by_shift is not None:
                for s in shifts:
                    edge_clipped_by_shift[s][j] = bool(seed_edge_diag.get(int(s), False))
            ux[j] = ux_j
            uy[j] = uy_j
            if np.isfinite(ux_j) and np.isfinite(uy_j):
                valid += 1
            if seed_progress is not None:
                seed_progress.advance()

    if seed_progress is not None:
        seed_progress.close()

    return VelocityMapResult(
        ux_map=ux.reshape(by, bx),
        uy_map=uy.reshape(by, bx),
        used_count_map=used_count.reshape(by, bx),
        shear_mask_vec=target_gate_vec,
        x_m=x_m,
        y_m=y_m,
        by=by,
        bx=bx,
        corr_by_shift=corr_by_shift,
        valid_seed_count=valid,
        total_seed_count=int(seeds.size),
        padded=padded,
        original_shape=original_shape,
        padded_shape=padded_shape,
        edge_clipped_by_shift=edge_clipped_by_shift,
    )
