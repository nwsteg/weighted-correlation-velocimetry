from __future__ import annotations

from collections.abc import Callable
import warnings

import numpy as np

from .correlation import (
    corr_matrix_positive_shift,
    corr_targets_for_seed_positive_shift,
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


def _fit_seed_from_corr_vectors(
    *,
    seed_idx: int,
    shear: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    shifts: tuple[int, ...],
    fs: float,
    options: EstimationOptions,
    corr_for_shift: Callable[[int], np.ndarray],
) -> tuple[float, float, int]:
    dx_all = x_m - x_m[seed_idx]
    dy_all = y_m - y_m[seed_idx]

    gate = shear.copy()
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
    shear_pctl: float = 50,
    origin: str = "upper",
    detrend_type: str = "linear",
    store_corr_by_shift: bool = False,
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
        shear = build_shear_mask_patchvec(
            f, by=by, bx=bx, bin_px=bin_px, shear_mask_px=shear_mask_px, shear_pctl=shear_pctl
        )
    else:
        shear = np.ones(by * bx, dtype=bool)
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
    ux = np.full(n_regions, np.nan, dtype=np.float32)
    uy = np.full(n_regions, np.nan, dtype=np.float32)
    used_count = np.zeros(n_regions, dtype=np.int32)

    corr_by_shift = (
        {s: np.full((n_regions, n_regions), np.nan, dtype=np.float32) for s in shifts}
        if store_corr_by_shift
        else {}
    )

    seeds = np.flatnonzero(shear)
    seed_progress = stage_progress("seed loop progress", int(seeds.size)) if stage_progress is not None else None
    valid = 0
    for j in seeds:
        corr_cache: dict[int, np.ndarray] = {}

        def _corr_for_shift(s: int) -> np.ndarray:
            r = corr_cache.get(s)
            if r is None:
                r = corr_targets_for_seed_positive_shift(region_res, int(j), int(s))
                corr_cache[s] = r
                if store_corr_by_shift:
                    corr_by_shift[s][:, j] = r
            return r

        ux_j, uy_j, n_any = _fit_seed_from_corr_vectors(
            seed_idx=int(j),
            shear=shear,
            x_m=x_m,
            y_m=y_m,
            shifts=shifts,
            fs=fs,
            options=options,
            corr_for_shift=_corr_for_shift,
        )
        used_count[j] = n_any
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
        shear_mask_vec=shear,
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
    )

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
    shear_pctl: float = 50,
    origin: str = "upper",
    detrend_type: str = "linear",
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
        shear = build_shear_mask_patchvec(
            f, by=by, bx=bx, bin_px=bin_px, shear_mask_px=shear_mask_px, shear_pctl=shear_pctl
        )
    else:
        shear = np.ones(by * bx, dtype=bool)
    if preprocess is not None:
        preprocess.advance()

    x_m, y_m = patch_centers_xy_m(ny, nx, by, bx, bin_px, extent_xd_yd, dj_mm, origin=origin)
    if preprocess is not None:
        preprocess.advance()
        preprocess.close()

    shifts = tuple(sorted(set(int(s) for s in shifts if int(s) > 0)))
    if not shifts:
        raise ValueError("shifts must include at least one positive integer")

    corr_by_shift: dict[int, np.ndarray] = {}
    shift_progress = stage_progress("per-shift processing", len(shifts)) if stage_progress is not None else None
    for s in shifts:
        corr_by_shift[s] = corr_matrix_positive_shift(region_res, s)
        if shift_progress is not None:
            shift_progress.advance()
    if shift_progress is not None:
        shift_progress.close()

    n_regions = by * bx
    ux = np.full(n_regions, np.nan, dtype=np.float32)
    uy = np.full(n_regions, np.nan, dtype=np.float32)
    used_count = np.zeros(n_regions, dtype=np.int32)

    seeds = np.flatnonzero(shear)
    seed_progress = stage_progress("seed loop progress", int(seeds.size)) if stage_progress is not None else None
    valid = 0
    for j in seeds:
        ux_j, uy_j, n_any = _fit_seed_from_corr_vectors(
            seed_idx=int(j),
            shear=shear,
            x_m=x_m,
            y_m=y_m,
            shifts=shifts,
            fs=fs,
            options=options,
            corr_for_shift=lambda s, _j=int(j): corr_by_shift[s][:, _j],
        )
        used_count[j] = n_any
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
        shear_mask_vec=shear,
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
    )
