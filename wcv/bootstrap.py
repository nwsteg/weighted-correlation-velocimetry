from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import sys
import warnings

import numpy as np

from .estimator_map import estimate_velocity_map_streaming
from .types import VelocityMapResult


EstimatorFn = Callable[..., VelocityMapResult]


@dataclass(frozen=True)
class BootstrapSummary:
    mean: np.ndarray
    std: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    valid_fraction: np.ndarray
    valid_count: np.ndarray


@dataclass(frozen=True)
class VelocityBootstrapResult:
    ux: BootstrapSummary
    uy: BootstrapSummary
    um: BootstrapSummary
    n_bootstrap: int
    block_length: int
    ci_percentiles: tuple[float, float]
    sample_shape: tuple[int, int]
    samples_ux: np.ndarray | None = None
    samples_uy: np.ndarray | None = None
    samples_um: np.ndarray | None = None


def default_block_length(n_frames: int) -> int:
    """Heuristic moving-block length for N in roughly [100, 500].

    Uses ``round(sqrt(N))`` with a floor of 8 frames to avoid overly short
    blocks that would inflate boundary artifacts and weaken temporal
    dependence preservation.
    """
    n = int(n_frames)
    if n <= 0:
        raise ValueError("n_frames must be > 0")
    return min(n, max(8, int(round(np.sqrt(n)))))


def moving_block_bootstrap_indices(
    n_frames: int,
    block_length: int,
    rng: np.random.Generator,
    *,
    circular: bool = False,
) -> np.ndarray:
    """Sample contiguous time blocks with replacement and trim to length N."""
    n = int(n_frames)
    l = int(block_length)
    if n <= 1:
        raise ValueError("n_frames must be > 1")
    if l <= 0:
        raise ValueError("block_length must be > 0")
    l = min(l, n)

    pieces: list[np.ndarray] = []
    remaining = n
    while remaining > 0:
        if circular:
            start = int(rng.integers(0, n))
            block = (start + np.arange(l, dtype=np.int64)) % n
        else:
            max_start = n - l
            start = int(rng.integers(0, max_start + 1))
            block = np.arange(start, start + l, dtype=np.int64)
        pieces.append(block)
        remaining -= l

    idx = np.concatenate(pieces)
    return idx[:n]


def _summary_from_samples(
    samples: np.ndarray,
    ci_percentiles: tuple[float, float],
    min_valid_fraction: float,
) -> BootstrapSummary:
    finite = np.isfinite(samples)
    valid_count = finite.sum(axis=0).astype(np.int32)
    valid_fraction = valid_count.astype(np.float32) / float(samples.shape[0])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(samples, axis=0)
            std = np.nanstd(samples, axis=0, ddof=1)
            ci_low = np.nanpercentile(samples, ci_percentiles[0], axis=0)
            ci_high = np.nanpercentile(samples, ci_percentiles[1], axis=0)

    bad = valid_fraction < float(min_valid_fraction)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    ci_low = ci_low.astype(np.float32)
    ci_high = ci_high.astype(np.float32)
    mean[bad] = np.nan
    std[bad] = np.nan
    ci_low[bad] = np.nan
    ci_high[bad] = np.nan

    return BootstrapSummary(
        mean=mean,
        std=std,
        ci_low=ci_low,
        ci_high=ci_high,
        valid_fraction=valid_fraction,
        valid_count=valid_count,
    )


def bootstrap_velocity_map(
    movie: np.ndarray,
    n_bootstrap: int,
    *,
    estimator: EstimatorFn = estimate_velocity_map_streaming,
    estimator_kwargs: dict | None = None,
    block_length: int | None = None,
    seed: int | None = None,
    circular: bool = False,
    ci_percentiles: tuple[float, float] = (2.5, 97.5),
    min_valid_fraction: float = 0.2,
    store_samples: bool = False,
    show_progress: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> VelocityBootstrapResult:
    """Estimate WCV uncertainty from one temporally-correlated movie.

    For each replicate, frames are resampled with moving blocks and passed into
    the supplied estimator unchanged as ``movie=movie[idx]``.
    """
    est_kwargs = dict(estimator_kwargs or {})
    movie_arr = np.asarray(movie)
    if movie_arr.ndim != 3:
        raise ValueError("movie must be a 3D array (time, y, x)")

    n_frames = int(movie_arr.shape[0])
    if n_frames < 2:
        raise ValueError("movie must include at least 2 frames")
    if int(n_bootstrap) <= 0:
        raise ValueError("n_bootstrap must be > 0")
    if not (0.0 <= float(min_valid_fraction) <= 1.0):
        raise ValueError("min_valid_fraction must be in [0, 1]")

    lo, hi = float(ci_percentiles[0]), float(ci_percentiles[1])
    if not (0.0 <= lo < hi <= 100.0):
        raise ValueError("ci_percentiles must satisfy 0 <= low < high <= 100")

    blen = default_block_length(n_frames) if block_length is None else int(block_length)
    if blen <= 0:
        raise ValueError("block_length must be > 0")

    rng = np.random.default_rng(seed)
    total_bootstrap = int(n_bootstrap)

    tqdm_bar = None
    use_text_fallback = False
    if progress_callback is None and show_progress:
        try:
            from tqdm.auto import tqdm

            tqdm_bar = tqdm(total=total_bootstrap, desc="Bootstrap", unit="rep")
        except Exception:
            use_text_fallback = True

    sample_ux: list[np.ndarray] = []
    sample_uy: list[np.ndarray] = []
    sample_um: list[np.ndarray] = []
    try:
        for done in range(1, total_bootstrap + 1):
            idx = moving_block_bootstrap_indices(n_frames, blen, rng, circular=circular)
            res = estimator(movie=movie_arr[idx], **est_kwargs)
            ux = np.asarray(res.ux_map, dtype=np.float32)
            uy = np.asarray(res.uy_map, dtype=np.float32)
            um = np.sqrt(ux * ux + uy * uy).astype(np.float32)
            sample_ux.append(ux)
            sample_uy.append(uy)
            sample_um.append(um)

            if progress_callback is not None:
                progress_callback(done, total_bootstrap)
            elif tqdm_bar is not None:
                try:
                    tqdm_bar.update(1)
                except Exception:
                    use_text_fallback = True
                    tqdm_bar.close()
                    tqdm_bar = None
            if use_text_fallback:
                print(f"Bootstrap {done}/{total_bootstrap}", file=sys.stderr)
    finally:
        if tqdm_bar is not None:
            tqdm_bar.close()

    ux_stack = np.stack(sample_ux, axis=0)
    uy_stack = np.stack(sample_uy, axis=0)
    um_stack = np.stack(sample_um, axis=0)

    return VelocityBootstrapResult(
        ux=_summary_from_samples(ux_stack, ci_percentiles, min_valid_fraction),
        uy=_summary_from_samples(uy_stack, ci_percentiles, min_valid_fraction),
        um=_summary_from_samples(um_stack, ci_percentiles, min_valid_fraction),
        n_bootstrap=total_bootstrap,
        block_length=int(blen),
        ci_percentiles=(lo, hi),
        sample_shape=ux_stack.shape[1:],
        samples_ux=ux_stack if store_samples else None,
        samples_uy=uy_stack if store_samples else None,
        samples_um=um_stack if store_samples else None,
    )
