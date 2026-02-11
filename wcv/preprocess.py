from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.signal import detrend as scipy_detrend


def detrend_series(y: np.ndarray, axis: int = -1, detrend_type: str = "linear") -> np.ndarray:
    """Detrend a series or matrix in float32."""
    arr = np.asarray(y, dtype=np.float32)
    return scipy_detrend(arr, axis=axis, type=detrend_type).astype(np.float32, copy=False)


def zscore_1d(y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    centered = arr - arr.mean()
    return (centered / (centered.std(ddof=1) + eps)).astype(np.float32, copy=False)


def regress_out_common_mode_2d(y: np.ndarray, g: np.ndarray, dtype_out=np.float32) -> np.ndarray:
    """Regress out common-mode signals from each row of y using intercept + g columns."""
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)

    if y.ndim != 2:
        raise ValueError(f"y must be 2D (n_series, nt). Got {y.ndim}D")
    if g.ndim == 1:
        g = g[:, None]
    elif g.ndim != 2:
        raise ValueError(f"g must be 1D or 2D. Got {g.ndim}D")

    n_series, nt = y.shape
    if g.shape[0] != nt:
        raise ValueError(f"time mismatch: y.shape[1]={nt} vs g.shape[0]={g.shape[0]}")

    x = np.column_stack([np.ones(nt), g])
    b, *_ = np.linalg.lstsq(x, y.T, rcond=None)
    yhat = (x @ b).T
    return (y - yhat).astype(dtype_out, copy=False)


def block_mean_timeseries(movie: np.ndarray, bin_px: int) -> tuple[np.ndarray, int, int]:
    """Return block-mean patch series as (n_regions, nt) using effective BIN_PX."""
    f = np.asarray(movie, dtype=np.float32)
    if f.ndim != 3:
        raise ValueError(f"movie must be (nt, ny, nx). Got {f.shape}")
    nt, ny, nx = f.shape
    if ny % bin_px or nx % bin_px:
        raise ValueError(f"bin_px={bin_px} must divide movie shape ({ny},{nx}).")

    by = ny // bin_px
    bx = nx // bin_px
    blk = f.reshape(nt, by, bin_px, bx, bin_px).mean(axis=(2, 4))
    return blk.reshape(nt, by * bx).T.astype(np.float32, copy=False), by, bx


def build_background_regressors(
    movie: np.ndarray,
    bg_boxes_px: Iterable[tuple[int, int, int, int]],
    detrend_type: str = "linear",
) -> np.ndarray:
    """Build G(t) matrix with one detrended/z-scored column per background clump."""
    f = np.asarray(movie, dtype=np.float32)
    nt, ny, nx = f.shape
    cols = []
    for box in bg_boxes_px:
        y0, y1, x0, x1 = map(int, box)
        if not (0 <= y0 < y1 <= ny and 0 <= x0 < x1 <= nx):
            raise ValueError(f"background box out of bounds: {box}")
        g = f[:, y0:y1, x0:x1].mean(axis=(1, 2))
        cols.append(zscore_1d(detrend_series(g, detrend_type=detrend_type)))
    if not cols:
        raise ValueError("at least one background box is required")
    return np.column_stack(cols).astype(np.float32, copy=False)
