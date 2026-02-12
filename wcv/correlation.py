from __future__ import annotations

import logging
import os

import numpy as np


logger = logging.getLogger(__name__)
_DEBUG_DTYPES = os.getenv("WCV_DEBUG_DTYPES", "0") == "1"


def _debug_assert_float32(name: str, arr: np.ndarray) -> None:
    if not _DEBUG_DTYPES:
        return
    if arr.dtype != np.float32:
        raise AssertionError(f"{name} expected float32, got {arr.dtype}")
    logger.debug("%s dtype=%s shape=%s", name, arr.dtype, arr.shape)


def sparse_useful_corr_row(
    corr_row: np.ndarray,
    *,
    candidate_mask: np.ndarray,
    rmin: float,
    top_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select useful correlations from a dense target-vector into sparse form.

    Returns ``(indices, values)`` where ``indices`` are target-bin indices and
    ``values`` are the corresponding correlations.
    """
    r = np.asarray(corr_row, dtype=np.float32)
    keep = np.asarray(candidate_mask, dtype=bool) & np.isfinite(r) & (r > 0) & (r >= float(rmin))
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return idx.astype(np.int32), np.empty(0, dtype=np.float32)

    vals = r[idx]
    if top_k is not None:
        k = int(top_k)
        if k <= 0:
            raise ValueError("top_k must be > 0 when provided")
        if idx.size > k:
            order = np.argpartition(vals, -k)[-k:]
            idx = idx[order]
            vals = vals[order]

    order = np.argsort(idx)
    return idx[order].astype(np.int32, copy=False), vals[order].astype(np.float32, copy=False)


def shifted_corr_regions(region_res: np.ndarray, seed_res: np.ndarray, shift: int) -> np.ndarray:
    """Pearson correlation between seed(t) and region_i(t+shift)."""
    nt = seed_res.size
    s = int(shift)
    if s > 0:
        a = region_res[:, s:]
        b = seed_res[: nt - s]
    elif s < 0:
        s = -s
        a = region_res[:, : nt - s]
        b = seed_res[s:]
    else:
        a = region_res
        b = seed_res

    n = b.size
    if n < 3:
        return np.full(region_res.shape[0], np.nan, dtype=np.float32)

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    _debug_assert_float32("shifted_corr_regions.a", a)
    _debug_assert_float32("shifted_corr_regions.b", b)

    b0 = np.array(b, dtype=np.float32, copy=True)
    b0 -= b0.mean()
    bstd = b0.std(ddof=1) + np.float32(1e-12)
    a0 = np.array(a, dtype=np.float32, copy=True)
    a0 -= a0.mean(axis=1, keepdims=True)
    astd = a0.std(axis=1, ddof=1) + np.float32(1e-12)
    cov = (a0 @ b0) / (n - 1)
    return (cov / (astd * bstd)).astype(np.float32, copy=False)


def corr_matrix_positive_shift(region_res: np.ndarray, shift: int) -> np.ndarray:
    """R[target, seed] = corr(region[target,s:], region[seed,:-s]) for positive shift."""
    seed_indices = np.arange(region_res.shape[0], dtype=np.int64)
    return corr_targets_for_seed_chunk_positive_shift(region_res, seed_indices, shift)


def corr_targets_for_seed_positive_shift(
    region_res: np.ndarray,
    seed_idx: int,
    shift: int,
    *,
    target_row_means: np.ndarray | None = None,
    target_row_stds: np.ndarray | None = None,
    seed_row_means: np.ndarray | None = None,
    seed_row_stds: np.ndarray | None = None,
    norm_denom: float | None = None,
) -> np.ndarray:
    """corr[target] = corr(region[target,s:], region[seed,:-s]) for positive shift."""
    del target_row_means, target_row_stds, seed_row_means, seed_row_stds, norm_denom
    seed_indices = np.array([seed_idx], dtype=np.int64)
    return corr_targets_for_seed_chunk_positive_shift(region_res, seed_indices, shift)[:, 0]


def corr_targets_for_seed_chunk_positive_shift(
    region_res: np.ndarray, seed_indices: np.ndarray, shift: int
) -> np.ndarray:
    """corr[target, seed_chunk] for positive shift.

    Returns matrix with shape ``(n_regions, n_chunk)``.
    """
    s = int(shift)
    if s <= 0:
        raise ValueError("shift must be > 0")

    seed_indices = np.asarray(seed_indices, dtype=np.int64)
    n_regions, nt = region_res.shape
    if seed_indices.ndim != 1:
        raise ValueError("seed_indices must be a 1D array")
    if np.any(seed_indices < 0) or np.any(seed_indices >= n_regions):
        raise IndexError("seed_indices out of bounds")

    n1 = nt - s
    if n1 < 3:
        return np.full((n_regions, seed_indices.size), np.nan, dtype=np.float32)

    a = np.array(region_res[:, s:], dtype=np.float32, copy=True)
    b = np.array(region_res[seed_indices, :-s], dtype=np.float32, copy=True)
    _debug_assert_float32("corr_targets_for_seed_chunk_positive_shift.a", a)
    _debug_assert_float32("corr_targets_for_seed_chunk_positive_shift.b", b)

    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    a /= a.std(axis=1, ddof=1, keepdims=True) + np.float32(1e-12)
    b /= b.std(axis=1, ddof=1, keepdims=True) + np.float32(1e-12)
    denom = np.float32(n1 - 1)
    return ((a @ b.T) / denom).astype(np.float32, copy=False)


def corr_targets_for_seed_block_positive_shift(
    region_res: np.ndarray,
    seed_indices: np.ndarray,
    shift: int,
    *,
    target_row_means: np.ndarray | None = None,
    target_row_stds: np.ndarray | None = None,
    seed_row_means: np.ndarray | None = None,
    seed_row_stds: np.ndarray | None = None,
    norm_denom: float | None = None,
) -> np.ndarray:
    """corr[seed_block, target] for positive shift.

    Returns matrix with shape ``(k, n_regions)`` where ``k = len(seed_indices)``.
    """
    del target_row_means, target_row_stds, seed_row_means, seed_row_stds, norm_denom
    return corr_targets_for_seed_chunk_positive_shift(region_res, seed_indices, shift).T
