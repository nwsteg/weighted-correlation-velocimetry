from __future__ import annotations

import numpy as np


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

    b0 = b - b.mean()
    bstd = b0.std(ddof=1) + 1e-12
    a0 = a - a.mean(axis=1, keepdims=True)
    astd = a0.std(axis=1, ddof=1) + 1e-12
    cov = (a0 @ b0) / (n - 1)
    return (cov / (astd * bstd)).astype(np.float32, copy=False)


def corr_matrix_positive_shift(region_res: np.ndarray, shift: int) -> np.ndarray:
    """R[target, seed] = corr(region[target,s:], region[seed,:-s]) for positive shift."""
    s = int(shift)
    if s <= 0:
        raise ValueError("shift must be > 0")
    n, nt = region_res.shape
    n1 = nt - s
    if n1 < 3:
        return np.full((n, n), np.nan, dtype=np.float32)

    a = region_res[:, s:].astype(np.float64, copy=False)
    b = region_res[:, :-s].astype(np.float64, copy=False)

    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    za = a0 / (a0.std(axis=1, ddof=1, keepdims=True) + 1e-12)
    zb = b0 / (b0.std(axis=1, ddof=1, keepdims=True) + 1e-12)
    return ((za @ zb.T) / float(n1 - 1)).astype(np.float32, copy=False)


def corr_targets_for_seed_positive_shift(
    region_res: np.ndarray, seed_idx: int, shift: int
) -> np.ndarray:
    """corr[target] = corr(region[target,s:], region[seed,:-s]) for positive shift."""
    s = int(shift)
    if s <= 0:
        raise ValueError("shift must be > 0")
    n_regions, nt = region_res.shape
    if seed_idx < 0 or seed_idx >= n_regions:
        raise IndexError("seed_idx out of bounds")

    n1 = nt - s
    if n1 < 3:
        return np.full(n_regions, np.nan, dtype=np.float32)

    a = region_res[:, s:].astype(np.float64, copy=False)
    b = region_res[seed_idx, :-s].astype(np.float64, copy=False)

    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean()
    za = a0 / (a0.std(axis=1, ddof=1, keepdims=True) + 1e-12)
    zb = b0 / (b0.std(ddof=1) + 1e-12)
    return ((za @ zb) / float(n1 - 1)).astype(np.float32, copy=False)


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

    a = region_res[:, s:].astype(np.float64, copy=False)
    b = region_res[seed_indices, :-s].astype(np.float64, copy=False)

    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    za = a0 / (a0.std(axis=1, ddof=1, keepdims=True) + 1e-12)
    zb = b0 / (b0.std(axis=1, ddof=1, keepdims=True) + 1e-12)
    return ((za @ zb.T) / float(n1 - 1)).astype(np.float32, copy=False)
