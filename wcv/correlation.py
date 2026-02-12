from __future__ import annotations

import numpy as np


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

    if target_row_means is None:
        a_means = a.mean(axis=1)
    else:
        a_means = np.asarray(target_row_means, dtype=np.float64)

    if target_row_stds is None:
        a_stds = a.std(axis=1, ddof=1) + 1e-12
    else:
        a_stds = np.asarray(target_row_stds, dtype=np.float64)

    if seed_row_means is None:
        b_mean = float(b.mean())
    else:
        b_mean = float(np.asarray(seed_row_means, dtype=np.float64)[seed_idx])

    if seed_row_stds is None:
        b_std = float(b.std(ddof=1) + 1e-12)
    else:
        b_std = float(np.asarray(seed_row_stds, dtype=np.float64)[seed_idx])

    denom = float(norm_denom) if norm_denom is not None else float(n1 - 1)

    a0 = a - a_means[:, None]
    b0 = b - b_mean
    za = a0 / a_stds[:, None]
    zb = b0 / b_std
    return ((za @ zb) / denom).astype(np.float32, copy=False)


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
        return np.full((seed_indices.size, n_regions), np.nan, dtype=np.float32)

    a = region_res[:, s:].astype(np.float64, copy=False)
    b = region_res[seed_indices, :-s].astype(np.float64, copy=False)

    if target_row_means is None:
        a_means = a.mean(axis=1)
    else:
        a_means = np.asarray(target_row_means, dtype=np.float64)

    if target_row_stds is None:
        a_stds = a.std(axis=1, ddof=1) + 1e-12
    else:
        a_stds = np.asarray(target_row_stds, dtype=np.float64)

    if seed_row_means is None:
        b_means = b.mean(axis=1)
    else:
        b_means = np.asarray(seed_row_means, dtype=np.float64)[seed_indices]

    if seed_row_stds is None:
        b_stds = b.std(axis=1, ddof=1) + 1e-12
    else:
        b_stds = np.asarray(seed_row_stds, dtype=np.float64)[seed_indices]

    denom = float(norm_denom) if norm_denom is not None else float(n1 - 1)

    za = (a - a_means[:, None]) / a_stds[:, None]
    zb = (b - b_means[:, None]) / b_stds[:, None]
    return ((zb @ za.T) / denom).astype(np.float32, copy=False)
