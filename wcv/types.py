from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

SparseCorrRow = tuple[np.ndarray, np.ndarray]
CorrByShift = Dict[int, np.ndarray | Dict[int, SparseCorrRow]]


@dataclass(frozen=True)
class GridSpec:
    """Grid configuration.

    PATCH_PX is the base patch size that must tile the image.
    BIN_PX = PATCH_PX * grid_stride_patches is the effective cell size used
    for correlation and centroid calculations ("superbinning").
    """

    patch_px: int
    grid_stride_patches: int = 1

    @property
    def bin_px(self) -> int:
        return int(self.patch_px) * int(self.grid_stride_patches)


@dataclass(frozen=True)
class EstimationOptions:
    """Shared estimator tuning options.

    Note:
        ``allow_bin_padding`` is deprecated. Prefer passing
        ``allow_bin_padding`` directly to estimator functions.
    """
    rmin: float = 0.2
    min_used: int = 10
    require_downstream: bool = True
    require_dy_positive: bool = False
    weight_power: float = 2.0
    allow_bin_padding: bool = False
    padding_mode: str = "edge"
    sparse_corr_storage: bool = False
    sparse_top_k: int | None = None
    edge_clip_reject_k: float | None = None
    edge_clip_sigma_mult: float = 2.0


@dataclass
class SingleSeedResult:
    ux: float
    uy: float
    shifts: Sequence[int]
    shifts_used_for_fit: Sequence[int]
    dx_bar_by_shift: Dict[int, float]
    dy_bar_by_shift: Dict[int, float]
    n_used_by_shift: Dict[int, int]
    corr_by_shift: Dict[int, np.ndarray]
    mask_by_shift: Dict[int, np.ndarray]
    dx_m: np.ndarray
    dy_m: np.ndarray
    by: int
    bx: int
    padded: bool = False
    original_shape: tuple[int, int] | None = None
    padded_shape: tuple[int, int] | None = None
    edge_clipped_by_shift: Dict[int, bool] | None = None
    edge_distance_by_shift: Dict[int, float] | None = None
    support_radius_by_shift: Dict[int, float] | None = None


@dataclass
class VelocityMapResult:
    ux_map: np.ndarray
    uy_map: np.ndarray
    used_count_map: np.ndarray
    shear_mask_vec: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    by: int
    bx: int
    corr_by_shift: CorrByShift
    valid_seed_count: int
    total_seed_count: int
    padded: bool = False
    original_shape: tuple[int, int] | None = None
    padded_shape: tuple[int, int] | None = None
    edge_clipped_by_shift: Dict[int, np.ndarray] | None = None
