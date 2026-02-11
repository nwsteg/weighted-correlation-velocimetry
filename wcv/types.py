from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


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
    rmin: float = 0.2
    min_used: int = 10
    require_downstream: bool = True
    require_dy_positive: bool = False
    weight_power: float = 2.0


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
    corr_by_shift: Dict[int, np.ndarray]
    valid_seed_count: int
    total_seed_count: int
