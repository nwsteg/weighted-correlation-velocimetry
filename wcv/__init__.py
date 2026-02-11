"""Weighted-correlation velocimetry package.

Superbinning convention
----------------------
- ``PATCH_PX``: base patch size that must tile the original image.
- ``grid_stride_patches``: number of base patches grouped per side.
- ``BIN_PX = PATCH_PX * grid_stride_patches``: effective region size used by
  correlation, centroid geometry, masks, and seed iteration.

This allows larger analysis cells without cropping and without changing the
original imshow extent mapping.
"""

from .estimator_map import estimate_velocity_map
from .estimator_single_seed import estimate_single_seed_velocity
from .types import EstimationOptions, GridSpec, SingleSeedResult, VelocityMapResult

__all__ = [
    "estimate_single_seed_velocity",
    "estimate_velocity_map",
    "GridSpec",
    "EstimationOptions",
    "SingleSeedResult",
    "VelocityMapResult",
]
