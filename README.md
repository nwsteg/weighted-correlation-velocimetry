# weighted-correlation-velocimetry

Refactored Python package for convection-velocity estimation from high-speed PLIF movies using patch-wise, time-lagged correlations and weighted-centroid displacement fitting.

## Package layout

- `wcv/preprocess.py`: detrending, z-scoring, common-mode regression, block means.
- `wcv/geometry.py`: extent conversions, patch/superpatch geometry, mask downsampling, box snapping.
- `wcv/correlation.py`: shifted correlations and full correlation matrices per positive shift.
- `wcv/estimator_single_seed.py`: single-seed estimator with per-shift diagnostics.
- `wcv/estimator_map.py`: velocity-map estimator over all seeds on the analysis grid.
- `wcv/plotting.py`: reusable plotting helpers.
- `examples/workflow_example.py`: minimal runnable workflow script.

## Superbinning convention

The package distinguishes two spatial scales:

1. `PATCH_PX` (base patch size): must tile the original image exactly.
2. `BIN_PX = PATCH_PX * grid_stride_patches` (effective analysis cell size): used everywhere that defines correlation-grid regions (block means, shear mask downsampling, geometry centers, and seed iteration).

This preserves the original image extents and avoids manual recropping/relabeling while allowing larger analysis bins.

## Quick start

```python
from wcv import GridSpec, EstimationOptions, estimate_single_seed_velocity

grid = GridSpec(patch_px=4, grid_stride_patches=4)  # BIN_PX=16
opts = EstimationOptions(rmin=0.2, min_used=10)

result = estimate_single_seed_velocity(
    movie=bgs_f,
    fs=15000,
    grid=grid,
    seed_box_px=seed_box,
    bg_boxes_px=bg_boxes,
    extent_xd_yd=(-1, 7, -2.5, 2.5),
    dj_mm=10.0,
    shifts=(1, 2, 3),
    options=opts,
)
print(result.ux, result.uy)
```
