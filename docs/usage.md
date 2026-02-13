# Usage

## Installation

```bash
pip install -e .
```

## Expected inputs

Both modern estimators use:

- `movie`: `np.ndarray` of shape `(nt, ny, nx)` (float recommended)
- `fs`: frame rate in Hz
- `grid`: `GridSpec(patch_px=..., grid_stride_patches=...)`
- `bg_boxes_px`: list of `(y0, y1, x0, x1)` background boxes
- `extent_xd_yd`: `(xmin, xmax, ymin, ymax)` in normalized coordinates (typically `x/D, y/D`)
- `dj_mm`: physical scale used to convert normalized coordinates to meters
- `shifts`: tuple of positive integer lags in frames
- `options`: `EstimationOptions`

Single-seed estimator additionally needs:

- `seed_box_px`: one-bin seed box `(y0, y1, x0, x1)` where width/height are exactly `bin_px`

## Outputs

### `estimate_single_seed_velocity`

Returns `SingleSeedResult` with:

- velocities: `ux`, `uy` (m/s)
- diagnostics by lag: `corr_by_shift`, `mask_by_shift`, `dx_bar_by_shift`, `dy_bar_by_shift`, `n_used_by_shift`
- edge-clip diagnostics by lag: `edge_clipped_by_shift`, `edge_distance_by_shift`, `support_radius_by_shift`
- grid shape: `by`, `bx`
- padding metadata: `padded`, `original_shape`, `padded_shape`

### `estimate_velocity_map`

Returns `VelocityMapResult` with:

- maps: `ux_map`, `uy_map`
- quality/support: `used_count_map`, `valid_seed_count`, `total_seed_count`
- target gating mask (used for candidate correlations): `shear_mask_vec`
- center coordinates: `x_m`, `y_m`
- padding metadata: `padded`, `original_shape`, `padded_shape`
- optional edge-clip diagnostics map by lag: `edge_clipped_by_shift`

### Edge-clip quality filter

`EstimationOptions` now supports an optional centroid-support clipping guard:

- `edge_clip_reject_k`: disabled when `None`; when set, rejects a lag fit if
  `edge_distance < edge_clip_reject_k * support_radius`.
- `edge_clip_sigma_mult`: multiplier used to convert weighted spread to
  `support_radius` (default `2.0`, i.e., ~2σ).

This filter is applied per seed and per shift before velocity fitting.


### Seed mask vs target mask

Map estimators now support two optional mask inputs with different roles:

- `seed_mask_px`: controls **which patches are allowed to act as seeds** (which bins get a velocity estimate).
- `shear_mask_px` (with `use_shear_mask=True`): controls **which patches are allowed as correlation targets/candidates** during fitting.

Backward compatibility behavior:

- If `seed_mask_px` is omitted (`None`), seed selection follows the same logic as before (uses the shear/target mask).
- If `use_shear_mask=False`, target gating defaults to all patches; with `seed_mask_px=None`, seed selection also defaults to all patches.

```python
vm = estimate_velocity_map(
    movie=movie,
    fs=50_000,
    grid=grid,
    bg_boxes_px=bg_boxes_px,
    extent_xd_yd=extent,
    dj_mm=Dj,
    shifts=(1, 2),
    options=opts,
    use_shear_mask=True,
    shear_mask_px=target_mask_px,   # who can be matched against each seed
    seed_mask_px=seed_only_mask_px, # which bins become seeds
)
```

## Minimal example

```python
import numpy as np
from wcv import GridSpec, EstimationOptions, estimate_single_seed_velocity, estimate_velocity_map
from wcv.geometry import frac_to_px_box

# movie: shape (nt, ny, nx)
# extent: (xmin, xmax, ymin, ymax) in x/D, y/D
# Dj: nozzle diameter in mm (or your physical reference length)

grid = GridSpec(patch_px=4, grid_stride_patches=4)  # bin_px = 16
opts = EstimationOptions(
    rmin=0.3,
    min_used=10,
    require_downstream=True,
    require_dy_positive=False,
    weight_power=2.0,
    allow_bin_padding=True,  # optional; warns when applied
)

nt, ny, nx = movie.shape
bin_px = grid.bin_px

seed_box_px = frac_to_px_box(
    px=0.08, py=0.67,
    w_px=bin_px, h_px=bin_px,
    nx=nx, ny=ny,
    grid_px=bin_px,
)

bg_boxes_px = [
    frac_to_px_box(px=0.10, py=0.10, w_px=3*bin_px, h_px=3*bin_px, nx=nx, ny=ny, grid_px=bin_px),
    frac_to_px_box(px=0.25, py=0.96, w_px=3*bin_px, h_px=3*bin_px, nx=nx, ny=ny, grid_px=bin_px),
]

res = estimate_single_seed_velocity(
    movie=movie,
    fs=50_000,
    grid=grid,
    seed_box_px=seed_box_px,
    bg_boxes_px=bg_boxes_px,
    extent_xd_yd=extent,
    dj_mm=Dj,
    shifts=(1, 2),
    options=opts,
    origin="upper",
)
print("single-seed:", res.ux, res.uy, "padded:", res.padded)

vm = estimate_velocity_map(
    movie=movie,
    fs=50_000,
    grid=grid,
    bg_boxes_px=bg_boxes_px,
    extent_xd_yd=extent,
    dj_mm=Dj,
    shifts=(1, 2),
    options=opts,
    use_shear_mask=False,
    origin="upper",
)
print("map valid seeds:", vm.valid_seed_count, "/", vm.total_seed_count)
```



## Progress hooks (notebook-friendly)

Both map estimators support optional progress hooks:

- `show_progress=True`: uses `tqdm.auto.tqdm` when available
- `progress_factory(total, stage)`: custom per-stage progress sink
- `on_progress(done, total, stage)`: direct callback

```python
from wcv import estimate_velocity_map, estimate_velocity_map_streaming

# Notebook/CLI auto progress bars (falls back to no-op if tqdm is unavailable)
vm = estimate_velocity_map(
    movie=movie,
    fs=50_000,
    grid=grid,
    bg_boxes_px=bg_boxes_px,
    extent_xd_yd=extent,
    dj_mm=Dj,
    shifts=(1, 2),
    options=opts,
    show_progress=True,
)

# Custom callback for logging stage-level updates
updates = []
_ = estimate_velocity_map_streaming(
    movie=movie,
    fs=50_000,
    grid=grid,
    bg_boxes_px=bg_boxes_px,
    extent_xd_yd=extent,
    dj_mm=Dj,
    shifts=(1, 2),
    options=opts,
    on_progress=lambda done, total, stage: updates.append((stage, done, total)),
)
```

Typical `stage` values include `preprocessing`, `per-shift processing` (materialized map estimator),
and `seed loop progress`.

## Building docs locally

```bash
pip install -r requirements-docs.txt
mkdocs serve
mkdocs build
```
