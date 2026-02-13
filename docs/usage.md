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

Interpretation details:

- accepted bins for a lag are those that pass the lag gating (`rmin`, sign, finite,
  directional constraints, `min_used`)
- weights are the same correlation-power weights used in centroid displacement,
  `w = |corr| ** weight_power`
- `support_radius` is computed from the weighted spread (second moments) of accepted bins
- `edge_distance` is the weighted-centroid distance to the nearest frame edge (in bin units)

Therefore, `support_radius` (and the circle shown in the interactive correlation view)
depends on both **which bins were accepted** and their **weights**.


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


## Time-series block bootstrap uncertainty

Use `bootstrap_velocity_map` to estimate uncertainty from a single temporally correlated movie.

- Default resampling uses a **moving-block bootstrap** (contiguous blocks sampled with replacement, concatenated, then trimmed to `N`).
- Default block length is `round(sqrt(N))` with a minimum of 8 frames (`default_block_length`). For `N=100..500`, this gives ~10..22 frame blocks so the lag-1 boundary fraction is about `1/L`.
- Each replicate runs the normal estimator on `movie[idx]` (no cross-replicate fit reuse).

```python
from wcv import bootstrap_velocity_map, estimate_velocity_map_streaming

bt = bootstrap_velocity_map(
    movie=movie,
    n_bootstrap=200,
    seed=123,
    estimator=estimate_velocity_map_streaming,
    estimator_kwargs=dict(
        fs=50_000,
        grid=grid,
        bg_boxes_px=bg_boxes_px,
        extent_xd_yd=extent,
        dj_mm=Dj,
        shifts=(1,),
        options=opts,
        use_shear_mask=False,
    ),
    show_progress=True,  # overall replicate progress
    # progress_callback=lambda done, total: print(done, total),
    # block_length=16,      # optional override
    # circular=True,        # optional circular block bootstrap
    ci_percentiles=(2.5, 97.5),
    min_valid_fraction=0.2,
)

ux_mean = bt.ux.mean
ux_std = bt.ux.std
ux_ci_lo = bt.ux.ci_low
ux_ci_hi = bt.ux.ci_high
ux_valid = bt.ux.valid_fraction
```

The same summaries are available for `uy` and speed magnitude `um`.

You can also provide `progress_callback(done, total)` to receive one update per bootstrap replicate.
When `show_progress=True`, WCV first tries a `tqdm` bar; if notebook widget rendering fails (for example `Error displaying widget: model not found`) or `tqdm` is unavailable, it falls back to a plain text `Bootstrap i/N` counter.

## Interpreting parameter-ensemble uncertainty outputs

`run_parameter_ensemble_uncertainty` reruns the estimator many times while perturbing key knobs
(`rmin`, `min_used`, `weight_power`, and `edge_clip_reject_k`) across user-defined ranges.
Each grid cell then gets statistics over that run stack.

### What `coverage` means

- WCV defines `coverage_map` as the fraction of ensemble runs where **both** `ux` and `uy`
  are finite at that cell.
- Mathematically: `coverage = mean(isfinite(ux_runs) & isfinite(uy_runs), axis=run)`.
- So:
  - `coverage = 1.0` means every sampled parameter set produced a valid vector there.
  - `coverage = 0.5` means only half of the parameter settings produced a valid vector.
  - `coverage = 0.0` means no valid vector from any run.

WCV then builds a `valid_mask` with `coverage >= min_valid_fraction`. Summary fields are masked
to `NaN` outside that reliability region.

### `ux_std` vs `sigma_param_ux`

In parameter-ensemble mode these are the same quantity:

- `ux_std` is `result.ux.std` (std-dev across parameter runs).
- `sigma_param_ux` is set directly to that same array (`sigma_param_ux = ux_stats.std`).

The duplicate naming exists so parameter uncertainty (`sigma_param_*`) can be combined with
bootstrap uncertainty (`sigma_boot_*`) when a bootstrap result is supplied.

### Undergrad-level interpretation of this uncertainty

Think of this as a **sensitivity test to tuning choices**:

- You did not change the movie; you changed plausible processing parameters.
- If velocity at a cell barely changes across those choices, parameter uncertainty is low.
- If it swings a lot, that velocity estimate is fragile to parameter tuning.

So this uncertainty tells you, cell-by-cell, **"How much does my answer depend on reasonable
analysis settings?"** It does **not** by itself capture all uncertainty sources (e.g., camera
noise, model mismatch, or temporal sampling variability unless you also add bootstrap).

### Two-stage uncertainty (bootstrap + parameter ensemble)

When `bootstrap_result` is provided to `run_parameter_ensemble_uncertainty`, WCV reports three
related maps for each component:

- `sigma_param`: spread across the parameter-ensemble runs (analysis-choice sensitivity).
- `sigma_boot`: spread from block bootstrap replicates (finite-data temporal resampling).
- `sigma_total`: a combined uncertainty computed as
  `sqrt(sigma_param**2 + sigma_boot**2)`.

In practice, this gives a useful decomposition: where `sigma_param` dominates, your estimate is
most sensitive to tuning choices; where `sigma_boot` dominates, your estimate is mostly limited by
finite noisy data.

### What these uncertainty maps are **not**

These are **precision/sensitivity** metrics, not direct **accuracy** metrics relative to the true
convection velocity. A small `sigma_param`, `sigma_boot`, or `sigma_total` means the estimate is
internally stable under the tested perturbations; it does **not** prove the estimate is close to a
physical ground truth.

In particular, these maps do not by themselves quantify systematic bias from sources such as model
mismatch, calibration error, optical distortion, tracer non-ideal behavior, or out-of-plane motion.
Absolute-accuracy claims still require external validation (for example synthetic-truth tests,
benchmark flows, or independent diagnostics).

## Building docs locally

```bash
pip install -r requirements-docs.txt
mkdocs serve
mkdocs build
```
