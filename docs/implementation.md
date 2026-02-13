# Implementation Map (Theory -> Code)

This page maps theory steps to the concrete module/function layout.

## Package structure

- `wcv/types.py`
  - `GridSpec`: defines `patch_px`, `grid_stride_patches`, and `bin_px` property.
  - `EstimationOptions`: thresholds/gating/weighting/padding controls.
  - `SingleSeedResult`, `VelocityMapResult`: typed outputs and diagnostics.
- `wcv/geometry.py`
  - Grid validation and bin-aligned padding helpers.
  - Coordinate transforms between pixel, extent, and metric units.
  - Box snapping and mask downsampling.
- `wcv/preprocess.py`
  - Block means, detrending, standardization, and common-mode regression.
- `wcv/correlation.py`
  - Lagged correlation kernels for single-seed and full-map workflows.
- `wcv/estimator_single_seed.py`
  - End-to-end single-seed velocity estimation.
- `wcv/estimator_map.py`
  - End-to-end map estimator over all seeds.
- `wcv/legacy.py`
  - Compatibility wrapper API for existing notebooks.

## Step-by-step mapping

## 1) Grid setup and shape handling

Theory: define analysis bins and ensure tileability.

Code:

- `GridSpec.bin_px`
- `validate_grid(...)`
- `compute_bin_aligned_padding(...)` + `np.pad(...)` when padding is enabled

Used in both estimators near the top of each function.

## 2) Build patch/bin signals

Theory: form bin-mean time series.

Code:

- `block_mean_timeseries(f, bin_px)` -> `y` / `region_raw`

## 3) Remove low-order trends and common background

Theory: detrend + regress out common background regressors.

Code:

- `detrend_series(...)` for `y_dt`, `seed_dt`, `region_dt`
- `build_background_regressors(...)` for matrix `g`
- `regress_out_common_mode_2d(...)` for `seed_res`, `region_res`

## 4) Correlation at multiple lags

Theory: compute lagged normalized cross-correlation.

Code:

- single seed: `shifted_corr_regions(region_res, seed_res, s)`
- map mode: `corr_matrix_positive_shift(region_res, s)`

## 5) Geometric displacement vectors

Theory: map grid cells to displacement vectors from seed(s).

Code:

- `patch_centers_dxdy_m(...)` in single-seed mode
- `patch_centers_xy_m(...)` + vector differences in map mode

## 6) Rejection/gating and weighted centroids

Theory: keep accepted correlations and compute weighted displacement per lag.

Code:

- thresholds via `EstimationOptions`:
  - `rmin`, `min_used`, `require_downstream`, `require_dy_positive`, `weight_power`
- single-seed diagnostics:
  - `mask_by_shift`, `n_used_by_shift`, `dx_bar_by_shift`, `dy_bar_by_shift`
- map mode per-seed accumulation:
  - `used_count`, lag-wise `taus/dxs/dys`

Edge-clipping guard (optional):

- enabled by `EstimationOptions.edge_clip_reject_k`
- computes weighted centroid/spread in bin-index space from accepted bins
- rejects lag contribution when `edge_distance < edge_clip_reject_k * support_radius`
- diagnostics surfaced as:
  - single seed: `edge_clipped_by_shift`, `edge_distance_by_shift`, `support_radius_by_shift`
  - map mode: `edge_clipped_by_shift` boolean maps by lag

The same accepted-bin weights (`|corr|**weight_power`) drive both displacement
centroids and edge-clipping support radius.

## 7) Fit velocity across lags

Theory: no-intercept fit of displacement vs lag time.

Code:

- `taus.append(s / fs)`
- `ux`, `uy` computed with `sum(taus * d*) / sum(taus**2)`

## 8) Results and diagnostics

- `SingleSeedResult` returns velocity, per-lag maps/masks, and fit diagnostics.
- `VelocityMapResult` returns `ux_map`, `uy_map`, accepted-count map, seed-mask info, and validity counts.
- Both include padding metadata (`padded`, `original_shape`, `padded_shape`).

## 9) Seed mask vs target mask in map estimators

Map estimators expose independent controls for:

- seed locations (`seed_mask_px` -> `seed_gate_vec`)
- candidate target bins (`shear_mask_px` with `use_shear_mask=True` -> `target_gate_vec`)

When `seed_mask_px` is omitted, seed gating defaults to the target gate for backward
compatibility.

## Compatibility layer

If migrating legacy notebooks gradually:

- use `estimate_velocity_per_shift_framework(...)` in `wcv/legacy.py`
- old argument names are preserved where practical and forwarded to modern estimators.
