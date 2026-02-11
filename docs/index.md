# Weighted Correlation Velocimetry

`weighted-correlation-velocimetry` estimates convection velocity from image sequences (for example, high-speed PLIF) using patch-averaged signals, lagged cross-correlation, and weighted displacement fitting.

## What this package does

- Builds an analysis grid from `GridSpec(patch_px, grid_stride_patches)`.
- Converts movie frames to patch/bin time series.
- Detrends and removes common background modes.
- Computes lagged correlation for one seed (`estimate_single_seed_velocity`) or all seeds (`estimate_velocity_map`).
- Converts lagged displacement to physical velocity using sampling rate and geometric calibration.

## Core API

- `wcv.estimate_single_seed_velocity`: velocity estimate for one user-selected seed region.
- `wcv.estimate_velocity_map`: velocity map over all seed bins.
- `wcv.types.GridSpec`: controls base patch and superbin size (`bin_px = patch_px * grid_stride_patches`).
- `wcv.types.EstimationOptions`: gating/weighting options (including optional edge padding).

See:

- [Theory](theory.md) for derivation and notation.
- [Implementation](implementation.md) for file/function mapping.
- [Usage](usage.md) for practical setup and example code.
