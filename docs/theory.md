# Theory and Estimation Model

This page documents the velocity-estimation theory used in this repository and maps equations to code-level variable names.

## 1. Inputs, notation, and units

Let the movie be

$$
I_t(y, x), \quad t=0,\dots,N_t-1,
$$

with pixel coordinates $(y,x)$ and time index $t$.

Repository names:

- movie tensor: `movie` / `f` (`estimate_single_seed_velocity`, `estimate_velocity_map`)
- shape: `(nt, ny, nx)`
- frame rate: `fs` (Hz)
- lag in frames: `s` from `shifts`

The analysis grid is controlled by:

- base patch size: `patch_px`
- superbin stride in patch units: `grid_stride_patches`
- effective bin size: `bin_px = patch_px * grid_stride_patches`

The grid must tile the image (strict mode) or the movie is optionally padded at the bottom/right before processing (padding mode).

## 2. Spatial binning (superbinning)

Patch/bin means are formed by block averaging:

$$
Y_j(t) = \frac{1}{B^2} \sum_{(y,x)\in\Omega_j} I_t(y,x),
$$

where $B=\texttt{bin\_px}$ and $\Omega_j$ is bin $j$.

In code:

- `block_mean_timeseries(movie, bin_px)` returns `y` with shape `(By*Bx, nt)`.
- `validate_grid(ny, nx, patch_px, grid_stride_patches)` enforces tileability and returns `(bin_px, by, bx)`.

## 3. Preprocessing and normalization

The implementation uses detrending and common-mode regression:

1. Detrend bin/seed traces:

$$
\tilde{Y}_j(t) = \operatorname{detrend}(Y_j(t)),
$$

using `detrend_series(..., detrend_type="linear"|"constant")`.

2. Build background regressors from user-selected background boxes:

$$
G_k(t), \quad k=1,\dots,K,
$$

using `build_background_regressors(f, bg_boxes_px, ...)`.

3. Regress out common background modes:

$$
R_j(t) = \tilde{Y}_j(t) - \hat{\beta}_j^\top G(t),
$$

using `regress_out_common_mode_2d`.

For single-seed mode, the seed trace is similarly detrended and regressed:

- `seed_raw`, `seed_dt`, `seed_res`.

## 4. Lagged correlation

For lag $s>0$, correlation is computed between region residual traces and shifted seed/region traces.

Single seed uses:

$$
r_j(s) = \operatorname{corr}(R_j(t+s), S(t)),
$$

implemented by `shifted_corr_regions(region_res, seed_res, s)`.

Map mode computes a full matrix for each lag:

$$
\mathbf{R}^{(s)}_{ij} = \operatorname{corr}(R_i(t+s), R_j(t)),
$$

implemented by `corr_matrix_positive_shift(region_res, s)`.

## 5. Gating, rejection, and quality metrics

For each lag, valid contributions satisfy code-level gates:

- finite correlation
- positive correlation (`r > 0`)
- threshold `r >= options.rmin`
- minimum accepted bin count `options.min_used`
- optional downstream constraint `dx > 0` when `options.require_downstream`
- optional positive vertical displacement `dy > 0` when `options.require_dy_positive`

Weighting is power-law in correlation magnitude:

$$
w_j = |r_j|^{p}, \quad p=\texttt{options.weight\_power}.
$$

Then per-shift weighted centroid displacements are:

$$
\bar{\Delta x}(s)=\frac{\sum_j w_j\,\Delta x_j}{\sum_j w_j},
\qquad
\bar{\Delta y}(s)=\frac{\sum_j w_j\,\Delta y_j}{\sum_j w_j}.
$$

Repository variables:

- `dx_bar_by_shift[s]`, `dy_bar_by_shift[s]`, `n_used_by_shift[s]` (single seed)
- `used_count_map` accumulation (map mode)

Note: this implementation does **not** pick a single best lag by max-$r$ per pixel. It combines accepted information per lag and then fits across lag values.

## 6. From lag to velocity

For lag $s$ frames:

$$
\tau_s = \frac{s}{f_s} \quad (\text{seconds}),
$$

where $f_s=\texttt{fs}$.

A no-intercept least-squares fit is used:

$$
U_x = \frac{\sum_s \tau_s \bar{\Delta x}(s)}{\sum_s \tau_s^2},
\qquad
U_y = \frac{\sum_s \tau_s \bar{\Delta y}(s)}{\sum_s \tau_s^2}.
$$

In code: `taus`, `dxs`, `dys`, `denom = sum(taus*taus)`, and output `ux`, `uy`.

### Units

Displacements are represented in meters before fitting:

- `patch_centers_xy_m` and `patch_centers_dxdy_m` convert extents in `x/D, y/D` to meters using `dj_mm`.
- Therefore, fitted `ux`, `uy` are in m/s.

If calibration is absent, the same framework can be interpreted in pixel/bin units by skipping physical conversion, but this repository’s main estimators use metric conversion with `dj_mm`.

## 7. Masks, shear selection, and map estimation

Map mode optionally limits seed locations using a shear mask:

- `build_shear_mask_patchvec(..., shear_mask_px, shear_pctl, thresh_patch)`
- If no pixel mask is provided, a percentile threshold on median frame intensity generates it.

This affects which seeds are solved (`seeds = np.flatnonzero(shear)`), not the correlation definition itself.

## 8. Padding and edge handling

When shape is not divisible by `bin_px`:

- strict mode (`allow_bin_padding=False` and `options.allow_bin_padding=False`) raises `ValueError`.
- padding mode pads bottom/right to the next multiple via `compute_bin_aligned_padding` and `np.pad(..., mode=options.padding_mode)`.

A `UserWarning` is emitted because edge padding can bias correlation near borders.

Result metadata tracks this:

- `padded` (bool)
- `original_shape`
- `padded_shape`

## 9. Coordinate mapping back to physical/image space

Grid-center coordinates are derived from the original extent mapping used for image display:

$$
x_D = x_{\min} + \frac{x_c}{n_x-1}(x_{\max}-x_{\min}),
$$

and similarly for $y_D$ (with `origin="upper"` handling vertical direction).

Functions:

- `patch_centers_xy_m` for absolute centers
- `patch_centers_dxdy_m` for displacements from seed center
- `box_px_to_extent_xywh` and `frac_to_px_box` for visualization and box construction

So indexing follows the padded processing grid when padding is active, while geometric conversion still follows the configured extent convention.
