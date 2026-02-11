# weighted-correlation-velocimetry

Refactored Python package for convection-velocity estimation from high-speed PLIF movies using patch-wise, time-lagged correlations and weighted-centroid displacement fitting.

## Documentation

Project documentation is published on GitHub Pages at:

`https://<user>.github.io/weighted-correlation-velocimetry/`

To preview docs locally:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

To build the static site:

```bash
mkdocs build --strict
```

## Install for JupyterLab

From the repo root (recommended editable install):

```bash
pip install -e .
```

If your notebook kernel uses a different interpreter, install into that kernel's Python:

```bash
python -m pip install -e .
```

Then in Jupyter, restart kernel and import normally.

## Superbinning convention

The package distinguishes two spatial scales:

1. `PATCH_PX` (base patch size): must tile the original image exactly.
2. `BIN_PX = PATCH_PX * grid_stride_patches` (effective analysis cell size): used everywhere that defines correlation-grid regions (block means, shear mask downsampling, geometry centers, and seed iteration).

This preserves the original image extents and avoids manual recropping/relabeling while allowing larger analysis bins.

If your movie dimensions are not divisible by `BIN_PX`, estimators default to strict mode and raise a `ValueError`. To enable edge padding, pass `allow_bin_padding=True` to the estimator call (preferred public API). `EstimationOptions.allow_bin_padding` is still honored for backward compatibility but is deprecated. When padding is applied, the estimator emits a `UserWarning` once per call including original/padded shapes and noting that edge padding can affect correlation/velocity estimates near image borders.

## Option A: modern API (recommended)

```python
from wcv import GridSpec, EstimationOptions, estimate_single_seed_velocity
from wcv.geometry import frac_to_px_box

grid = GridSpec(patch_px=4, grid_stride_patches=1)
opts = EstimationOptions(rmin=0.3, min_used=10)

nt, ny, nx = bgs_f.shape
bin_px = grid.bin_px

seed_box_px = frac_to_px_box(seed_px, seed_py, bin_px, bin_px, nx, ny, grid_px=bin_px)
bg_boxes_px = [
    frac_to_px_box(a["px"], a["py"], 3 * bin_px, 3 * bin_px, nx, ny, grid_px=bin_px)
    for a in BG_ANCHORS
]

res = estimate_single_seed_velocity(
    movie=bgs_f,
    fs=fs,
    grid=grid,
    seed_box_px=seed_box_px,
    bg_boxes_px=bg_boxes_px,
    extent_xd_yd=extent,
    dj_mm=Dj,
    shifts=(1,),
    options=opts,
    allow_bin_padding=True,  # preferred way to enable padding (warns once if applied)
)
print(res.ux, res.uy)
```

## Option B: drop-in notebook compatibility API

If you want to keep old notebook-style calls with minimal edits:

```python
from wcv import estimate_velocity_per_shift_framework, draw_boxes_debug
from wcv.geometry import frac_to_px_box
```

Your existing calls can stay mostly unchanged (including `PATCH_PX=` in `frac_to_px_box(...)` and `estimate_velocity_per_shift_framework(...)`). With `make_plots=True`, the compatibility wrapper now renders (i) seed/BG overlay, (ii) per-shift correlation maps with used-mask overlay (fixed color limits -0.25 to 0.25), (iii) per-shift accepted-bin scatter dx vs tau (sizes ~ |r|^2) with centroid/fit, and (iv) accepted-bin dx vs dy scatter with centroid markers.

```python
nt, ny, nx = bgs_f.shape

seed_box_px = frac_to_px_box(
    px=seed_px, py=seed_py,
    w_px=PATCH_PX, h_px=PATCH_PX,
    nx=nx, ny=ny,
    PATCH_PX=PATCH_PX,   # legacy alias supported
)

BG_W_PX = BG_NX_PATCH * PATCH_PX
BG_H_PX = BG_NY_PATCH * PATCH_PX
bg_boxes_px = [
    frac_to_px_box(
        px=a["px"], py=a["py"],
        w_px=BG_W_PX, h_px=BG_H_PX,
        nx=nx, ny=ny,
        PATCH_PX=PATCH_PX,  # legacy alias supported
    )
    for a in BG_ANCHORS
]

frame_vis = np.median(bgs_f, axis=0)
draw_boxes_debug(frame_vis, extent, seed_box_px, bg_boxes_px, BG_ANCHORS)

Ux, Uy, diag = estimate_velocity_per_shift_framework(
    bgs_f=bgs_f,
    fs=fs,
    PATCH_PX=PATCH_PX,
    seed_box_px=seed_box_px,
    bg_boxes_px=bg_boxes_px,
    extent_xD_yD=extent,
    Dj_mm=Dj,
    shifts=(1,),
    RMIN=0.3,
    MIN_USED=10,
    REQUIRE_DOWNSTREAM=True,
    allow_bin_padding=True,  # preferred way to enable padding (warns once if applied)
    make_plots=True,
    frame_vis=frame_vis,
)
```

## Package layout

- `wcv/preprocess.py`: detrending, z-scoring, common-mode regression, block means.
- `wcv/geometry.py`: extent conversions, patch/superpatch geometry, mask downsampling, box snapping.
- `wcv/correlation.py`: shifted correlations and full correlation matrices per positive shift.
- `wcv/estimator_single_seed.py`: single-seed estimator with per-shift diagnostics.
- `wcv/estimator_map.py`: velocity-map estimator over all seeds on the analysis grid.
- `wcv/plotting.py`: reusable plotting helpers.
- `wcv/legacy.py`: compatibility wrappers for old notebook API names/signatures.
- `examples/workflow_example.py`: minimal runnable workflow script.
