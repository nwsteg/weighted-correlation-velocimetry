"""Minimal end-to-end workflow example.

Replace synthetic movie creation with your real `bgs_f` array from PLIF data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from wcv import EstimationOptions, GridSpec, estimate_single_seed_velocity, estimate_velocity_map
from wcv.geometry import frac_to_px_box
from wcv.plotting import plot_seed_and_background_boxes, plot_single_seed_correlation_maps


# 1) Load movie (nt, ny, nx). Here: synthetic demo movie with convection-like drift.
nt, ny, nx = 250, 128, 192
rng = np.random.default_rng(4)
base = rng.normal(0, 0.02, size=(nt, ny, nx)).astype(np.float32)
for t in range(nt):
    x0 = int(20 + 0.35 * t)
    y0 = int(50 + 0.06 * t)
    if 0 <= x0 < nx - 20 and 0 <= y0 < ny - 16:
        base[t, y0 : y0 + 16, x0 : x0 + 20] += 0.18
bgs_f = base

# 2) Define geometry/units.
extent_xd_yd = (-1.0, 7.0, -2.5, 2.5)
dj_mm = 10.0
fs = 15000.0

# 3) Configure base patch and superbin stride.
grid = GridSpec(patch_px=4, grid_stride_patches=4)  # BIN_PX=16
bin_px = grid.bin_px

# 4) Define seed and BG boxes snapped to BIN_PX lattice.
seed_box = frac_to_px_box(0.20, 0.45, bin_px, bin_px, nx, ny, grid_px=bin_px)
bg_boxes = [
    frac_to_px_box(0.05, 0.15, 2 * bin_px, 2 * bin_px, nx, ny, grid_px=bin_px),
    frac_to_px_box(0.80, 0.10, 2 * bin_px, 2 * bin_px, nx, ny, grid_px=bin_px),
]

# 5) Estimate single-seed velocity.
opts = EstimationOptions(rmin=0.05, min_used=4, require_downstream=True)
single = estimate_single_seed_velocity(
    movie=bgs_f,
    fs=fs,
    grid=grid,
    seed_box_px=seed_box,
    bg_boxes_px=bg_boxes,
    extent_xd_yd=extent_xd_yd,
    dj_mm=dj_mm,
    shifts=(1, 2, 3),
    options=opts,
)
print(f"single-seed Ux={single.ux:.3f} m/s, Uy={single.uy:.3f} m/s")

# 6) Estimate full velocity map.
vm = estimate_velocity_map(
    movie=bgs_f,
    fs=fs,
    grid=grid,
    bg_boxes_px=bg_boxes,
    extent_xd_yd=extent_xd_yd,
    dj_mm=dj_mm,
    shifts=(1, 2, 3),
    options=opts,
)
print(f"valid seeds: {vm.valid_seed_count}/{vm.total_seed_count}")

# 7) Plot key diagnostics.
frame_vis = np.median(bgs_f, axis=0)
plot_seed_and_background_boxes(frame_vis, extent_xd_yd, seed_box, bg_boxes)
plot_single_seed_correlation_maps(single, frame_vis, extent_xd_yd)

plt.figure(figsize=(6, 3.5))
plt.imshow(vm.ux_map, cmap="viridis")
plt.colorbar(label="Ux (m/s)")
plt.title("Velocity map Ux on superbin grid")
plt.tight_layout()
plt.show()
