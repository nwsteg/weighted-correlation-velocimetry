from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .geometry import box_px_to_extent_xywh
from .types import SingleSeedResult


def plot_seed_and_background_boxes(frame_vis, extent_xd_yd, seed_box_px, bg_boxes_px, origin="upper"):
    ny, nx = frame_vis.shape
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    ax.imshow(frame_vis, extent=extent_xd_yd, cmap="gray")

    sx, sy, sw, sh = box_px_to_extent_xywh(seed_box_px, extent_xd_yd, nx, ny, origin=origin)
    ax.add_patch(patches.Rectangle((sx, sy), sw, sh, fill=False, ec="r", lw=2, label="seed"))

    for i, box in enumerate(bg_boxes_px):
        x, y, w, h = box_px_to_extent_xywh(box, extent_xd_yd, nx, ny, origin=origin)
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, ec="c", lw=2, label="bg" if i == 0 else None))

    ax.set_xlabel("x/D")
    ax.set_ylabel("y/D")
    ax.legend(loc="upper right")
    return fig, ax


def plot_single_seed_correlation_maps(
    result: SingleSeedResult,
    frame_vis: np.ndarray,
    extent_xd_yd,
    vmin: float = -0.25,
    vmax: float = 0.25,
):
    figs = []
    for s in result.shifts:
        r = result.corr_by_shift[s].reshape(result.by, result.bx)
        m = result.mask_by_shift[s].reshape(result.by, result.bx)
        fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
        ax.imshow(frame_vis, extent=extent_xd_yd, cmap="gray", alpha=0.3)
        im = ax.imshow(r, extent=extent_xd_yd, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
        rgba = np.zeros((result.by, result.bx, 4), dtype=float)
        rgba[..., 1] = 0.85
        rgba[..., 2] = 1.0
        rgba[..., 3] = 0.7 * m.astype(float)
        ax.imshow(rgba, extent=extent_xd_yd, interpolation="nearest")
        ax.set_title(f"shift={s}, used={result.n_used_by_shift[s]}")
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        fig.colorbar(im, ax=ax, label="Pearson r")
        figs.append(fig)
    return figs
