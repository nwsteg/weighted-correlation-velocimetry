from __future__ import annotations

import numpy as np


def validate_grid(ny: int, nx: int, patch_px: int, grid_stride_patches: int) -> tuple[int, int, int]:
    """Validate base/super grid and return (bin_px, By, Bx)."""
    if patch_px <= 0 or grid_stride_patches <= 0:
        raise ValueError("patch_px and grid_stride_patches must be positive integers")
    if ny % patch_px or nx % patch_px:
        raise ValueError("patch_px must tile image dimensions exactly")
    bin_px = patch_px * grid_stride_patches
    if ny % bin_px or nx % bin_px:
        raise ValueError(
            f"superbin bin_px={bin_px} must tile image dimensions ({ny},{nx}) exactly"
        )
    return bin_px, ny // bin_px, nx // bin_px


def compute_bin_aligned_padding(ny: int, nx: int, bin_px: int) -> tuple[int, int, int, int]:
    """Return padded (ny, nx) and pad sizes to align with bin_px."""
    if bin_px <= 0:
        raise ValueError("bin_px must be a positive integer")
    ny_pad = ((int(ny) + bin_px - 1) // bin_px) * bin_px
    nx_pad = ((int(nx) + bin_px - 1) // bin_px) * bin_px
    return ny_pad, nx_pad, ny_pad - int(ny), nx_pad - int(nx)


def box_px_to_extent_xywh(box_px, extent_xd_yd, nx: int, ny: int, origin: str = "upper"):
    xmin, xmax, ymin, ymax = map(float, extent_xd_yd)
    y0, y1, x0, x1 = map(int, box_px)

    x0d = xmin + (x0 / (nx - 1)) * (xmax - xmin)
    x1d = xmin + (x1 / (nx - 1)) * (xmax - xmin)
    if origin == "upper":
        y0d = ymax - (y0 / (ny - 1)) * (ymax - ymin)
        y1d = ymax - (y1 / (ny - 1)) * (ymax - ymin)
    else:
        y0d = ymin + (y0 / (ny - 1)) * (ymax - ymin)
        y1d = ymin + (y1 / (ny - 1)) * (ymax - ymin)

    y_lo, y_hi = min(y0d, y1d), max(y0d, y1d)
    return x0d, y_lo, (x1d - x0d), (y_hi - y_lo)


def patch_centers_xy_m(ny, nx, by, bx, bin_px, extent_xd_yd, dj_mm, origin="upper"):
    """Absolute center coordinates (x_m, y_m) for each BIN_PX cell."""
    if by * bin_px != ny or bx * bin_px != nx:
        raise ValueError("by/bx/bin_px inconsistent with image shape")

    xmin, xmax, ymin, ymax = map(float, extent_xd_yd)
    yc = np.arange(by) * bin_px + (bin_px - 1) / 2.0
    xc = np.arange(bx) * bin_px + (bin_px - 1) / 2.0
    yc_grid, xc_grid = np.meshgrid(yc, xc, indexing="ij")

    xd = xmin + (xc_grid / (nx - 1)) * (xmax - xmin)
    if origin == "upper":
        yd = ymax - (yc_grid / (ny - 1)) * (ymax - ymin)
    else:
        yd = ymin + (yc_grid / (ny - 1)) * (ymax - ymin)

    d_to_m = float(dj_mm) * 1e-3
    return (xd * d_to_m).ravel().astype(np.float32), (yd * d_to_m).ravel().astype(np.float32)


def patch_centers_dxdy_m(ny, nx, by, bx, bin_px, extent_xd_yd, seed_box_px, dj_mm, origin="upper"):
    """Displacement vectors from seed center to each BIN_PX cell center."""
    x_m, y_m = patch_centers_xy_m(ny, nx, by, bx, bin_px, extent_xd_yd, dj_mm, origin=origin)

    sy0, sy1, sx0, sx1 = map(int, seed_box_px)
    xs = (sx0 + sx1 - 1) / 2.0
    ys = (sy0 + sy1 - 1) / 2.0

    xmin, xmax, ymin, ymax = map(float, extent_xd_yd)
    xsd = xmin + (xs / (nx - 1)) * (xmax - xmin)
    if origin == "upper":
        ysd = ymax - (ys / (ny - 1)) * (ymax - ymin)
    else:
        ysd = ymin + (ys / (ny - 1)) * (ymax - ymin)
    d_to_m = float(dj_mm) * 1e-3

    return x_m - xsd * d_to_m, y_m - ysd * d_to_m


def px_box_snap_aligned(
    x_px,
    y_px,
    w_px,
    h_px,
    nx,
    ny,
    grid_px=None,
    require_multiple=True,
    GRID_PX=None,
    PATCH_PX=None,
):
    """Snap a box to a lattice.

    Parameters accept `grid_px` (preferred) plus legacy aliases `GRID_PX` and
    `PATCH_PX` for notebook compatibility.
    """
    if grid_px is None:
        grid_px = GRID_PX if GRID_PX is not None else PATCH_PX
    if grid_px is None:
        raise ValueError("grid_px (or GRID_PX/PATCH_PX alias) is required")
    grid_px = int(grid_px)
    w_px = int(w_px)
    h_px = int(h_px)
    if require_multiple and (w_px % grid_px or h_px % grid_px):
        raise ValueError("box width/height must be multiples of grid_px")
    x0 = (int(x_px) // grid_px) * grid_px
    y0 = (int(y_px) // grid_px) * grid_px
    x0 = int(np.clip(x0, 0, nx - w_px))
    y0 = int(np.clip(y0, 0, ny - h_px))
    return y0, y0 + h_px, x0, x0 + w_px


def frac_to_px_box(
    px,
    py,
    w_px,
    h_px,
    nx,
    ny,
    grid_px=None,
    require_multiple=True,
    GRID_PX=None,
    PATCH_PX=None,
):
    x = int(np.clip(np.round(float(px) * (nx - 1)), 0, nx - 1))
    y = int(np.clip(np.round(float(py) * (ny - 1)), 0, ny - 1))
    return px_box_snap_aligned(
        x, y, w_px, h_px, nx, ny, grid_px=grid_px, GRID_PX=GRID_PX, PATCH_PX=PATCH_PX, require_multiple=require_multiple
    )


def mask_px_to_patch_vec(mask_px, by, bx, bin_px, thresh=0.5):
    m = np.asarray(mask_px, dtype=bool)
    nyg, nxg = by * bin_px, bx * bin_px
    if m.shape[0] < nyg or m.shape[1] < nxg:
        raise ValueError(f"mask too small for requested grid: need ({nyg},{nxg}), got {m.shape}")
    blk = m[:nyg, :nxg].reshape(by, bin_px, bx, bin_px).mean(axis=(1, 3))
    return (blk > float(thresh)).ravel()


def build_shear_mask_patchvec(
    movie, by, bx, bin_px, shear_mask_px=None, shear_pctl=50, thresh_patch=0.5, use_abs=False
):
    f = np.asarray(movie)
    if shear_mask_px is None:
        frame = np.median(f[:, : by * bin_px, : bx * bin_px], axis=0)
        src = np.abs(frame) if use_abs else frame
        thr = float(np.percentile(src, float(shear_pctl)))
        shear_mask_px = src > thr
    return mask_px_to_patch_vec(shear_mask_px, by=by, bx=bx, bin_px=bin_px, thresh=thresh_patch)
