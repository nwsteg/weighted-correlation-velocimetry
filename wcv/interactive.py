from __future__ import annotations

"""Interactive single-seed correlation explorers for PLIF movies.

Two entrypoints are provided:
- `make_plif_interactive_widget()` for notebook use (ipywidgets).
- `launch_plif_interactive_gui()` for command-line desktop GUI use (matplotlib widgets).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.widgets import RadioButtons, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection side effect

from .estimator_single_seed import estimate_single_seed_velocity
from .geometry import frac_to_px_box
from .types import EstimationOptions, GridSpec


def load_extent(extent_file: str | Path) -> tuple[float, float, float, float] | None:
    """Load extent from an extent.txt style file containing `xs xe ys ye`."""
    extent_path = Path(extent_file)
    if not extent_path.exists():
        return None

    try:
        line = extent_path.read_text(encoding="utf-8").strip()
        xs, xe, ys, ye = map(float, line.split()[:4])
        return xs, xe, ys, ye
    except Exception:
        return None


def _parse_frame_slice(spec: str | None) -> slice:
    if spec is None or spec.strip() == "":
        return slice(None)

    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("Frame range must look like start:end or start:end:step.")
    vals = [int(p) if p.strip() else None for p in parts]
    if len(vals) == 2:
        vals.append(None)
    return slice(*vals)


def _collect_tif_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(list(path.glob("*.tif")) + list(path.glob("*.tiff")))
    return []


def _read_tif_stack(path: Path) -> np.ndarray:
    try:
        import tifffile

        arr = np.asarray(tifffile.imread(path), dtype=np.float32)
    except ImportError:
        arr = np.asarray(plt.imread(path), dtype=np.float32)

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim > 3:
        raise ValueError(f"Unsupported TIFF dimensionality for {path}: {arr.shape}")
    return arr


def load_user_plif(
    source: str | Path,
    frame_range: str | None,
    extent_file: str | Path,
    fs_hz: float,
    dj_mm: float = 1.0,
) -> tuple[np.ndarray, tuple[float, float, float, float], float, float]:
    """Load TIFF sequence from a file or folder, apply frame slicing, and normalize."""
    source_path = Path(source)
    tif_paths = _collect_tif_paths(source_path)
    if not tif_paths:
        raise ValueError("No .tif/.tiff files found at the selected source.")

    stacks = [_read_tif_stack(p) for p in tif_paths]
    movie = np.concatenate(stacks, axis=0)
    movie = movie[_parse_frame_slice(frame_range)]
    if movie.size == 0:
        raise ValueError("Selected frame range produced an empty sequence.")

    max_val = float(np.max(np.abs(movie)))
    if max_val <= 0:
        raise ValueError("Input frames are all zero.")
    movie = (movie / max_val).astype(np.float32, copy=False)

    extent = load_extent(extent_file)
    if extent is None:
        raise ValueError("Could not load extent. Ensure extent.txt contains: xs xe ys ye")

    return movie, extent, float(dj_mm), float(fs_hz)


def _prompt_gui_inputs() -> tuple[str, str, str, float]:
    import tkinter as tk
    from tkinter import filedialog, simpledialog

    root = tk.Tk()
    root.withdraw()

    source_file = filedialog.askopenfilename(
        title="Select a TIFF file (Cancel to pick a folder)",
        filetypes=(("TIFF", "*.tif *.tiff"), ("All files", "*.*")),
    )
    source = source_file or filedialog.askdirectory(title="Select folder containing TIFF files")
    if not source:
        raise ValueError("No TIFF source selected.")

    frame_range = simpledialog.askstring(
        "Frame range",
        "Enter frame range as start:end or start:end:step (leave blank for all):",
        initialvalue="100:200",
        parent=root,
    )
    fs_hz = simpledialog.askfloat("Sample rate", "Enter sample rate [Hz]:", initialvalue=50000.0, parent=root)
    if fs_hz is None or fs_hz <= 0:
        raise ValueError("Sample rate must be positive.")

    extent_file = filedialog.askopenfilename(
        title="Select extent.txt",
        filetypes=(("Text", "*.txt"), ("All files", "*.*")),
    )
    if not extent_file:
        raise ValueError("No extent file selected.")

    root.destroy()
    return source, frame_range or "", extent_file, float(fs_hz)


def load_hardcoded_plif() -> tuple[np.ndarray, tuple[float, float, float, float], float, float]:
    """Backward-compatible loader that now prompts for user-selected inputs."""
    source, frame_range, extent_file, fs_hz = _prompt_gui_inputs()
    return load_user_plif(
        source=source,
        frame_range=frame_range,
        extent_file=extent_file,
        fs_hz=fs_hz,
        dj_mm=1.0,
    )


def _default_bg_boxes(nx: int, ny: int, bin_px: int) -> list[tuple[int, int, int, int]]:
    # Keep simple/robust: use 2x2-bin corners as common-mode regressors.
    w = 2 * bin_px
    h = 2 * bin_px
    anchors = [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92)]
    return [frac_to_px_box(px, py, w, h, nx, ny, grid_px=bin_px) for px, py in anchors]


def _seed_box_from_click(
    nx: int, ny: int, bin_px: int, px_frac: float, py_frac: float
) -> tuple[int, int, int, int]:
    return frac_to_px_box(
        px=px_frac,
        py=py_frac,
        w_px=bin_px,
        h_px=bin_px,
        nx=nx,
        ny=ny,
        grid_px=bin_px,
    )


def _compute_corr_result(
    movie: np.ndarray,
    extent_xd_yd: tuple[float, float, float, float],
    dj_mm: float,
    fs: float,
    nx: int,
    ny: int,
    px: float,
    py: float,
    patch_px: int,
    stride: int,
    shift: int,
    rmin: float,
    weight_power: float,
    edge_clip_k: float,
):
    grid = GridSpec(patch_px=int(patch_px), grid_stride_patches=int(stride))
    bin_px = grid.bin_px
    seed_box = _seed_box_from_click(nx, ny, bin_px, px, py)
    bg_boxes = _default_bg_boxes(nx, ny, bin_px)

    result = estimate_single_seed_velocity(
        movie=movie,
        fs=fs,
        grid=grid,
        seed_box_px=seed_box,
        bg_boxes_px=bg_boxes,
        extent_xd_yd=extent_xd_yd,
        dj_mm=dj_mm,
        shifts=(int(shift),),
        options=EstimationOptions(
            rmin=float(rmin),
            weight_power=float(weight_power),
            min_used=3,
            require_downstream=False,
            require_dy_positive=False,
            edge_clip_reject_k=float(edge_clip_k),
        ),
        allow_bin_padding=True,
    )
    corr_vec = np.asarray(result.corr_by_shift[int(shift)], dtype=float)
    corr_grid = corr_vec.reshape(result.by, result.bx)
    edge_clipped = bool((result.edge_clipped_by_shift or {}).get(int(shift), False))
    edge_distance = float((result.edge_distance_by_shift or {}).get(int(shift), np.nan))
    support_radius = float((result.support_radius_by_shift or {}).get(int(shift), np.nan))
    return result, seed_box, corr_grid, bin_px, edge_clipped, edge_distance, support_radius


def _edge_circle_params(
    *,
    result,
    shift: int,
    weight_power: float,
    bin_px: int,
    nx: int,
    ny: int,
    extent_xd_yd: tuple[float, float, float, float],
) -> tuple[float, float, float] | None:
    mask = np.asarray(result.mask_by_shift.get(int(shift), np.array([], dtype=bool)), dtype=bool)
    corr = np.asarray(result.corr_by_shift.get(int(shift), np.array([], dtype=float)), dtype=float)
    if mask.size != result.by * result.bx or corr.size != mask.size:
        return None

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None

    weights = (np.abs(corr[idx]).astype(np.float64) ** float(weight_power))
    wsum = float(weights.sum())
    if wsum <= 0 or not np.isfinite(wsum):
        return None

    rows = (idx // result.bx).astype(np.float64)
    cols = (idx % result.bx).astype(np.float64)
    r_bar = float(np.sum(weights * rows) / wsum)
    c_bar = float(np.sum(weights * cols) / wsum)

    xpix = c_bar * bin_px + (bin_px - 1) / 2.0
    ypix = r_bar * bin_px + (bin_px - 1) / 2.0
    xmin, xmax, ymin, ymax = map(float, extent_xd_yd)
    x_center = float(np.interp(xpix, [0, nx - 1], [xmin, xmax]))
    y_center = float(np.interp(ypix, [0, ny - 1], [ymax, ymin]))

    support_radius = float((result.support_radius_by_shift or {}).get(int(shift), np.nan))
    if not np.isfinite(support_radius):
        return None
    radius_x = abs(float((support_radius * float(bin_px) * (xmax - xmin)) / max(nx - 1, 1)))
    return x_center, y_center, radius_x


def make_plif_interactive_widget() -> None:
    """Create and display an interactive seed-click correlation explorer in notebooks."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover - runtime environment dependent
        raise ImportError(
            "ipywidgets is required for notebook mode. Install with `pip install ipywidgets`."
        ) from exc

    movie, extent_xd_yd, dj_mm, fs = load_hardcoded_plif()
    _, ny, nx = movie.shape
    mean_img = movie.mean(axis=0)

    patch_px = widgets.IntSlider(value=8, min=2, max=64, step=2, description="PATCH_PX")
    stride = widgets.IntSlider(value=1, min=1, max=8, step=1, description="stride")
    shift = widgets.IntSlider(value=1, min=1, max=20, step=1, description="shift")
    rmin = widgets.FloatSlider(value=0.2, min=0.0, max=0.95, step=0.01, description="rmin")
    weight_power = widgets.FloatSlider(value=2.0, min=0.5, max=5.0, step=0.5, description="weight")
    edge_clip_k = widgets.FloatSlider(value=1.0, min=0.0, max=4.0, step=0.1, description="edge k (d<k*r)")
    vlim = widgets.FloatSlider(value=0.5, min=0.05, max=1.0, step=0.05, description="|corr| lim")
    view_mode = widgets.ToggleButtons(options=["2D", "3D"], value="2D", description="view")

    status = widgets.HTML(value="Click the mean image to choose a seed.")

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_corr2d = fig.add_subplot(gs[0, 1])
    ax_corr3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_corr3d.set_visible(False)

    ax_img.set_title("Mean image (click for seed)")
    ax_img.set_xlabel("x/D")
    ax_img.set_ylabel("y/D")
    ax_corr2d.set_title("Correlation map (2D)")
    ax_corr2d.set_xlabel("x/D")
    ax_corr2d.set_ylabel("y/D")

    ax_img.imshow(mean_img, cmap="gray", extent=extent_xd_yd, origin="upper")
    click_marker, = ax_img.plot([], [], "r+", ms=12, mew=2)
    seed_rect = None
    edge_circle = None
    surf = None

    corr_im = ax_corr2d.imshow(
        np.full_like(mean_img, np.nan, dtype=float),
        cmap="RdBu_r",
        vmin=-vlim.value,
        vmax=vlim.value,
        extent=extent_xd_yd,
        origin="upper",
    )
    divider = make_axes_locatable(ax_corr2d)
    cax = divider.append_axes("right", size="4%", pad=0.06)
    plt.colorbar(corr_im, cax=cax, label="corr")

    state = {"px": None, "py": None}

    def _render(corr_grid: np.ndarray, bin_px_val: int):
        nonlocal surf
        if view_mode.value == "2D":
            ax_corr2d.set_visible(True)
            ax_corr3d.set_visible(False)
            corr_img = np.repeat(np.repeat(corr_grid, bin_px_val, axis=0), bin_px_val, axis=1)
            corr_im.set_data(corr_img[:ny, :nx])
            corr_im.set_clim(-float(vlim.value), float(vlim.value))
        else:
            ax_corr2d.set_visible(False)
            ax_corr3d.set_visible(True)
            if surf is not None:
                surf.remove()
            x = np.linspace(extent_xd_yd[0], extent_xd_yd[1], corr_grid.shape[1])
            y = np.linspace(extent_xd_yd[3], extent_xd_yd[2], corr_grid.shape[0])
            xx, yy = np.meshgrid(x, y)
            ax_corr3d.clear()
            surf = ax_corr3d.plot_surface(
                xx,
                yy,
                corr_grid,
                cmap="RdBu_r",
                vmin=-float(vlim.value),
                vmax=float(vlim.value),
                linewidth=0,
                antialiased=False,
            )
            ax_corr3d.set_title("Correlation map (3D surface)")
            ax_corr3d.set_xlabel("x/D")
            ax_corr3d.set_ylabel("y/D")
            ax_corr3d.set_zlabel("corr")

    def _update(*_args):
        nonlocal seed_rect, edge_circle
        if state["px"] is None or state["py"] is None:
            return
        try:
            result, seed_box, corr_grid, bin_px_val, edge_clipped, edge_distance, support_radius = _compute_corr_result(
                movie,
                extent_xd_yd,
                dj_mm,
                fs,
                nx,
                ny,
                state["px"],
                state["py"],
                int(patch_px.value),
                int(stride.value),
                int(shift.value),
                float(rmin.value),
                float(weight_power.value),
                float(edge_clip_k.value),
            )
            _render(corr_grid, bin_px_val)

            y0, y1, x0, x1 = seed_box
            x0d, x1d = np.interp([x0, x1], [0, nx - 1], [extent_xd_yd[0], extent_xd_yd[1]])
            y0d, y1d = np.interp([y0, y1], [0, ny - 1], [extent_xd_yd[3], extent_xd_yd[2]])
            if seed_rect is not None:
                seed_rect.remove()
            seed_rect = plt.Rectangle((x0d, min(y0d, y1d)), x1d - x0d, abs(y1d - y0d), fill=False, ec="yellow", lw=1.5)
            ax_img.add_patch(seed_rect)

            circle_params = _edge_circle_params(
                result=result,
                shift=int(shift.value),
                weight_power=float(weight_power.value),
                bin_px=int(bin_px_val),
                nx=nx,
                ny=ny,
                extent_xd_yd=extent_xd_yd,
            )
            if edge_circle is not None:
                edge_circle.remove()
                edge_circle = None
            if circle_params is not None:
                cx, cy, radius_x = circle_params
                edge_circle = Circle(
                    (cx, cy),
                    radius_x,
                    fill=False,
                    lw=2.0,
                    ls="--",
                    ec=("red" if edge_clipped else "lime"),
                )
                ax_corr2d.add_patch(edge_circle)


            status.value = (
                f"<b>Seed:</b> ({state['px']:.3f}, {state['py']:.3f}) | "
                f"bin_px={bin_px_val} | used bins={result.n_used_by_shift[int(shift.value)]} | "
                f"edge clipped={edge_clipped} [check: d < k*r] "
                f"(d={edge_distance:.2f}, r={support_radius:.2f}, k={float(edge_clip_k.value):.2f}) | "
                f"Ux={result.ux:.3f} m/s, Uy={result.uy:.3f} m/s"
            )
            fig.canvas.draw_idle()
        except Exception as exc:  # pragma: no cover
            status.value = f"<span style='color:#b00'><b>Error:</b> {exc}</span>"

    def _on_click(event):
        if event.inaxes != ax_img or event.xdata is None or event.ydata is None:
            return
        px = float(np.clip((event.xdata - extent_xd_yd[0]) / (extent_xd_yd[1] - extent_xd_yd[0]), 0.0, 1.0))
        py = float(np.clip((extent_xd_yd[3] - event.ydata) / (extent_xd_yd[3] - extent_xd_yd[2]), 0.0, 1.0))
        state["px"], state["py"] = px, py
        click_marker.set_data([event.xdata], [event.ydata])
        _update()

    fig.canvas.mpl_connect("button_press_event", _on_click)
    for w in (patch_px, stride, shift, rmin, weight_power, edge_clip_k, vlim, view_mode):
        w.observe(_update, names="value")

    controls = widgets.VBox([widgets.HBox([patch_px, stride, shift]), widgets.HBox([rmin, weight_power, edge_clip_k, vlim, view_mode]), status])
    display(controls)
    display(fig)


def launch_plif_interactive_gui(
    *,
    source: str | None = None,
    frame_range: str | None = None,
    extent_file: str | None = None,
    fs_hz: float | None = None,
) -> None:
    """Launch a command-line desktop GUI with 2D/3D correlation-map toggle."""
    if source is None or extent_file is None or fs_hz is None:
        source, frame_range, extent_file, fs_hz = _prompt_gui_inputs()

    movie, extent_xd_yd, dj_mm, fs = load_user_plif(
        source=source,
        frame_range=frame_range,
        extent_file=extent_file,
        fs_hz=float(fs_hz),
    )
    _, ny, nx = movie.shape
    mean_img = movie.mean(axis=0)

    fig = plt.figure(figsize=(14, 8))
    ax_img = fig.add_axes([0.05, 0.22, 0.40, 0.73])
    ax_corr2d = fig.add_axes([0.50, 0.22, 0.45, 0.73])
    ax_corr3d = fig.add_axes([0.50, 0.22, 0.45, 0.73], projection="3d")
    ax_corr3d.set_visible(False)

    ax_img.set_title("Mean image (click for seed)")
    ax_img.set_xlabel("x/D")
    ax_img.set_ylabel("y/D")
    ax_img.imshow(mean_img, cmap="gray", extent=extent_xd_yd, origin="upper")
    click_marker, = ax_img.plot([], [], "r+", ms=12, mew=2)

    ax_corr2d.set_title("Correlation map (2D)")
    ax_corr2d.set_xlabel("x/D")
    ax_corr2d.set_ylabel("y/D")
    corr_im = ax_corr2d.imshow(
        np.full_like(mean_img, np.nan, dtype=float),
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        extent=extent_xd_yd,
        origin="upper",
    )
    divider = make_axes_locatable(ax_corr2d)
    cax = divider.append_axes("right", size="4%", pad=0.06)
    plt.colorbar(corr_im, cax=cax, label="corr")

    ax_patch = fig.add_axes([0.08, 0.14, 0.30, 0.03])
    ax_stride = fig.add_axes([0.08, 0.10, 0.30, 0.03])
    ax_shift = fig.add_axes([0.08, 0.06, 0.30, 0.03])
    ax_rmin = fig.add_axes([0.56, 0.14, 0.30, 0.03])
    ax_weight = fig.add_axes([0.56, 0.10, 0.30, 0.03])
    ax_edge_k = fig.add_axes([0.56, 0.02, 0.30, 0.03])
    ax_vlim = fig.add_axes([0.56, 0.06, 0.30, 0.03])
    ax_mode = fig.add_axes([0.40, 0.04, 0.12, 0.12])

    s_patch = Slider(ax_patch, "PATCH_PX", 2, 64, valinit=8, valstep=2)
    s_stride = Slider(ax_stride, "stride", 1, 8, valinit=1, valstep=1)
    s_shift = Slider(ax_shift, "shift", 1, 20, valinit=1, valstep=1)
    s_rmin = Slider(ax_rmin, "rmin", 0.0, 0.95, valinit=0.2, valstep=0.01)
    s_weight = Slider(ax_weight, "weight", 0.5, 5.0, valinit=2.0, valstep=0.5)
    s_vlim = Slider(ax_vlim, "|corr| lim", 0.05, 1.0, valinit=0.5, valstep=0.05)
    s_edge_k = Slider(ax_edge_k, "edge k (d<k*r)", 0.0, 4.0, valinit=1.0, valstep=0.1)
    mode = RadioButtons(ax_mode, ("2D", "3D"), active=0)

    status_txt = fig.text(0.05, 0.01, "Click the mean image to choose a seed.", fontsize=10)
    seed_rect = None
    edge_circle = None
    surf = None
    state = {"px": None, "py": None}

    def _render(corr_grid: np.ndarray, bin_px_val: int):
        nonlocal surf
        if mode.value_selected == "2D":
            ax_corr2d.set_visible(True)
            ax_corr3d.set_visible(False)
            corr_img = np.repeat(np.repeat(corr_grid, bin_px_val, axis=0), bin_px_val, axis=1)
            corr_im.set_data(corr_img[:ny, :nx])
            corr_im.set_clim(-float(s_vlim.val), float(s_vlim.val))
        else:
            ax_corr2d.set_visible(False)
            ax_corr3d.set_visible(True)
            if surf is not None:
                surf.remove()
            x = np.linspace(extent_xd_yd[0], extent_xd_yd[1], corr_grid.shape[1])
            y = np.linspace(extent_xd_yd[3], extent_xd_yd[2], corr_grid.shape[0])
            xx, yy = np.meshgrid(x, y)
            ax_corr3d.clear()
            surf = ax_corr3d.plot_surface(
                xx,
                yy,
                corr_grid,
                cmap="RdBu_r",
                vmin=-float(s_vlim.val),
                vmax=float(s_vlim.val),
                linewidth=0,
                antialiased=False,
            )
            ax_corr3d.set_title("Correlation map (3D surface)")
            ax_corr3d.set_xlabel("x/D")
            ax_corr3d.set_ylabel("y/D")
            ax_corr3d.set_zlabel("corr")

    def _update(*_args):
        nonlocal seed_rect, edge_circle
        if state["px"] is None or state["py"] is None:
            return
        try:
            result, seed_box, corr_grid, bin_px_val, edge_clipped, edge_distance, support_radius = _compute_corr_result(
                movie,
                extent_xd_yd,
                dj_mm,
                fs,
                nx,
                ny,
                state["px"],
                state["py"],
                int(s_patch.val),
                int(s_stride.val),
                int(s_shift.val),
                float(s_rmin.val),
                float(s_weight.val),
                float(s_edge_k.val),
            )
            _render(corr_grid, bin_px_val)

            y0, y1, x0, x1 = seed_box
            x0d, x1d = np.interp([x0, x1], [0, nx - 1], [extent_xd_yd[0], extent_xd_yd[1]])
            y0d, y1d = np.interp([y0, y1], [0, ny - 1], [extent_xd_yd[3], extent_xd_yd[2]])
            if seed_rect is not None:
                seed_rect.remove()
            seed_rect = plt.Rectangle((x0d, min(y0d, y1d)), x1d - x0d, abs(y1d - y0d), fill=False, ec="yellow", lw=1.5)
            ax_img.add_patch(seed_rect)

            circle_params = _edge_circle_params(
                result=result,
                shift=int(s_shift.val),
                weight_power=float(s_weight.val),
                bin_px=int(bin_px_val),
                nx=nx,
                ny=ny,
                extent_xd_yd=extent_xd_yd,
            )
            if edge_circle is not None:
                edge_circle.remove()
                edge_circle = None
            if circle_params is not None:
                cx, cy, radius_x = circle_params
                edge_circle = Circle(
                    (cx, cy),
                    radius_x,
                    fill=False,
                    lw=2.0,
                    ls="--",
                    ec=("red" if edge_clipped else "lime"),
                )
                ax_corr2d.add_patch(edge_circle)


            status_txt.set_text(
                f"Seed=({state['px']:.3f}, {state['py']:.3f}) | bin_px={bin_px_val} | "
                f"used bins={result.n_used_by_shift[int(s_shift.val)]} | edge clipped={edge_clipped} "
                f"[check: d < k*r] (d={edge_distance:.2f}, r={support_radius:.2f}, k={float(s_edge_k.val):.2f}) | "
                f"Ux={result.ux:.3f} m/s, Uy={result.uy:.3f} m/s"
            )
            fig.canvas.draw_idle()
        except Exception as exc:  # pragma: no cover
            status_txt.set_text(f"Error: {exc}")
            fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes != ax_img or event.xdata is None or event.ydata is None:
            return
        px = float(np.clip((event.xdata - extent_xd_yd[0]) / (extent_xd_yd[1] - extent_xd_yd[0]), 0.0, 1.0))
        py = float(np.clip((extent_xd_yd[3] - event.ydata) / (extent_xd_yd[3] - extent_xd_yd[2]), 0.0, 1.0))
        state["px"], state["py"] = px, py
        click_marker.set_data([event.xdata], [event.ydata])
        _update()

    fig.canvas.mpl_connect("button_press_event", _on_click)
    for s in (s_patch, s_stride, s_shift, s_rmin, s_weight, s_edge_k, s_vlim):
        s.on_changed(_update)
    mode.on_clicked(_update)

    plt.show()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for interactive explorers."""
    parser = argparse.ArgumentParser(description="PLIF interactive correlation explorer")
    parser.add_argument(
        "--mode",
        choices=("gui", "notebook"),
        default="gui",
        help="Launch desktop GUI or notebook widget mode (default: gui)",
    )
    parser.add_argument("--source", type=str, default=None, help="TIFF file or folder of TIFF files")
    parser.add_argument("--frames", type=str, default=None, help="Frame slice, e.g. 100:200")
    parser.add_argument("--extent", type=str, default=None, help="Path to extent.txt with: xs xe ys ye")
    parser.add_argument("--fs", type=float, default=None, help="Sample rate in Hz")
    args = parser.parse_args(argv)

    if args.mode == "gui":
        launch_plif_interactive_gui(
            source=args.source,
            frame_range=args.frames,
            extent_file=args.extent,
            fs_hz=args.fs,
        )
    else:
        make_plif_interactive_widget()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
