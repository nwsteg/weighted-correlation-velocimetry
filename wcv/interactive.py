from __future__ import annotations

"""Interactive single-seed correlation explorers for PLIF movies.

Two entrypoints are provided:
- `make_plif_interactive_widget()` for notebook use (ipywidgets).
- `launch_plif_interactive_gui()` for command-line desktop GUI use (matplotlib widgets).
"""

import argparse
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection side effect

from .estimator_single_seed import estimate_single_seed_velocity
from .geometry import frac_to_px_box
from .types import EstimationOptions, GridSpec


@dataclass(frozen=True)
class _DatasetConfig:
    module_path: str = "G:/labcode"
    aux_dir: str = r"G:/2025 JICF4 Simul PLIF Schlieren/FST1060/plif/plif_aux"
    name: str = "plif"
    dj_mm: float = 3.5
    fs_hz: float = 50_000.0


def load_hardcoded_plif() -> tuple[np.ndarray, tuple[float, float, float, float], float, float]:
    """Load and normalize the hardcoded PLIF dataset from the user snippet."""
    cfg = _DatasetConfig()

    if cfg.module_path not in sys.path:
        sys.path.append(cfg.module_path)

    from image_dataset import ImageDataset  # pylint: disable=import-error

    plif = ImageDataset(name=cfg.name, aux_dir=cfg.aux_dir)

    bgs = plif.images[100:-10]
    bgs = bgs / np.max(bgs)
    bgs_f = bgs.astype(np.float32, copy=False)

    extent = tuple((np.asarray(plif.extent) / cfg.dj_mm).astype(float))
    return bgs_f, extent, cfg.dj_mm, cfg.fs_hz


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
        ),
        allow_bin_padding=True,
    )
    corr_vec = np.asarray(result.corr_by_shift[int(shift)], dtype=float)
    corr_grid = corr_vec.reshape(result.by, result.bx)
    return result, seed_box, corr_grid, bin_px


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
    surf = None

    corr_im = ax_corr2d.imshow(
        np.full_like(mean_img, np.nan, dtype=float),
        cmap="RdBu_r",
        vmin=-vlim.value,
        vmax=vlim.value,
        extent=extent_xd_yd,
        origin="upper",
    )
    plt.colorbar(corr_im, ax=ax_corr2d, label="corr")

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
        nonlocal seed_rect
        if state["px"] is None or state["py"] is None:
            return
        try:
            result, seed_box, corr_grid, bin_px_val = _compute_corr_result(
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
            )
            _render(corr_grid, bin_px_val)

            y0, y1, x0, x1 = seed_box
            x0d, x1d = np.interp([x0, x1], [0, nx - 1], [extent_xd_yd[0], extent_xd_yd[1]])
            y0d, y1d = np.interp([y0, y1], [0, ny - 1], [extent_xd_yd[3], extent_xd_yd[2]])
            if seed_rect is not None:
                seed_rect.remove()
            seed_rect = plt.Rectangle((x0d, min(y0d, y1d)), x1d - x0d, abs(y1d - y0d), fill=False, ec="yellow", lw=1.5)
            ax_img.add_patch(seed_rect)

            status.value = (
                f"<b>Seed:</b> ({state['px']:.3f}, {state['py']:.3f}) | "
                f"bin_px={bin_px_val} | used bins={result.n_used_by_shift[int(shift.value)]} | "
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
    for w in (patch_px, stride, shift, rmin, weight_power, vlim, view_mode):
        w.observe(_update, names="value")

    controls = widgets.VBox([widgets.HBox([patch_px, stride, shift]), widgets.HBox([rmin, weight_power, vlim, view_mode]), status])
    display(controls)
    display(fig)


def launch_plif_interactive_gui() -> None:
    """Launch a command-line desktop GUI with 2D/3D correlation-map toggle."""
    movie, extent_xd_yd, dj_mm, fs = load_hardcoded_plif()
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
    plt.colorbar(corr_im, ax=ax_corr2d, label="corr")

    ax_patch = fig.add_axes([0.08, 0.14, 0.30, 0.03])
    ax_stride = fig.add_axes([0.08, 0.10, 0.30, 0.03])
    ax_shift = fig.add_axes([0.08, 0.06, 0.30, 0.03])
    ax_rmin = fig.add_axes([0.56, 0.14, 0.30, 0.03])
    ax_weight = fig.add_axes([0.56, 0.10, 0.30, 0.03])
    ax_vlim = fig.add_axes([0.56, 0.06, 0.30, 0.03])
    ax_mode = fig.add_axes([0.40, 0.04, 0.12, 0.12])

    s_patch = Slider(ax_patch, "PATCH_PX", 2, 64, valinit=8, valstep=2)
    s_stride = Slider(ax_stride, "stride", 1, 8, valinit=1, valstep=1)
    s_shift = Slider(ax_shift, "shift", 1, 20, valinit=1, valstep=1)
    s_rmin = Slider(ax_rmin, "rmin", 0.0, 0.95, valinit=0.2, valstep=0.01)
    s_weight = Slider(ax_weight, "weight", 0.5, 5.0, valinit=2.0, valstep=0.5)
    s_vlim = Slider(ax_vlim, "|corr| lim", 0.05, 1.0, valinit=0.5, valstep=0.05)
    mode = RadioButtons(ax_mode, ("2D", "3D"), active=0)

    status_txt = fig.text(0.05, 0.01, "Click the mean image to choose a seed.", fontsize=10)
    seed_rect = None
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
        nonlocal seed_rect
        if state["px"] is None or state["py"] is None:
            return
        try:
            result, seed_box, corr_grid, bin_px_val = _compute_corr_result(
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
            )
            _render(corr_grid, bin_px_val)

            y0, y1, x0, x1 = seed_box
            x0d, x1d = np.interp([x0, x1], [0, nx - 1], [extent_xd_yd[0], extent_xd_yd[1]])
            y0d, y1d = np.interp([y0, y1], [0, ny - 1], [extent_xd_yd[3], extent_xd_yd[2]])
            if seed_rect is not None:
                seed_rect.remove()
            seed_rect = plt.Rectangle((x0d, min(y0d, y1d)), x1d - x0d, abs(y1d - y0d), fill=False, ec="yellow", lw=1.5)
            ax_img.add_patch(seed_rect)

            status_txt.set_text(
                f"Seed=({state['px']:.3f}, {state['py']:.3f}) | bin_px={bin_px_val} | "
                f"used bins={result.n_used_by_shift[int(s_shift.val)]} | Ux={result.ux:.3f} m/s, Uy={result.uy:.3f} m/s"
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
    for s in (s_patch, s_stride, s_shift, s_rmin, s_weight, s_vlim):
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
    args = parser.parse_args(argv)

    if args.mode == "gui":
        launch_plif_interactive_gui()
    else:
        make_plif_interactive_widget()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
