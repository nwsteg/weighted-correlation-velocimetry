from __future__ import annotations

"""Interactive single-seed correlation explorer for PLIF movies.

This helper is designed for notebook use (Jupyter/VS Code notebooks).
It loads the hardcoded PLIF dataset and provides controls for common WCV
parameters (bin size, stride, shift, thresholds). Click the mean-image panel
to choose a seed location and update the correlation map.
"""

import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from .estimator_single_seed import estimate_single_seed_velocity
from .types import EstimationOptions, GridSpec
from .geometry import frac_to_px_box


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


def make_plif_interactive_widget() -> None:
    """Create and display an interactive seed-click correlation explorer."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover - runtime environment dependent
        raise ImportError(
            "ipywidgets is required for this widget. Install with `pip install ipywidgets`."
        ) from exc

    movie, extent_xd_yd, dj_mm, fs = load_hardcoded_plif()
    _, ny, nx = movie.shape
    mean_img = movie.mean(axis=0)

    # Controls (common WCV knobs + display options)
    patch_px = widgets.IntSlider(value=8, min=2, max=64, step=2, description="PATCH_PX")
    stride = widgets.IntSlider(value=1, min=1, max=8, step=1, description="stride")
    shift = widgets.IntSlider(value=1, min=1, max=20, step=1, description="shift")
    rmin = widgets.FloatSlider(value=0.2, min=0.0, max=0.95, step=0.01, description="rmin")
    weight_power = widgets.FloatSlider(value=2.0, min=0.5, max=5.0, step=0.5, description="weight")
    vlim = widgets.FloatSlider(value=0.5, min=0.05, max=1.0, step=0.05, description="|corr| lim")

    status = widgets.HTML(value="Click the mean image to choose a seed.")

    fig, (ax_img, ax_corr) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax_img.set_title("Mean image (click for seed)")
    ax_img.set_xlabel("x/D")
    ax_img.set_ylabel("y/D")
    ax_corr.set_title("Correlation map")
    ax_corr.set_xlabel("x/D")
    ax_corr.set_ylabel("y/D")

    ax_img.imshow(mean_img, cmap="gray", extent=extent_xd_yd, origin="upper")
    click_marker, = ax_img.plot([], [], "r+", ms=12, mew=2)
    seed_rect = None

    corr_im = ax_corr.imshow(
        np.full_like(mean_img, np.nan, dtype=float),
        cmap="RdBu_r",
        vmin=-vlim.value,
        vmax=vlim.value,
        extent=extent_xd_yd,
        origin="upper",
    )
    plt.colorbar(corr_im, ax=ax_corr, label="corr")

    state = {"px": None, "py": None}

    def _seed_box_from_click(bin_px: int, px_frac: float, py_frac: float) -> tuple[int, int, int, int]:
        return frac_to_px_box(
            px=px_frac,
            py=py_frac,
            w_px=bin_px,
            h_px=bin_px,
            nx=nx,
            ny=ny,
            grid_px=bin_px,
        )

    def _default_bg_boxes(bin_px: int) -> list[tuple[int, int, int, int]]:
        # Keep simple/robust: use 2x2-bin corners as common-mode regressors.
        w = 2 * bin_px
        h = 2 * bin_px
        anchors = [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92)]
        return [frac_to_px_box(px, py, w, h, nx, ny, grid_px=bin_px) for px, py in anchors]

    def _update_corr_map(*_args):
        nonlocal seed_rect
        if state["px"] is None or state["py"] is None:
            return

        try:
            grid = GridSpec(patch_px=int(patch_px.value), grid_stride_patches=int(stride.value))
            bin_px = grid.bin_px
            seed_box = _seed_box_from_click(bin_px, state["px"], state["py"])
            bg_boxes = _default_bg_boxes(bin_px)

            result = estimate_single_seed_velocity(
                movie=movie,
                fs=fs,
                grid=grid,
                seed_box_px=seed_box,
                bg_boxes_px=bg_boxes,
                extent_xd_yd=extent_xd_yd,
                dj_mm=dj_mm,
                shifts=(int(shift.value),),
                options=EstimationOptions(
                    rmin=float(rmin.value),
                    weight_power=float(weight_power.value),
                    min_used=3,
                    require_downstream=False,
                    require_dy_positive=False,
                ),
                allow_bin_padding=True,
            )

            corr_vec = np.asarray(result.corr_by_shift[int(shift.value)], dtype=float)
            corr_grid = corr_vec.reshape(result.by, result.bx)
            corr_img = np.repeat(np.repeat(corr_grid, bin_px, axis=0), bin_px, axis=1)
            corr_im.set_data(corr_img[:ny, :nx])
            corr_im.set_clim(-float(vlim.value), float(vlim.value))

            y0, y1, x0, x1 = seed_box
            x0d, x1d = np.interp([x0, x1], [0, nx - 1], [extent_xd_yd[0], extent_xd_yd[1]])
            y0d, y1d = np.interp([y0, y1], [0, ny - 1], [extent_xd_yd[3], extent_xd_yd[2]])

            if seed_rect is not None:
                seed_rect.remove()
            seed_rect = plt.Rectangle(
                (x0d, min(y0d, y1d)),
                x1d - x0d,
                abs(y1d - y0d),
                fill=False,
                ec="yellow",
                lw=1.5,
            )
            ax_img.add_patch(seed_rect)

            status.value = (
                f"<b>Seed:</b> ({state['px']:.3f}, {state['py']:.3f}) | "
                f"bin_px={bin_px} | used bins={result.n_used_by_shift[int(shift.value)]} | "
                f"Ux={result.ux:.3f} m/s, Uy={result.uy:.3f} m/s"
            )
            fig.canvas.draw_idle()
        except Exception as exc:  # pragma: no cover - interactive diagnostics
            status.value = f"<span style='color:#b00'><b>Error:</b> {exc}</span>"

    def _on_click(event):
        if event.inaxes != ax_img or event.xdata is None or event.ydata is None:
            return

        px = float(np.clip((event.xdata - extent_xd_yd[0]) / (extent_xd_yd[1] - extent_xd_yd[0]), 0.0, 1.0))
        py = float(np.clip((extent_xd_yd[3] - event.ydata) / (extent_xd_yd[3] - extent_xd_yd[2]), 0.0, 1.0))
        state["px"], state["py"] = px, py
        click_marker.set_data([event.xdata], [event.ydata])
        _update_corr_map()

    cid = fig.canvas.mpl_connect("button_press_event", _on_click)
    del cid  # keep linter quiet; connection lives in canvas callback registry.

    for w in (patch_px, stride, shift, rmin, weight_power, vlim):
        w.observe(_update_corr_map, names="value")

    controls = widgets.VBox(
        [
            widgets.HBox([patch_px, stride, shift]),
            widgets.HBox([rmin, weight_power, vlim]),
            status,
        ]
    )

    display(controls)
    display(fig)


