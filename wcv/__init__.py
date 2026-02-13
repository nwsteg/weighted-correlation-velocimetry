"""Weighted-correlation velocimetry package.

Superbinning convention
----------------------
- ``PATCH_PX``: base patch size that must tile the original image.
- ``grid_stride_patches``: number of base patches grouped per side.
- ``BIN_PX = PATCH_PX * grid_stride_patches``: effective region size used by
  correlation, centroid geometry, masks, and seed iteration.

This allows larger analysis cells without cropping and without changing the
original imshow extent mapping.
"""

from .estimator_map import (
    estimate_velocity_map,
    estimate_velocity_map_hybrid,
    estimate_velocity_map_streaming,
)
from .bootstrap import (
    BootstrapSummary,
    VelocityBootstrapResult,
    bootstrap_velocity_map,
    default_block_length,
    moving_block_bootstrap_indices,
)
from .parameter_ensemble import (
    ParameterEnsembleResult,
    ParameterEnsembleStats,
    plot_parameter_ensemble_summary,
    run_parameter_ensemble_uncertainty,
    sample_parameter_ensemble,
    save_parameter_ensemble_bundle,
)
from .estimator_single_seed import estimate_single_seed_velocity
from .interactive import load_hardcoded_plif, make_plif_interactive_widget
from .legacy import draw_boxes_debug, estimate_velocity_per_shift_framework
from .types import EstimationOptions, GridSpec, SingleSeedResult, VelocityMapResult


def load_hardcoded_plif():
    from .interactive import load_hardcoded_plif as _impl

    return _impl()


def make_plif_interactive_widget() -> None:
    from .interactive import make_plif_interactive_widget as _impl

    return _impl()


def launch_plif_interactive_gui() -> None:
    from .interactive import launch_plif_interactive_gui as _impl

    return _impl()


__all__ = [
    "estimate_single_seed_velocity",
    "estimate_velocity_map",
    "estimate_velocity_map_streaming",
    "estimate_velocity_map_hybrid",
    "bootstrap_velocity_map",
    "moving_block_bootstrap_indices",
    "default_block_length",
    "BootstrapSummary",
    "VelocityBootstrapResult",
    "GridSpec",
    "EstimationOptions",
    "SingleSeedResult",
    "VelocityMapResult",
    "estimate_velocity_per_shift_framework",
    "draw_boxes_debug",
    "load_hardcoded_plif",
    "make_plif_interactive_widget",
    "launch_plif_interactive_gui",
    "ParameterEnsembleStats",
    "ParameterEnsembleResult",
    "sample_parameter_ensemble",
    "run_parameter_ensemble_uncertainty",
    "save_parameter_ensemble_bundle",
    "plot_parameter_ensemble_summary",
]
