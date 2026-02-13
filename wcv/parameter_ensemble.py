from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from .bootstrap import VelocityBootstrapResult
from .estimator_map import estimate_velocity_map_streaming
from .types import EstimationOptions, VelocityMapResult

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParameterEnsembleStats:
    mean: np.ndarray
    std: np.ndarray
    median: np.ndarray
    iqr: np.ndarray
    p2_5: np.ndarray
    p97_5: np.ndarray
    relative_std: np.ndarray


@dataclass(frozen=True)
class ParameterEnsembleResult:
    ux: ParameterEnsembleStats
    uy: ParameterEnsembleStats
    coverage_map: np.ndarray
    valid_mask: np.ndarray
    ux_runs: np.ndarray
    uy_runs: np.ndarray
    valid_seed_count: np.ndarray
    sampled_rmin: np.ndarray
    sampled_min_used: np.ndarray
    sampled_weight_power: np.ndarray
    sampled_edge_clip_reject_k: np.ndarray
    sampled_edge_clip_is_none: np.ndarray
    n_param: int
    seed: int
    min_valid_fraction: float
    sigma_param_ux: np.ndarray
    sigma_param_uy: np.ndarray
    sigma_boot_ux: np.ndarray | None = None
    sigma_boot_uy: np.ndarray | None = None
    sigma_total_ux: np.ndarray | None = None
    sigma_total_uy: np.ndarray | None = None
    domain_summary: dict[str, float] | None = None
    line_profile_summary: dict[str, np.ndarray] | None = None


def _latin_hypercube_unit(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    try:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=n_dim, seed=int(rng.integers(0, 2**31 - 1)))
        return sampler.random(n_samples)
    except Exception:
        u = np.empty((n_samples, n_dim), dtype=np.float64)
        for d in range(n_dim):
            bins = (np.arange(n_samples, dtype=np.float64) + rng.random(n_samples)) / float(n_samples)
            rng.shuffle(bins)
            u[:, d] = bins
        return u


def sample_parameter_ensemble(
    n_param: int,
    *,
    seed: int,
    edge_clip_none_fraction: float = 0.1,
    rmin_bounds: tuple[float, float] = (0.2, 0.5),
    min_used_bounds: tuple[int, int] = (10, 60),
    weight_power_bounds: tuple[float, float] = (1.0, 3.0),
    edge_clip_bounds: tuple[float, float] = (0.5, 2.0),
) -> dict[str, np.ndarray]:
    if int(n_param) <= 0:
        raise ValueError("n_param must be > 0")
    if not (0.0 <= float(edge_clip_none_fraction) < 1.0):
        raise ValueError("edge_clip_none_fraction must be in [0, 1)")

    rng = np.random.default_rng(seed)
    u = _latin_hypercube_unit(int(n_param), 4, rng)

    rmin = rmin_bounds[0] + u[:, 0] * (rmin_bounds[1] - rmin_bounds[0])
    min_used = min_used_bounds[0] + u[:, 1] * (min_used_bounds[1] - min_used_bounds[0])
    min_used = np.rint(min_used).astype(np.int32)
    min_used = np.clip(min_used, min_used_bounds[0], min_used_bounds[1])

    weight_power = weight_power_bounds[0] + u[:, 2] * (weight_power_bounds[1] - weight_power_bounds[0])
    edge_clip = edge_clip_bounds[0] + u[:, 3] * (edge_clip_bounds[1] - edge_clip_bounds[0])

    edge_clip_is_none = rng.random(int(n_param)) < float(edge_clip_none_fraction)
    edge_clip = edge_clip.astype(np.float32)
    edge_clip[edge_clip_is_none] = np.nan

    return {
        "rmin": rmin.astype(np.float32),
        "min_used": min_used,
        "weight_power": weight_power.astype(np.float32),
        "edge_clip_reject_k": edge_clip,
        "edge_clip_is_none": edge_clip_is_none,
    }


def _compute_stats(samples: np.ndarray, valid_mask: np.ndarray, eps: float) -> ParameterEnsembleStats:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mean = np.nanmean(samples, axis=0)
        std = np.nanstd(samples, axis=0, ddof=1)
        median = np.nanmedian(samples, axis=0)
        q25 = np.nanpercentile(samples, 25.0, axis=0)
        q75 = np.nanpercentile(samples, 75.0, axis=0)
        p2_5 = np.nanpercentile(samples, 2.5, axis=0)
        p97_5 = np.nanpercentile(samples, 97.5, axis=0)

    iqr = q75 - q25
    relative = std / np.maximum(np.abs(mean), float(eps))

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    median = median.astype(np.float32)
    iqr = iqr.astype(np.float32)
    p2_5 = p2_5.astype(np.float32)
    p97_5 = p97_5.astype(np.float32)
    relative = relative.astype(np.float32)

    mean[~valid_mask] = np.nan
    std[~valid_mask] = np.nan
    median[~valid_mask] = np.nan
    iqr[~valid_mask] = np.nan
    p2_5[~valid_mask] = np.nan
    p97_5[~valid_mask] = np.nan
    relative[~valid_mask] = np.nan

    return ParameterEnsembleStats(
        mean=mean,
        std=std,
        median=median,
        iqr=iqr,
        p2_5=p2_5,
        p97_5=p97_5,
        relative_std=relative,
    )


def _line_profiles(field: np.ndarray) -> dict[str, np.ndarray]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        by_x = np.nanmedian(field, axis=0).astype(np.float32)
        by_y = np.nanmedian(field, axis=1).astype(np.float32)
    return {"median_over_y": by_x, "median_over_x": by_y}


def run_parameter_ensemble_uncertainty(
    movie: np.ndarray,
    *,
    estimator_kwargs: dict[str, Any],
    baseline_options: EstimationOptions,
    bootstrap_result: VelocityBootstrapResult | None = None,
    estimator: Any = estimate_velocity_map_streaming,
    n_param: int = 120,
    seed: int = 0,
    edge_clip_none_fraction: float = 0.1,
    min_valid_fraction: float = 0.8,
    eps: float = 1e-6,
    n_jobs: int = 1,
    progress_every: int = 10,
) -> ParameterEnsembleResult:
    if not (0.0 <= float(min_valid_fraction) <= 1.0):
        raise ValueError("min_valid_fraction must be in [0, 1]")
    sampled = sample_parameter_ensemble(
        int(n_param),
        seed=int(seed),
        edge_clip_none_fraction=float(edge_clip_none_fraction),
    )

    est_kwargs = dict(estimator_kwargs)

    def _single_run(i: int) -> tuple[np.ndarray, np.ndarray, int]:
        edge_k = None if bool(sampled["edge_clip_is_none"][i]) else float(sampled["edge_clip_reject_k"][i])
        opts = EstimationOptions(
            rmin=float(sampled["rmin"][i]),
            min_used=int(sampled["min_used"][i]),
            require_downstream=baseline_options.require_downstream,
            require_dy_positive=baseline_options.require_dy_positive,
            weight_power=float(sampled["weight_power"][i]),
            allow_bin_padding=baseline_options.allow_bin_padding,
            padding_mode=baseline_options.padding_mode,
            sparse_corr_storage=baseline_options.sparse_corr_storage,
            sparse_top_k=baseline_options.sparse_top_k,
            edge_clip_reject_k=edge_k,
            edge_clip_sigma_mult=baseline_options.edge_clip_sigma_mult,
        )
        local_kwargs = dict(est_kwargs)
        local_kwargs["options"] = opts
        res: VelocityMapResult = estimator(movie=movie, **local_kwargs)
        return (
            np.asarray(res.ux_map, dtype=np.float32),
            np.asarray(res.uy_map, dtype=np.float32),
            int(res.valid_seed_count),
        )

    first_ux, first_uy, first_count = _single_run(0)
    shape = first_ux.shape
    ux_runs = np.full((int(n_param),) + shape, np.nan, dtype=np.float32)
    uy_runs = np.full((int(n_param),) + shape, np.nan, dtype=np.float32)
    valid_seed_count = np.zeros(int(n_param), dtype=np.int32)
    ux_runs[0] = first_ux
    uy_runs[0] = first_uy
    valid_seed_count[0] = first_count

    remaining = list(range(1, int(n_param)))
    if int(n_jobs) > 1 and remaining:
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            for done, (idx, out) in enumerate(zip(remaining, ex.map(_single_run, remaining)), start=2):
                ux, uy, cnt = out
                ux_runs[idx] = ux
                uy_runs[idx] = uy
                valid_seed_count[idx] = cnt
                if progress_every > 0 and (done % int(progress_every) == 0 or done == int(n_param)):
                    LOGGER.info("Parameter ensemble %d/%d", done, int(n_param))
    else:
        for done, idx in enumerate(remaining, start=2):
            ux, uy, cnt = _single_run(idx)
            ux_runs[idx] = ux
            uy_runs[idx] = uy
            valid_seed_count[idx] = cnt
            if progress_every > 0 and (done % int(progress_every) == 0 or done == int(n_param)):
                LOGGER.info("Parameter ensemble %d/%d", done, int(n_param))

    finite_joint = np.isfinite(ux_runs) & np.isfinite(uy_runs)
    coverage = finite_joint.mean(axis=0).astype(np.float32)
    valid_mask = coverage >= float(min_valid_fraction)

    ux_stats = _compute_stats(ux_runs, valid_mask, eps)
    uy_stats = _compute_stats(uy_runs, valid_mask, eps)

    sigma_param_ux = ux_stats.std
    sigma_param_uy = uy_stats.std

    sigma_boot_ux = bootstrap_result.ux.std.astype(np.float32) if bootstrap_result is not None else None
    sigma_boot_uy = bootstrap_result.uy.std.astype(np.float32) if bootstrap_result is not None else None

    sigma_total_ux = None
    sigma_total_uy = None
    if sigma_boot_ux is not None:
        sigma_total_ux = np.sqrt(np.square(sigma_param_ux) + np.square(sigma_boot_ux)).astype(np.float32)
    if sigma_boot_uy is not None:
        sigma_total_uy = np.sqrt(np.square(sigma_param_uy) + np.square(sigma_boot_uy)).astype(np.float32)

    domain_summary = None
    line_profiles = None
    if sigma_boot_ux is not None and sigma_boot_uy is not None and sigma_total_ux is not None and sigma_total_uy is not None:
        cell_mask = valid_mask & np.isfinite(sigma_boot_ux) & np.isfinite(sigma_boot_uy)
        domain_summary = {
            "sigma_param_ux_median": float(np.nanmedian(sigma_param_ux[cell_mask])),
            "sigma_boot_ux_median": float(np.nanmedian(sigma_boot_ux[cell_mask])),
            "sigma_total_ux_median": float(np.nanmedian(sigma_total_ux[cell_mask])),
            "sigma_param_uy_median": float(np.nanmedian(sigma_param_uy[cell_mask])),
            "sigma_boot_uy_median": float(np.nanmedian(sigma_boot_uy[cell_mask])),
            "sigma_total_uy_median": float(np.nanmedian(sigma_total_uy[cell_mask])),
        }
        line_profiles = {
            "sigma_param_ux": _line_profiles(sigma_param_ux),
            "sigma_boot_ux": _line_profiles(sigma_boot_ux),
            "sigma_total_ux": _line_profiles(sigma_total_ux),
            "sigma_param_uy": _line_profiles(sigma_param_uy),
            "sigma_boot_uy": _line_profiles(sigma_boot_uy),
            "sigma_total_uy": _line_profiles(sigma_total_uy),
        }

    return ParameterEnsembleResult(
        ux=ux_stats,
        uy=uy_stats,
        coverage_map=coverage,
        valid_mask=valid_mask,
        ux_runs=ux_runs,
        uy_runs=uy_runs,
        valid_seed_count=valid_seed_count,
        sampled_rmin=sampled["rmin"],
        sampled_min_used=sampled["min_used"],
        sampled_weight_power=sampled["weight_power"],
        sampled_edge_clip_reject_k=sampled["edge_clip_reject_k"],
        sampled_edge_clip_is_none=sampled["edge_clip_is_none"],
        n_param=int(n_param),
        seed=int(seed),
        min_valid_fraction=float(min_valid_fraction),
        sigma_param_ux=sigma_param_ux,
        sigma_param_uy=sigma_param_uy,
        sigma_boot_ux=sigma_boot_ux,
        sigma_boot_uy=sigma_boot_uy,
        sigma_total_ux=sigma_total_ux,
        sigma_total_uy=sigma_total_uy,
        domain_summary=domain_summary,
        line_profile_summary=line_profiles,
    )


def save_parameter_ensemble_bundle(result: ParameterEnsembleResult, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ux_mean": result.ux.mean,
        "ux_std": result.ux.std,
        "ux_median": result.ux.median,
        "ux_iqr": result.ux.iqr,
        "ux_p2_5": result.ux.p2_5,
        "ux_p97_5": result.ux.p97_5,
        "ux_relative_std": result.ux.relative_std,
        "uy_mean": result.uy.mean,
        "uy_std": result.uy.std,
        "uy_median": result.uy.median,
        "uy_iqr": result.uy.iqr,
        "uy_p2_5": result.uy.p2_5,
        "uy_p97_5": result.uy.p97_5,
        "uy_relative_std": result.uy.relative_std,
        "coverage_map": result.coverage_map,
        "valid_mask": result.valid_mask,
        "ux_runs": result.ux_runs,
        "uy_runs": result.uy_runs,
        "valid_seed_count": result.valid_seed_count,
        "sampled_rmin": result.sampled_rmin,
        "sampled_min_used": result.sampled_min_used,
        "sampled_weight_power": result.sampled_weight_power,
        "sampled_edge_clip_reject_k": result.sampled_edge_clip_reject_k,
        "sampled_edge_clip_is_none": result.sampled_edge_clip_is_none,
        "n_param": np.asarray([result.n_param], dtype=np.int32),
        "seed": np.asarray([result.seed], dtype=np.int64),
        "min_valid_fraction": np.asarray([result.min_valid_fraction], dtype=np.float32),
        "sigma_param_ux": result.sigma_param_ux,
        "sigma_param_uy": result.sigma_param_uy,
    }
    if result.sigma_boot_ux is not None:
        payload["sigma_boot_ux"] = result.sigma_boot_ux
    if result.sigma_boot_uy is not None:
        payload["sigma_boot_uy"] = result.sigma_boot_uy
    if result.sigma_total_ux is not None:
        payload["sigma_total_ux"] = result.sigma_total_ux
    if result.sigma_total_uy is not None:
        payload["sigma_total_uy"] = result.sigma_total_uy

    np.savez_compressed(out, **payload)
    return out


def plot_parameter_ensemble_summary(
    result: ParameterEnsembleResult,
    *,
    component: str = "ux",
    cmap: str = "viridis",
):
    import matplotlib.pyplot as plt

    if component not in {"ux", "uy"}:
        raise ValueError("component must be 'ux' or 'uy'")

    stats = result.ux if component == "ux" else result.uy
    sigma_param = result.sigma_param_ux if component == "ux" else result.sigma_param_uy
    sigma_boot = result.sigma_boot_ux if component == "ux" else result.sigma_boot_uy

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), constrained_layout=True)

    im0 = axes[0].imshow(stats.mean, cmap=cmap)
    axes[0].set_title(f"{component} mean")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(stats.std, cmap=cmap)
    axes[1].set_title(f"{component} std")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(result.coverage_map, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("coverage")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(sigma_param, cmap=cmap)
    axes[3].set_title(f"sigma_param ({component})")
    fig.colorbar(im3, ax=axes[3], fraction=0.046)

    if sigma_boot is not None:
        im4 = axes[4].imshow(sigma_boot, cmap=cmap)
        axes[4].set_title(f"sigma_boot ({component})")
        fig.colorbar(im4, ax=axes[4], fraction=0.046)
    else:
        axes[4].axis("off")
        axes[4].set_title("sigma_boot unavailable")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, axes
