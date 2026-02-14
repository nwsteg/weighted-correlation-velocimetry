#!/usr/bin/env python3
"""Profile WCV map estimators across bin sizes.

This script compares runtime/memory trade-offs across estimator modes:
- materialized (`estimate_velocity_map`)
- streaming (`estimate_velocity_map_streaming`)
- hybrid (streaming with configurable chunking, when supported by API)

Examples
--------
# Synthetic benchmark
python scripts/profile_velocity_map.py \
  --mode all \
  --shape 200,1024,1024 \
  --bin-sizes 4,8,16,32 \
  --shifts 1,2 \
  --repeat 3 \
  --hybrid-chunk-size 256

# Real data saved as .npy (nt, ny, nx)
python scripts/profile_velocity_map.py \
  --movie-npy /path/to/movie.npy \
  --extent 0,6,-1,7 \
  --mode all \
  --bin-sizes 4,8,16,32 \
  --shifts 1 \
  --csv-out profile_results.csv
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import io
import pstats
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wcv
from wcv import EstimationOptions, GridSpec
from wcv.geometry import frac_to_px_box


@dataclass
class RunSummary:
    estimator: str
    bin_px: int
    runtime_s: float
    peak_mem_mb: float
    valid_seed_count: int
    total_seed_count: int
    n_regions: int
    nt: int
    ny: int
    nx: int
    shift_count: int


@dataclass
class ProfileSummary:
    estimator: str
    bin_px: int
    runtime_s: float
    runtime_median_s: float
    runtime_std_s: float
    peak_mem_mb: float
    valid_seed_count: int
    total_seed_count: int
    valid_frac: float
    n_regions: int
    nt: int
    ny: int
    nx: int
    shift_count: int
    seeds_per_s: float
    corr_pairs: int
    corr_pairs_per_s: float


@dataclass
class EstimatorSpec:
    label: str
    fn_name: str
    chunk_size: int | None


def parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in text.split(",") if v.strip())


def parse_csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(v.strip()) for v in text.split(",") if v.strip())


def load_movie(movie_npy: str | None, shape: tuple[int, int, int], seed: int) -> np.ndarray:
    if movie_npy:
        arr = np.load(movie_npy)
        if arr.ndim != 3:
            raise ValueError(f"movie must be 3D (nt, ny, nx). Got {arr.shape}")
        return np.asarray(arr, dtype=np.float32)

    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=np.float32)


def call_estimator(estimator: EstimatorSpec, fn: Any, **kwargs: Any) -> Any:
    if estimator.label == "materialized":
        kwargs.pop("seed_chunk_size", None)
    return fn(**kwargs)


def make_bg_boxes(
    ny: int,
    nx: int,
    bin_px: int,
    box_size_patches: tuple[int, int],
    anchors: tuple[tuple[float, float], ...],
) -> list[tuple[int, int, int, int]]:
    bg_w_px = box_size_patches[0] * bin_px
    bg_h_px = box_size_patches[1] * bin_px
    return [
        frac_to_px_box(
            px=ax,
            py=ay,
            w_px=bg_w_px,
            h_px=bg_h_px,
            nx=nx,
            ny=ny,
            grid_px=bin_px,
        )
        for ax, ay in anchors
    ]


def _print_cprofile_stats(profiler: cProfile.Profile, estimator: str, bin_px: int, top: int) -> None:
    print(f"\n--- cProfile (cumtime): {estimator}, bin={bin_px} ---")
    s_cum = io.StringIO()
    pstats.Stats(profiler, stream=s_cum).sort_stats("cumtime").print_stats(top)
    print(s_cum.getvalue())

    print(f"--- cProfile (tottime): {estimator}, bin={bin_px} ---")
    s_tot = io.StringIO()
    pstats.Stats(profiler, stream=s_tot).sort_stats("tottime").print_stats(top)
    print(s_tot.getvalue())


def profile_once(
    *,
    estimator: EstimatorSpec,
    movie: np.ndarray,
    fs: float,
    extent: tuple[float, float, float, float],
    dj_mm: float,
    base_patch_px: int,
    bin_px: int,
    shifts: tuple[int, ...],
    options: EstimationOptions,
    use_shear_mask: bool,
    show_progress: bool,
    do_cprofile: bool,
    profile_top: int,
) -> RunSummary:
    stride = bin_px // base_patch_px
    grid = GridSpec(patch_px=base_patch_px, grid_stride_patches=stride)
    _, ny, nx = movie.shape

    ny_pad = ((ny + bin_px - 1) // bin_px) * bin_px
    nx_pad = ((nx + bin_px - 1) // bin_px) * bin_px
    bg_boxes_px = make_bg_boxes(
        ny=ny_pad,
        nx=nx_pad,
        bin_px=bin_px,
        box_size_patches=(1, 1),
        anchors=((0.10, 0.10), (0.25, 0.96), (0.90, 0.80)),
    )

    if not hasattr(wcv, estimator.fn_name):
        raise ValueError(f"Unknown estimator function on wcv module: {estimator.fn_name}")
    fn = getattr(wcv, estimator.fn_name)

    kwargs = dict(
        movie=movie,
        fs=fs,
        grid=grid,
        bg_boxes_px=bg_boxes_px,
        extent_xd_yd=extent,
        dj_mm=dj_mm,
        shifts=shifts,
        options=options,
        allow_bin_padding=True,
        use_shear_mask=use_shear_mask,
        shear_mask_px=None,
        shear_pctl=50,
        origin="upper",
        detrend_type="linear",
        show_progress=show_progress,
        chunk_size=estimator.chunk_size,
        seed_chunk_size=estimator.chunk_size,
        return_corr_by_shift=False,
        store_corr_by_shift=False,
    )

    profiler: cProfile.Profile | None = cProfile.Profile() if do_cprofile else None
    tracemalloc.start()
    t0 = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    vm = call_estimator(estimator, fn, **kwargs)
    if profiler is not None:
        profiler.disable()
    runtime_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if profiler is not None:
        _print_cprofile_stats(profiler, estimator.label, bin_px, profile_top)

    n_regions = int(vm.by * vm.bx)
    return RunSummary(
            estimator=estimator.label,
            bin_px=bin_px,
            runtime_s=runtime_s,
            peak_mem_mb=peak / (1024 * 1024),
            valid_seed_count=int(vm.valid_seed_count),
            total_seed_count=int(vm.total_seed_count),
            n_regions=n_regions,
            nt=int(movie.shape[0]),
            ny=int(movie.shape[1]),
            nx=int(movie.shape[2]),
            shift_count=len(shifts),
        )


def summarize_runs(runs: list[RunSummary]) -> ProfileSummary:
    if not runs:
        raise ValueError("runs cannot be empty")
    best = min(runs, key=lambda r: r.runtime_s)
    runtime_arr = np.asarray([r.runtime_s for r in runs], dtype=np.float64)
    valid_frac = 0.0 if best.total_seed_count == 0 else best.valid_seed_count / best.total_seed_count
    seeds_per_s = 0.0 if best.runtime_s <= 0 else best.total_seed_count / best.runtime_s
    corr_pairs = int(best.n_regions * best.total_seed_count * best.shift_count)
    corr_pairs_per_s = 0.0 if best.runtime_s <= 0 else corr_pairs / best.runtime_s

    return ProfileSummary(
        estimator=best.estimator,
        bin_px=best.bin_px,
        runtime_s=float(best.runtime_s),
        runtime_median_s=float(np.median(runtime_arr)),
        runtime_std_s=float(np.std(runtime_arr, ddof=0)),
        peak_mem_mb=float(best.peak_mem_mb),
        valid_seed_count=int(best.valid_seed_count),
        total_seed_count=int(best.total_seed_count),
        valid_frac=float(valid_frac),
        n_regions=int(best.n_regions),
        nt=int(best.nt),
        ny=int(best.ny),
        nx=int(best.nx),
        shift_count=int(best.shift_count),
        seeds_per_s=float(seeds_per_s),
        corr_pairs=int(corr_pairs),
        corr_pairs_per_s=float(corr_pairs_per_s),
    )


def print_relative_summary(rows: list[ProfileSummary]) -> None:
    grouped: dict[int, list[ProfileSummary]] = {}
    for row in rows:
        grouped.setdefault(row.bin_px, []).append(row)

    print("\nRelative comparison (per bin):")
    for bin_px in sorted(grouped):
        items = grouped[bin_px]
        baseline = next((r for r in items if r.estimator == "materialized"), None)
        if baseline is None:
            baseline = min(items, key=lambda r: r.runtime_s)
        print(f"  bin={bin_px}: baseline={baseline.estimator}")
        for r in sorted(items, key=lambda x: x.runtime_s):
            speedup = baseline.runtime_s / r.runtime_s if r.runtime_s > 0 else np.nan
            mem_ratio = r.peak_mem_mb / baseline.peak_mem_mb if baseline.peak_mem_mb > 0 else np.nan
            print(
                f"    {r.estimator:<12} time={r.runtime_s:8.3f}s "
                f"speedup_vs_base={speedup:6.2f}x "
                f"peak={r.peak_mem_mb:8.1f}MB mem_vs_base={mem_ratio:6.2f}x"
            )


def write_csv(path: Path, rows: list[ProfileSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "estimator",
                "bin_px",
                "runtime_s",
                "runtime_median_s",
                "runtime_std_s",
                "peak_mem_mb",
                "valid_seed_count",
                "total_seed_count",
                "valid_frac",
                "n_regions",
                "nt",
                "ny",
                "nx",
                "shift_count",
                "seeds_per_s",
                "corr_pairs",
                "corr_pairs_per_s",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.estimator,
                    r.bin_px,
                    f"{r.runtime_s:.6f}",
                    f"{r.runtime_median_s:.6f}",
                    f"{r.runtime_std_s:.6f}",
                    f"{r.peak_mem_mb:.3f}",
                    r.valid_seed_count,
                    r.total_seed_count,
                    f"{r.valid_frac:.6f}",
                    r.n_regions,
                    r.nt,
                    r.ny,
                    r.nx,
                    r.shift_count,
                    f"{r.seeds_per_s:.3f}",
                    r.corr_pairs,
                    f"{r.corr_pairs_per_s:.1f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile WCV velocity-map estimators")
    p.add_argument("--movie-npy", type=str, default=None, help="Path to .npy movie (nt, ny, nx)")
    p.add_argument("--shape", type=str, default="80,256,256", help="Synthetic shape nt,ny,nx")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic movie")
    p.add_argument("--extent", type=str, default="0,1,0,1", help="extent xmin,xmax,ymin,ymax")
    p.add_argument("--fs", type=float, default=50_000.0, help="Sampling rate")
    p.add_argument("--dj-mm", type=float, default=1.0, help="Dj in mm")
    p.add_argument("--base-patch-px", type=int, default=2)
    p.add_argument("--bin-sizes", type=str, default="4,8,16,32")
    p.add_argument("--shifts", type=str, default="1")
    p.add_argument(
        "--mode",
        choices=["materialized", "streaming", "hybrid", "both", "all"],
        default="all",
    )
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--hybrid-chunk-size", type=int, default=256)
    p.add_argument("--streaming-chunk-size", type=int, default=None)
    p.add_argument("--use-shear-mask", action="store_true")
    p.add_argument("--show-progress", action="store_true")
    p.add_argument("--rmin", type=float, default=0.3)
    p.add_argument("--min-used", type=int, default=15)
    p.add_argument("--weight-power", type=float, default=2.0)
    p.add_argument("--do-cprofile", action="store_true")
    p.add_argument("--profile-top", type=int, default=20)
    p.add_argument("--csv-out", type=str, default=None, help="Optional CSV output path")
    return p.parse_args()


def resolve_estimators(args: argparse.Namespace) -> list[EstimatorSpec]:
    all_specs = {
        "materialized": EstimatorSpec("materialized", "estimate_velocity_map", None),
        "streaming": EstimatorSpec("streaming", "estimate_velocity_map_streaming", args.streaming_chunk_size),
        "hybrid": EstimatorSpec("hybrid", "estimate_velocity_map_streaming", args.hybrid_chunk_size),
    }
    if args.mode in {"all", "both"}:
        return [all_specs["materialized"], all_specs["streaming"], all_specs["hybrid"]]
    return [all_specs[args.mode]]


def main() -> None:
    args = parse_args()

    shape = parse_csv_ints(args.shape)
    if len(shape) != 3:
        raise ValueError("--shape must have 3 values: nt,ny,nx")
    extent = parse_csv_floats(args.extent)
    if len(extent) != 4:
        raise ValueError("--extent must have 4 values: xmin,xmax,ymin,ymax")

    bin_sizes = parse_csv_ints(args.bin_sizes)
    shifts = parse_csv_ints(args.shifts)
    if not shifts:
        raise ValueError("At least one positive shift is required")

    movie = load_movie(args.movie_npy, shape, args.seed)
    movie = np.asarray(movie, dtype=np.float32, copy=False)

    opts = EstimationOptions(
        rmin=float(args.rmin),
        min_used=int(args.min_used),
        require_downstream=True,
        require_dy_positive=False,
        weight_power=float(args.weight_power),
        allow_bin_padding=True,
        padding_mode="edge",
    )

    estimators = resolve_estimators(args)

    rows: list[ProfileSummary] = []
    print(
        f"Movie shape: {movie.shape}, shifts={shifts}, bins={bin_sizes}, repeat={args.repeat}, "
        f"modes={[e.label for e in estimators]}"
    )

    for bin_px in bin_sizes:
        if bin_px % args.base_patch_px != 0:
            print(f"[skip] bin {bin_px}: not divisible by base_patch_px={args.base_patch_px}")
            continue

        for estimator in estimators:
            runs: list[RunSummary] = []
            for _ in range(args.repeat):
                run = profile_once(
                    estimator=estimator,
                    movie=movie,
                    fs=float(args.fs),
                    extent=(float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])),
                    dj_mm=float(args.dj_mm),
                    base_patch_px=int(args.base_patch_px),
                    bin_px=int(bin_px),
                    shifts=tuple(int(s) for s in shifts),
                    options=opts,
                    use_shear_mask=bool(args.use_shear_mask),
                    show_progress=bool(args.show_progress),
                    do_cprofile=bool(args.do_cprofile),
                    profile_top=int(args.profile_top),
                )
                runs.append(run)


            summary = summarize_runs(runs)
            rows.append(summary)
            print(
                f"[{summary.estimator}] bin={summary.bin_px:>3d} | "
                f"best={summary.runtime_s:8.3f}s median={summary.runtime_median_s:8.3f}s "
                f"std={summary.runtime_std_s:7.4f}s | peak={summary.peak_mem_mb:8.1f} MB | "
                f"valid={summary.valid_seed_count}/{summary.total_seed_count} "
                f"({100.0 * summary.valid_frac:5.1f}%) | seeds/s={summary.seeds_per_s:8.1f}"
            )

    print_relative_summary(rows)

    if args.csv_out:
        out = Path(args.csv_out)
        write_csv(out, rows)
        print(f"\nWrote CSV: {out}")


if __name__ == "__main__":
    main()
