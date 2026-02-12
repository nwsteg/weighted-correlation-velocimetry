#!/usr/bin/env python3
"""Profile WCV map estimators across bin sizes.

This script is intended to reproduce/compare the performance trade-offs between
`estimate_velocity_map` (materialized correlation matrices),
`estimate_velocity_map_streaming` (seed-by-seed correlations), and
`estimate_velocity_map_hybrid` (batched seed/shift streaming).

Examples
--------
# Synthetic benchmark
python scripts/profile_velocity_map.py \
  --mode all \
  --shape 200,1024,1024 \
  --bin-sizes 4,8,16,32 \
  --shifts 1 \
  --repeat 2 \
  --profile-top 25

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
import inspect
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


def call_estimator(fn: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    use_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**use_kwargs)


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


def profile_once(
    *,
    estimator_name: str,
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
    chunk_size: int | None,
    shift_chunk_size: int | None,
    max_corr_buffer_mb: float | None,
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

    if estimator_name == "estimate_velocity_map_streaming":
        fn = wcv.estimate_velocity_map_streaming
    elif estimator_name == "estimate_velocity_map_hybrid":
        fn = wcv.estimate_velocity_map_hybrid
    elif estimator_name == "estimate_velocity_map":
        fn = wcv.estimate_velocity_map
    else:
        raise ValueError(f"Unknown estimator: {estimator_name}")

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
        chunk_size=chunk_size,
        seed_chunk_size=chunk_size,
        shift_chunk_size=shift_chunk_size,
        max_corr_buffer_mb=max_corr_buffer_mb,
        return_corr_by_shift=False,
        store_corr_by_shift=False,
    )

    profiler: cProfile.Profile | None = cProfile.Profile() if do_cprofile else None
    tracemalloc.start()
    t0 = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    vm = call_estimator(fn, **kwargs)
    if profiler is not None:
        profiler.disable()
    runtime_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if profiler is not None:
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats("cumtime").print_stats(profile_top)
        print(f"\n--- cProfile: {estimator_name}, bin={bin_px} ---")
        print(s.getvalue())

    n_regions = int(vm.by * vm.bx)
    return RunSummary(
        estimator=estimator_name,
        bin_px=bin_px,
        runtime_s=runtime_s,
        peak_mem_mb=peak / (1024 * 1024),
        valid_seed_count=int(vm.valid_seed_count),
        total_seed_count=int(vm.total_seed_count),
        n_regions=n_regions,
        nt=int(movie.shape[0]),
        ny=int(movie.shape[1]),
        nx=int(movie.shape[2]),
    )


def write_csv(path: Path, rows: list[RunSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "estimator",
                "bin_px",
                "runtime_s",
                "peak_mem_mb",
                "valid_seed_count",
                "total_seed_count",
                "n_regions",
                "nt",
                "ny",
                "nx",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.estimator,
                    r.bin_px,
                    f"{r.runtime_s:.6f}",
                    f"{r.peak_mem_mb:.3f}",
                    r.valid_seed_count,
                    r.total_seed_count,
                    r.n_regions,
                    r.nt,
                    r.ny,
                    r.nx,
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
    p.add_argument("--mode", choices=["materialized", "streaming", "hybrid", "all"], default="all")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--use-shear-mask", action="store_true")
    p.add_argument("--show-progress", action="store_true")
    p.add_argument("--chunk-size", type=int, default=None, help="Seed chunk size for streaming/hybrid modes")
    p.add_argument("--shift-chunk-size", type=int, default=None, help="Shift chunk size for hybrid mode")
    p.add_argument("--max-corr-buffer-mb", type=float, default=None, help="Max temporary correlation buffer size in MB for hybrid mode")
    p.add_argument("--rmin", type=float, default=0.3)
    p.add_argument("--min-used", type=int, default=15)
    p.add_argument("--weight-power", type=float, default=2.0)
    p.add_argument("--do-cprofile", action="store_true")
    p.add_argument("--profile-top", type=int, default=20)
    p.add_argument("--csv-out", type=str, default=None, help="Optional CSV output path")
    return p.parse_args()


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

    estimators: list[str]
    if args.mode == "all":
        estimators = [
            "estimate_velocity_map",
            "estimate_velocity_map_streaming",
            "estimate_velocity_map_hybrid",
        ]
    elif args.mode == "materialized":
        estimators = ["estimate_velocity_map"]
    elif args.mode == "streaming":
        estimators = ["estimate_velocity_map_streaming"]
    else:
        estimators = ["estimate_velocity_map_hybrid"]

    rows: list[RunSummary] = []
    print(f"Movie shape: {movie.shape}, shifts={shifts}, bins={bin_sizes}, repeat={args.repeat}")

    for bin_px in bin_sizes:
        if bin_px % args.base_patch_px != 0:
            print(f"[skip] bin {bin_px}: not divisible by base_patch_px={args.base_patch_px}")
            continue

        for estimator_name in estimators:
            best: RunSummary | None = None
            for _ in range(args.repeat):
                run = profile_once(
                    estimator_name=estimator_name,
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
                    chunk_size=args.chunk_size,
                    shift_chunk_size=args.shift_chunk_size,
                    max_corr_buffer_mb=args.max_corr_buffer_mb,
                    do_cprofile=bool(args.do_cprofile),
                    profile_top=int(args.profile_top),
                )
                if best is None or run.runtime_s < best.runtime_s:
                    best = run

            assert best is not None
            rows.append(best)
            print(
                f"[{best.estimator}] bin={best.bin_px:>3d} | "
                f"time={best.runtime_s:8.3f}s | peak={best.peak_mem_mb:8.1f} MB | "
                f"valid={best.valid_seed_count}/{best.total_seed_count}"
            )

    if args.csv_out:
        out = Path(args.csv_out)
        write_csv(out, rows)
        print(f"\nWrote CSV: {out}")


if __name__ == "__main__":
    main()
