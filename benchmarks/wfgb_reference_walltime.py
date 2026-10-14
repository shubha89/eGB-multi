"""Wall-time benchmark for the standalone wFGB reference implementation.

This benchmark compares:

* installed ``fastgb`` frequency-domain XYZ generation;
* ``egb_multi.wfgb.synthesize_wfgb``, a CPU reference implementation of the
  Riegger-Robertsson-Andersson windowed FGB algorithmic structure.

The outputs are not intended as an apples-to-apples accuracy comparison:
``fastgb`` returns compact frequency-domain slices, while this wFGB reference
returns windowed time-domain Michelson channels.  The purpose here is to test
the cost of following the paper's grouping/windowing steps in Python.

Run with:

    python benchmarks/wfgb_reference_walltime.py

Outputs:

    benchmarks/wfgb_reference_walltime.csv
    benchmarks/wfgb_reference_walltime.png
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.external import FastGBBackend, default_lisaorbits
from egb_multi.sources import GalacticBinaryParams
from egb_multi.wfgb import WFGBConfig, synthesize_wfgb


def make_sources(n_sources: int, seed: int = 1234) -> list[GalacticBinaryParams]:
    rng = np.random.default_rng(seed)
    return [
        GalacticBinaryParams(
            f0=float(rng.uniform(5.0e-5, 3.0e-4)),
            fdot=0.0,
            amplitude=1.0e-22,
            beta=float(rng.uniform(-0.9, 0.9)),
            lambda_=float(rng.uniform(-np.pi, np.pi)),
            psi=float(rng.uniform(0.0, np.pi)),
            inclination=float(rng.uniform(0.0, np.pi)),
            phi0=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        for _ in range(n_sources)
    ]


def median_elapsed(func, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    counts = [1, 8, 32, 128, 512]
    repeats = 3
    duration = 24.0 * 3600.0
    time_samples = 2048
    dt = duration / time_samples
    fastgb_fd_bins = 128
    orbits = default_lisaorbits("equal")
    config = WFGBConfig(
        duration=duration,
        dt=dt,
        buffer_duration=2.0 * duration,
        window_a=0.36,
        window_power=4,
        oversampling_power=3,
        min_slow_samples=1024,
        db=1,
        use_time_varying_delays=True,
    )
    fastgb = FastGBBackend(delta_t=dt, duration=duration, n_samples=fastgb_fd_bins, orbits=orbits)

    warm = make_sources(8)
    fastgb.frequency_domain_xyz(warm)
    synthesize_wfgb(warm, config, orbits=orbits)

    rows = []
    for count in counts:
        sources = make_sources(count)
        fastgb_s = median_elapsed(lambda: fastgb.frequency_domain_xyz(sources), repeats)
        result = synthesize_wfgb(sources, config, orbits=orbits)
        group_count = len(result.groups)
        max_group_size = max(len(group.source_indices) for group in result.groups)
        slow_samples_total = sum(group.slow_samples for group in result.groups)
        wfgb_s = median_elapsed(lambda: synthesize_wfgb(sources, config, orbits=orbits), repeats)
        row = {
            "sources": count,
            "groups": group_count,
            "max_group_size": max_group_size,
            "slow_samples_total_over_groups": slow_samples_total,
            "duration_s": duration,
            "dt_s": dt,
            "buffer_duration_s": config.buffer_duration,
            "window_a": config.window_a,
            "db": result.db,
            "fastgb_fd_xyz_s": fastgb_s,
            "wfgb_reference_xyz_s": wfgb_s,
            "wfgb_over_fastgb": wfgb_s / fastgb_s,
        }
        rows.append(row)
        print(
            f"{count:4d} sources -> {group_count:3d} groups | "
            f"fastgb {fastgb_s:.5f}s | wFGB reference {wfgb_s:.5f}s | "
            f"ratio {wfgb_s / fastgb_s:.1f}x"
        )

    csv_path = out_dir / "wfgb_reference_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["sources"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    axes[0].plot(x, [row["fastgb_fd_xyz_s"] for row in rows], marker="o", label="installed fastgb")
    axes[0].plot(x, [row["wfgb_reference_xyz_s"] for row in rows], marker="s", label="standalone wFGB reference")
    axes[1].plot(x, [row["groups"] for row in rows], marker="D", label="base-frequency groups")
    axes[1].plot(x, [row["wfgb_over_fastgb"] for row in rows], marker="^", label="wFGB reference / fastgb")
    axes[0].set_ylabel("median wall time [s]")
    axes[1].set_ylabel("count or ratio")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(value)) for value in x])
        ax.set_xlabel("number of sources")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("standalone wFGB reference cost")
    axes[1].set_title("grouping and relative cost")

    png_path = out_dir / "wfgb_reference_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
