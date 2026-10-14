"""Wall-time benchmark: FastGB vs triliear and exact-link responses.

Run with:

    python benchmarks/fastgb_walltime.py

Outputs:

    benchmarks/walltime_fastgb_triliear_exact_xyz.png
    benchmarks/walltime_fastgb_triliear_exact_xyz.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.constants import JULIAN_YEAR_S
from egb_multi.external import FastGBBackend, default_lisaorbits, state_from_lisaorbits
from egb_multi.response import triliear_source_batch
from egb_multi.sources import GalacticBinaryParams
from egb_multi.tdi import compute_triliear_xyz, compute_xyz


def make_sources(n_sources: int, seed: int = 1234) -> list[GalacticBinaryParams]:
    rng = np.random.default_rng(seed)
    return [
        GalacticBinaryParams(
            f0=float(rng.uniform(5.0e-4, 3.0e-3)),
            fdot=float(rng.uniform(-1.0e-18, 1.0e-18)),
            amplitude=float(10.0 ** rng.uniform(-23.5, -21.5)),
            beta=float(rng.uniform(-0.9, 0.9)),
            lambda_=float(rng.uniform(-np.pi, np.pi)),
            psi=float(rng.uniform(0.0, np.pi)),
            inclination=float(rng.uniform(0.0, np.pi)),
            phi0=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        for _ in range(n_sources)
    ]


def run_fastgb(backend: FastGBBackend, sources: list[GalacticBinaryParams]) -> np.ndarray:
    return backend.frequency_domain_xyz(sources).xyz


def run_triliear(state, sources: list[GalacticBinaryParams]) -> np.ndarray:
    return triliear_source_batch(state, sources)


def run_exact_link_xyz(state, sources: list[GalacticBinaryParams]) -> dict[str, np.ndarray]:
    return compute_xyz(state, sources, generation=1, measurement_order=5, delay_order=5)


def run_triliear_xyz(state, sources: list[GalacticBinaryParams]) -> dict[str, np.ndarray]:
    return compute_triliear_xyz(state, sources)


def median_elapsed(func, repeats: int) -> float:
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        elapsed.append(time.perf_counter() - start)
    return float(np.median(elapsed))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    counts = [1, 2, 4, 8, 16, 32]
    repeats = 5

    duration = 365.0 * 24.0 * 3600.0
    fastgb_bins = 128
    triliear_samples = 2048
    times = np.linspace(0.0, JULIAN_YEAR_S, triliear_samples, endpoint=False)

    orbits = default_lisaorbits("esa-trailing")
    state = state_from_lisaorbits(orbits, times)
    fastgb = FastGBBackend(delta_t=5.0, duration=duration, n_samples=fastgb_bins, orbits=orbits)

    rows = []
    for count in counts:
        sources = make_sources(count)
        fastgb_time = median_elapsed(lambda: run_fastgb(fastgb, sources), repeats)
        triliear_time = median_elapsed(lambda: run_triliear(state, sources), repeats)
        triliear_xyz_time = median_elapsed(lambda: run_triliear_xyz(state, sources), repeats)
        exact_xyz_time = median_elapsed(lambda: run_exact_link_xyz(state, sources), repeats)
        rows.append(
            {
                "sources": count,
                "fastgb_s": fastgb_time,
                "triliear_s": triliear_time,
                "triliear_xyz_s": triliear_xyz_time,
                "exact_link_xyz_s": exact_xyz_time,
                "ratio_triliear_over_fastgb": triliear_time / fastgb_time,
                "ratio_triliear_xyz_over_fastgb": triliear_xyz_time / fastgb_time,
                "ratio_exact_link_xyz_over_fastgb": exact_xyz_time / fastgb_time,
            }
        )
        print(
            f"{count:3d} sources | FastGB {fastgb_time:.6f} s | "
            f"beta-triliear {triliear_time:.6f} s | "
            f"triliear XYZ {triliear_xyz_time:.6f} s | "
            f"exact-link XYZ {exact_xyz_time:.6f} s"
        )

    csv_path = out_dir / "walltime_fastgb_triliear_exact_xyz.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["sources"] for row in rows])
    fastgb_y = np.array([row["fastgb_s"] for row in rows])
    triliear_y = np.array([row["triliear_s"] for row in rows])
    triliear_xyz_y = np.array([row["triliear_xyz_s"] for row in rows])
    exact_y = np.array([row["exact_link_xyz_s"] for row in rows])

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(x, fastgb_y, marker="o", label="FastGB")
    ax.plot(x, triliear_y, marker="s", label="beta-triliear pairs")
    ax.plot(x, triliear_xyz_y, marker="D", label="triliear pyTDI XYZ default")
    ax.plot(x, exact_y, marker="^", label="exact-link pyTDI XYZ")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of sources")
    ax.set_ylabel("median wall time [s]")
    ax.set_title("FastGB vs beta-triliear vs exact-link XYZ wall time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    png_path = out_dir / "walltime_fastgb_triliear_exact_xyz.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
