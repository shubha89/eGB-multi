"""Wall-time benchmark for beta-link pyTDI options.

Run with:

    python benchmarks/beta_walltime_options.py

Outputs:

    benchmarks/beta_walltime_options.csv
    benchmarks/beta_walltime_options.png
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
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.sources import GalacticBinaryParams
from egb_multi.tdi import compute_beta_xyz, compute_xyz


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
    times = np.linspace(0.0, JULIAN_YEAR_S, 2048, endpoint=False)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)

    variants = {
        "beta1_bary_i3": lambda sources: compute_beta_xyz(
            state,
            sources,
            order=1,
            barycentric_phase=True,
            measurement_order=3,
            delay_order=3,
        ),
        "beta1_bary_i5": lambda sources: compute_beta_xyz(
            state,
            sources,
            order=1,
            barycentric_phase=True,
            measurement_order=5,
            delay_order=5,
        ),
        "beta_exact_bary_i3": lambda sources: compute_beta_xyz(
            state,
            sources,
            order="exact-transfer",
            barycentric_phase=True,
            measurement_order=3,
            delay_order=3,
        ),
        "beta_exact_bary_i5": lambda sources: compute_beta_xyz(
            state,
            sources,
            order="exact-transfer",
            barycentric_phase=True,
            measurement_order=5,
            delay_order=5,
        ),
        "beta_exact_bary_ret_i5": lambda sources: compute_beta_xyz(
            state,
            sources,
            order="exact-transfer",
            barycentric_phase=True,
            retarded_geometry=True,
            measurement_order=5,
            delay_order=5,
        ),
        "exact_link_i5": lambda sources: compute_xyz(
            state,
            sources,
            measurement_order=5,
            delay_order=5,
        ),
    }

    rows = []
    for count in counts:
        sources = make_sources(count)
        row = {"sources": count}
        print(f"{count} sources")
        for name, func in variants.items():
            elapsed = median_elapsed(lambda f=func: f(sources), repeats)
            row[f"{name}_s"] = elapsed
            print(f"  {name}: {elapsed:.6f} s")
        rows.append(row)

    csv_path = out_dir / "beta_walltime_options.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["sources"] for row in rows])
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for name in variants:
        y = np.array([row[f"{name}_s"] for row in rows])
        ax.plot(x, y, marker="o", label=name)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of sources")
    ax.set_ylabel("median wall time [s]")
    ax.set_title("beta-link pyTDI wall-time options")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    png_path = out_dir / "beta_walltime_options.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
