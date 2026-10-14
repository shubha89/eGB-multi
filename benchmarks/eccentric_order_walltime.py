"""Wall-time benchmark for eccentric triliear harmonic order.

Run with:

    python benchmarks/eccentric_order_walltime.py

Outputs:

    benchmarks/eccentric_order_walltime.csv
    benchmarks/eccentric_order_walltime.png
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
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.tdi import compute_triliear_xyz_eccentric


def make_sources(n_sources: int, eccentricity: float, seed: int = 1234) -> list[EccentricBinaryParams]:
    rng = np.random.default_rng(seed)
    return [
        EccentricBinaryParams(
            mean_motion=float(np.pi * rng.uniform(5.0e-4, 3.0e-3)),
            eccentricity=eccentricity,
            m1_solar=float(rng.uniform(0.35, 0.8)),
            m2_solar=float(rng.uniform(0.25, 0.8)),
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
    orders = [0, 1, 2]
    repeats = 5
    eccentricity = 0.1
    times = np.linspace(0.0, JULIAN_YEAR_S, 2048, endpoint=False)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)

    rows = []
    for count in counts:
        sources = make_sources(count, eccentricity)
        row = {"sources": count, "eccentricity": eccentricity}
        print(f"{count} sources")
        for eccentric_order in orders:
            elapsed = median_elapsed(
                lambda o=eccentric_order: compute_triliear_xyz_eccentric(
                    state,
                    sources,
                    eccentric_order=o,
                    harmonic_samples=512,
                ),
                repeats,
            )
            row[f"eccentric_order_{eccentric_order}_s"] = elapsed
            print(f"  eccentric_order={eccentric_order}: {elapsed:.6f} s")
        rows.append(row)

    csv_path = out_dir / "eccentric_order_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["sources"] for row in rows])
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for eccentric_order in orders:
        ax.plot(
            x,
            [row[f"eccentric_order_{eccentric_order}_s"] for row in rows],
            marker="o",
            label=f"eccentric_order={eccentric_order}",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of sources")
    ax.set_ylabel("median wall time [s]")
    ax.set_title("eccentric triliear XYZ harmonic-order wall time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    png_path = out_dir / "eccentric_order_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
