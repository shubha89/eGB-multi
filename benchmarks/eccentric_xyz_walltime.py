"""Wall-time comparison: circular FastGB, circular triliear XYZ, eccentric triliear XYZ.

FastGB is used as the circular frequency-domain baseline. The eccentric
triliear timings use the experimental small-e harmonic-carrier response;
FastGB has no eccentric source model in this package.

Run with:

    python benchmarks/eccentric_xyz_walltime.py

Outputs:

    benchmarks/eccentric_xyz_walltime.csv
    benchmarks/eccentric_xyz_walltime.png
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
from egb_multi.external import FastGBBackend, default_lisaorbits, state_from_lisaorbits
from egb_multi.sources import GalacticBinaryParams
from egb_multi.tdi import compute_triliear_xyz, compute_triliear_xyz_eccentric


def make_circular_sources(n_sources: int, seed: int = 1234) -> list[GalacticBinaryParams]:
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


def make_eccentric_sources(circular_sources: list[GalacticBinaryParams], eccentricity: float) -> list[EccentricBinaryParams]:
    return [
        EccentricBinaryParams(
            mean_motion=np.pi * source.f0,
            eccentricity=eccentricity,
            m1_solar=0.6,
            m2_solar=0.4,
            inclination=source.inclination,
            phi0=source.phi0,
        )
        for source in circular_sources
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
    eccentricities = [0.1, 0.3, 0.5, 0.8]
    repeats = 5

    duration = 365.0 * 24.0 * 3600.0
    fastgb_bins = 128
    samples = 2048
    times = np.linspace(0.0, JULIAN_YEAR_S, samples, endpoint=False)

    orbits = default_lisaorbits("esa-trailing")
    state = state_from_lisaorbits(orbits, times)
    fastgb = FastGBBackend(delta_t=5.0, duration=duration, n_samples=fastgb_bins, orbits=orbits)

    rows = []
    for count in counts:
        circular_sources = make_circular_sources(count)
        fastgb_s = median_elapsed(lambda: fastgb.frequency_domain_xyz(circular_sources), repeats)
        circular_triliear_xyz_s = median_elapsed(lambda: compute_triliear_xyz(state, circular_sources), repeats)

        row = {
            "sources": count,
            "fastgb_circular_s": fastgb_s,
            "triliear_xyz_circular_s": circular_triliear_xyz_s,
            "triliear_xyz_circular_over_fastgb": circular_triliear_xyz_s / fastgb_s,
        }
        print(
            f"{count:3d} sources | FastGB circular {fastgb_s:.6f} s | "
            f"triliear XYZ circular {circular_triliear_xyz_s:.6f} s"
        )

        for eccentricity in eccentricities:
            eccentric_sources = make_eccentric_sources(circular_sources, eccentricity)
            eccentric_triliear_xyz_s = median_elapsed(
                lambda s=eccentric_sources: compute_triliear_xyz_eccentric(
                    state,
                    s,
                    eccentric_order=1,
                    harmonic_samples=512,
                ),
                repeats,
            )
            row[f"triliear_xyz_e{eccentricity:.1f}_s"] = eccentric_triliear_xyz_s
            row[f"triliear_xyz_e{eccentricity:.1f}_over_fastgb"] = eccentric_triliear_xyz_s / fastgb_s
            row[f"eccentric_over_circular_e{eccentricity:.1f}"] = eccentric_triliear_xyz_s / circular_triliear_xyz_s
            print(
                f"    e={eccentricity:.1f}: harmonic-order-1 triliear XYZ {eccentric_triliear_xyz_s:.6f} s"
            )
        rows.append(row)

    csv_path = out_dir / "eccentric_xyz_walltime.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["sources"] for row in rows])
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.plot(x, [row["fastgb_circular_s"] for row in rows], marker="o", label="FastGB circular XYZ")
    ax.plot(x, [row["triliear_xyz_circular_s"] for row in rows], marker="s", label="triliear XYZ circular")
    for eccentricity in eccentricities:
        ax.plot(
            x,
            [row[f"triliear_xyz_e{eccentricity:.1f}_s"] for row in rows],
            marker="D",
            linewidth=1.0,
            label=f"triliear XYZ eccentric e={eccentricity:.1f}",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of sources")
    ax.set_ylabel("median wall time [s]")
    ax.set_title("FastGB vs circular and eccentric triliear XYZ wall time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    png_path = out_dir / "eccentric_xyz_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
