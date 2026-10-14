"""Wall-time benchmark for eccentric source generation plus triliear response.

Run with:

    python benchmarks/eccentric_walltime.py

Outputs:

    benchmarks/eccentric_walltime.csv
    benchmarks/eccentric_walltime.png
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
from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain, mikkola_kepler
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.sources import GalacticBinaryParams
from egb_multi.tdi import compute_triliear_xyz


def make_eccentric_sources(n_sources: int, eccentricity: float, seed: int = 1234) -> list[EccentricBinaryParams]:
    rng = np.random.default_rng(seed)
    return [
        EccentricBinaryParams(
            mean_motion=float(np.pi * rng.uniform(5.0e-4, 3.0e-3)),
            eccentricity=eccentricity,
            m1_solar=float(rng.uniform(0.35, 0.8)),
            m2_solar=float(rng.uniform(0.25, 0.8)),
            inclination=float(rng.uniform(0.0, np.pi)),
            phi0=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        for _ in range(n_sources)
    ]


def make_circular_response_sources(n_sources: int, seed: int = 1234) -> list[GalacticBinaryParams]:
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


def eccentric_source_batch(sources: list[EccentricBinaryParams], times: np.ndarray) -> np.ndarray:
    return np.stack([eccentric_complex_strain(source, times) for source in sources], axis=0)


def kepler_only_batch(sources: list[EccentricBinaryParams], times: np.ndarray) -> list[np.ndarray]:
    return [mikkola_kepler(source.mean_motion * (times - source.t0), source.eccentricity) for source in sources]


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    counts = [1, 2, 4, 8, 16, 32]
    eccentricities = [0.1, 0.5, 0.8]
    repeats = 5
    times = np.linspace(0.0, JULIAN_YEAR_S, 2048, endpoint=False)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)

    rows = []
    for count in counts:
        circular_sources = make_circular_response_sources(count)
        response_s = median_elapsed(
            lambda: compute_triliear_xyz(
                state,
                circular_sources,
                measurement_order=3,
                delay_order=3,
            ),
            repeats,
        )
        print(f"{count} sources")
        print(f"  triliear_xyz_default: {response_s:.6f} s")
        for eccentricity in eccentricities:
            sources = make_eccentric_sources(count, eccentricity)
            kepler_s = median_elapsed(lambda s=sources: kepler_only_batch(s, times), repeats)
            source_s = median_elapsed(lambda s=sources: eccentric_source_batch(s, times), repeats)
            rows.append(
                {
                    "sources": count,
                    "eccentricity": eccentricity,
                    "kepler_s": kepler_s,
                    "eccentric_source_s": source_s,
                    "triliear_xyz_default_s": response_s,
                    "source_over_response": source_s / response_s,
                }
            )
            print(
                f"  e={eccentricity:.1f}: kepler={kepler_s:.6f} s, "
                f"source={source_s:.6f} s, source/response={source_s / response_s:.3f}"
            )

    csv_path = out_dir / "eccentric_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for eccentricity in eccentricities:
        subset = [row for row in rows if row["eccentricity"] == eccentricity]
        x = np.array([row["sources"] for row in subset])
        ax.plot(x, [row["eccentric_source_s"] for row in subset], marker="o", label=f"eccentric source e={eccentricity:.1f}")
    base = [row for row in rows if row["eccentricity"] == eccentricities[0]]
    ax.plot(
        [row["sources"] for row in base],
        [row["triliear_xyz_default_s"] for row in base],
        marker="s",
        color="black",
        label="triliear XYZ default",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of sources")
    ax.set_ylabel("median wall time [s]")
    ax.set_title("eccentric source generation vs triliear response")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    png_path = out_dir / "eccentric_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
