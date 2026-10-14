"""Wall-time comparison for brute-force circular and eccentric source responses.

This uses the same brute-force eccentric time-domain link evaluator for all
eccentricities, with `e=0` as the circular case.

Run with:

    python benchmarks/bruteforce_circular_eccentric_walltime.py

Outputs:

    benchmarks/bruteforce_circular_eccentric_walltime.csv
    benchmarks/bruteforce_circular_eccentric_walltime.png
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eccentric_carrier_savings import brute_force_eccentric_xyz

from egb_multi.constants import JULIAN_YEAR_S
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits, state_from_lisaorbits


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
    eccentricities = [0.0, 0.01, 0.03, 0.1, 0.3, 0.5]
    repeats = 5
    times = np.linspace(0.0, JULIAN_YEAR_S, 2048, endpoint=False)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)

    rows = []
    for count in counts:
        row = {"sources": count}
        print(f"{count} sources")
        circular_s = None
        for eccentricity in eccentricities:
            sources = make_sources(count, eccentricity)
            elapsed = median_elapsed(lambda s=sources: brute_force_eccentric_xyz(state, s), repeats)
            row[f"e{eccentricity:.2f}_s"] = elapsed
            if eccentricity == 0.0:
                circular_s = elapsed
            row[f"e{eccentricity:.2f}_over_circular"] = elapsed / circular_s if circular_s else 1.0
            print(f"  e={eccentricity:.2f}: {elapsed:.6f} s")
        rows.append(row)

    csv_path = out_dir / "bruteforce_circular_eccentric_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["sources"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for eccentricity in eccentricities:
        key = f"e{eccentricity:.2f}"
        label = "circular e=0" if eccentricity == 0.0 else f"e={eccentricity:.2f}"
        axes[0].plot(x, [row[f"{key}_s"] for row in rows], marker="o", label=label)
        axes[1].plot(x, [row[f"{key}_over_circular"] for row in rows], marker="o", label=label)

    axes[0].set_ylabel("median wall time [s]")
    axes[1].set_ylabel("wall time / circular")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of sources")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_title("brute-force XYZ wall time")
    axes[1].set_title("relative to circular")

    png_path = out_dir / "bruteforce_circular_eccentric_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
