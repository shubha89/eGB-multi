"""Benchmark Python vs JAX batched harmonic eccentric triliear response.

Run with:

    python benchmarks/jax_harmonic_triliear_walltime.py

Outputs:

    benchmarks/jax_harmonic_triliear_walltime.csv
    benchmarks/jax_harmonic_triliear_walltime.png
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

from jax_bruteforce_links_walltime import xyz_from_links

from egb_multi.constants import JULIAN_YEAR_S
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_harmonic_links_grouped_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.tdi import compute_triliear_xyz_eccentric


def make_sources(n_sources: int, eccentricity: float = 0.05, seed: int = 1234) -> list[EccentricBinaryParams]:
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
    counts = [1, 8, 32, 128, 512, 2048]
    repeats = 5
    eccentricity = 0.05
    eccentric_order = 2
    harmonic_samples = 512
    batch_size = 512
    times = np.linspace(0.0, JULIAN_YEAR_S, 2048, endpoint=False)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)

    rows = []
    for count in counts:
        sources = make_sources(count, eccentricity)
        python_s = median_elapsed(
            lambda s=sources: compute_triliear_xyz_eccentric(
                state,
                s,
                eccentric_order=eccentric_order,
                harmonic_samples=harmonic_samples,
            ),
            repeats,
        )
        jax_s = median_elapsed(
            lambda s=sources: xyz_from_links(
                state,
                eccentric_harmonic_links_grouped_jax(
                    s,
                    geometry,
                    eccentric_order=eccentric_order,
                    harmonic_samples=harmonic_samples,
                    batch_size=batch_size,
                ),
            ),
            repeats,
        )
        rows.append(
            {
                "sources": count,
                "eccentricity": eccentricity,
                "eccentric_order": eccentric_order,
                "harmonic_samples": harmonic_samples,
                "batch_size": batch_size,
                "python_triliear_xyz_s": python_s,
                "jax_triliear_xyz_s": jax_s,
                "speedup_python_over_jax": python_s / jax_s,
            }
        )
        print(
            f"{count:4d} sources | Python harmonic triliear {python_s:.6f} s | "
            f"JAX harmonic triliear {jax_s:.6f} s | speedup {python_s / jax_s:.2f}x"
        )

    csv_path = out_dir / "jax_harmonic_triliear_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["sources"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].plot(x, [row["python_triliear_xyz_s"] for row in rows], marker="o", label="Python harmonic triliear")
    axes[0].plot(x, [row["jax_triliear_xyz_s"] for row in rows], marker="s", label="JAX harmonic triliear")
    axes[1].plot(x, [row["speedup_python_over_jax"] for row in rows], marker="s", label="speedup")
    axes[0].set_ylabel("median wall time [s]")
    axes[1].set_ylabel("Python / JAX speedup")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of sources")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_yscale("log")
    axes[0].set_title("eccentric harmonic triliear XYZ")
    axes[1].set_title("batched JAX speedup")

    png_path = out_dir / "jax_harmonic_triliear_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
