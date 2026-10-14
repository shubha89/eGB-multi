"""Benchmark JAX brute-force eccentric links against the NumPy brute-force path.

Run with:

    python benchmarks/jax_bruteforce_links_walltime.py

Outputs:

    benchmarks/jax_bruteforce_links_walltime.csv
    benchmarks/jax_bruteforce_links_walltime.png
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

from eccentric_carrier_savings import brute_force_eccentric_links, brute_force_eccentric_xyz

from egb_multi.constants import JULIAN_YEAR_S
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.tdi import pytdi_data_from_links


def make_sources(n_sources: int, eccentricity: float = 0.1, seed: int = 1234) -> list[EccentricBinaryParams]:
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


def xyz_from_links(state, links):
    from pytdi.michelson import compute_factorized_michelson

    data = pytdi_data_from_links(state, links)
    return {
        "X": compute_factorized_michelson(data, rot=0, order=3, delay_order=3, generation=1, unit="frequency"),
        "Y": compute_factorized_michelson(data, rot=1, order=3, delay_order=3, generation=1, unit="frequency"),
        "Z": compute_factorized_michelson(data, rot=2, order=3, delay_order=3, generation=1, unit="frequency"),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    counts = [1, 8, 32, 128, 512]
    repeats = 5
    batch_size = 128
    times = np.linspace(0.0, JULIAN_YEAR_S, 2048, endpoint=False)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)

    rows = []
    for count in counts:
        sources = make_sources(count)
        numpy_links_s = median_elapsed(lambda: brute_force_eccentric_links(state, sources), repeats)

        start = time.perf_counter()
        eccentric_links_jax(sources, geometry, batch_size=batch_size)
        jax_first_links_s = time.perf_counter() - start

        jax_links_s = median_elapsed(lambda: eccentric_links_jax(sources, geometry, batch_size=batch_size), repeats)
        numpy_xyz_s = median_elapsed(lambda: brute_force_eccentric_xyz(state, sources), repeats)
        jax_xyz_s = median_elapsed(lambda: xyz_from_links(state, eccentric_links_jax(sources, geometry, batch_size=batch_size)), repeats)
        rows.append(
            {
                "sources": count,
                "samples": times.size,
                "batch_size": batch_size,
                "numpy_links_s": numpy_links_s,
                "jax_first_links_s": jax_first_links_s,
                "jax_links_s": jax_links_s,
                "numpy_xyz_s": numpy_xyz_s,
                "jax_links_plus_pytdi_xyz_s": jax_xyz_s,
                "links_speedup": numpy_links_s / jax_links_s,
                "xyz_speedup": numpy_xyz_s / jax_xyz_s,
            }
        )
        print(
            f"{count:4d} sources | links NumPy {numpy_links_s:.6f} s | "
            f"links JAX {jax_links_s:.6f} s | links speedup {numpy_links_s / jax_links_s:.2f}x | "
            f"XYZ NumPy {numpy_xyz_s:.6f} s | XYZ JAX-links {jax_xyz_s:.6f} s | "
            f"XYZ speedup {numpy_xyz_s / jax_xyz_s:.2f}x"
        )

    csv_path = out_dir / "jax_bruteforce_links_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["sources"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].plot(x, [row["numpy_links_s"] for row in rows], marker="o", label="NumPy links")
    axes[0].plot(x, [row["jax_links_s"] for row in rows], marker="s", label="JAX links")
    axes[0].plot(x, [row["numpy_xyz_s"] for row in rows], marker="^", label="NumPy links + pyTDI XYZ")
    axes[0].plot(x, [row["jax_links_plus_pytdi_xyz_s"] for row in rows], marker="D", label="JAX links + pyTDI XYZ")
    axes[1].plot(x, [row["links_speedup"] for row in rows], marker="s", label="links")
    axes[1].plot(x, [row["xyz_speedup"] for row in rows], marker="D", label="XYZ")

    axes[0].set_ylabel("median wall time [s]")
    axes[1].set_ylabel("NumPy / JAX speedup")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of sources")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_title("brute-force eccentric response")
    axes[1].set_title("steady-state speedup")

    png_path = out_dir / "jax_bruteforce_links_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
