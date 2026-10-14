"""Benchmark interpolated JAX eccentric links against exact-retarded JAX links.

Run with:

    python benchmarks/jax_interpolated_links_walltime.py

Outputs:

    benchmarks/jax_interpolated_links_walltime.csv
    benchmarks/jax_interpolated_links_walltime.png
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_links_interpolated_jax, eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits


def make_sources(n_sources: int, eccentricity: float = 0.1, seed: int = 1234) -> list[EccentricBinaryParams]:
    rng = np.random.default_rng(seed)
    return [
        EccentricBinaryParams(
            mean_motion=float(np.pi * rng.uniform(5.0e-5, 3.0e-4)),
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


def link_relerr(model: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> float:
    numerator = sum(np.linalg.norm(model[label] - reference[label]) ** 2 for label in reference)
    denominator = sum(np.linalg.norm(reference[label]) ** 2 for label in reference)
    return float(np.sqrt(numerator / denominator))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    counts = [32, 128, 512]
    repeats = 3
    batch_size = 128
    duration = 1.0 * 24.0 * 3600.0
    samples = 8192
    times = np.linspace(0.0, duration, samples, endpoint=False)
    dt = times[1] - times[0]
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)

    rows = []
    print(f"samples={samples}, duration_days=1, dt={dt:.3f} s, batch_size={batch_size}")
    for count in counts:
        sources = make_sources(count)
        exact = eccentric_links_jax(sources, geometry, batch_size=batch_size)
        interpolated = eccentric_links_interpolated_jax(sources, geometry, batch_size=batch_size)
        residual = link_relerr(interpolated, exact)
        exact_s = median_elapsed(lambda s=sources: eccentric_links_jax(s, geometry, batch_size=batch_size), repeats)
        interp_s = median_elapsed(lambda s=sources: eccentric_links_interpolated_jax(s, geometry, batch_size=batch_size), repeats)
        rows.append(
            {
                "sources": count,
                "samples": samples,
                "dt_s": dt,
                "batch_size": batch_size,
                "exact_jax_links_s": exact_s,
                "interpolated_jax_links_s": interp_s,
                "speedup_exact_over_interpolated": exact_s / interp_s,
                "interpolated_relative_link_residual": residual,
            }
        )
        print(
            f"{count:4d} sources | exact {exact_s:.6f} s | interpolated {interp_s:.6f} s | "
            f"speedup {exact_s / interp_s:.2f}x | residual {residual:.3e}"
        )

    csv_path = out_dir / "jax_interpolated_links_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["sources"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].plot(x, [row["exact_jax_links_s"] for row in rows], marker="o", label="exact-retarded JAX")
    axes[0].plot(x, [row["interpolated_jax_links_s"] for row in rows], marker="s", label="interpolated JAX")
    axes[1].plot(x, [row["interpolated_relative_link_residual"] for row in rows], marker="s", label="residual")
    axes[0].set_ylabel("median wall time [s]")
    axes[1].set_ylabel("relative link residual")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of sources")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_title("JAX link wall time")
    axes[1].set_title("interpolation error")

    png_path = out_dir / "jax_interpolated_links_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
