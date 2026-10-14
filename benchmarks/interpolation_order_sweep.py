"""Sweep interpolation order for JAX brute-force eccentric links.

Run with:

    python benchmarks/interpolation_order_sweep.py

Outputs:

    benchmarks/interpolation_order_sweep.csv
    benchmarks/interpolation_order_sweep.png
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


def elapsed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def link_relerr(model: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> float:
    numerator = sum(np.linalg.norm(model[label] - reference[label]) ** 2 for label in reference)
    denominator = sum(np.linalg.norm(reference[label]) ** 2 for label in reference)
    return float(np.sqrt(numerator / denominator))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    n_sources = 128
    eccentricity = 0.1
    batch_size = 128
    duration = 1.0 * 24.0 * 3600.0
    samples = 8192
    threshold = 1.0e-2
    orders = [1, 3, 5, 7]
    times = np.linspace(0.0, duration, samples, endpoint=False)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)
    sources = make_sources(n_sources, eccentricity)

    print(
        f"sources={n_sources}, e={eccentricity}, samples={samples}, "
        f"dt={times[1] - times[0]:.3f} s, threshold={threshold:.1e}"
    )
    exact_links, exact_s = elapsed(lambda: eccentric_links_jax(sources, geometry, batch_size=batch_size))
    print(f"exact retarded links: {exact_s:.6f} s")

    rows = []
    previous_links = None
    for order in orders:
        links, wall_s = elapsed(
            lambda order=order: eccentric_links_interpolated_jax(
                sources,
                geometry,
                batch_size=batch_size,
                interpolation_order=order,
            )
        )
        residual_exact = link_relerr(links, exact_links)
        residual_previous = np.nan if previous_links is None else link_relerr(links, previous_links)
        accepted_by_successive_delta = bool(np.isnan(residual_previous) or residual_previous < threshold)
        rows.append(
            {
                "interpolation_order": order,
                "sources": n_sources,
                "eccentricity": eccentricity,
                "samples": samples,
                "dt_s": times[1] - times[0],
                "batch_size": batch_size,
                "exact_links_s": exact_s,
                "interpolated_links_s": wall_s,
                "speedup_exact_over_interpolated": exact_s / wall_s,
                "residual_vs_exact": residual_exact,
                "residual_vs_previous_order": residual_previous,
                "successive_delta_threshold": threshold,
                "accepted_by_successive_delta": accepted_by_successive_delta,
            }
        )
        print(
            f"order {order}: {wall_s:.6f} s | speedup {exact_s / wall_s:.2f}x | "
            f"vs exact {residual_exact:.3e} | vs previous {residual_previous:.3e}"
        )
        previous_links = links

    csv_path = out_dir / "interpolation_order_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["interpolation_order"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    axes[0].plot(x, [row["interpolated_links_s"] for row in rows], marker="o", label="interpolated")
    axes[0].axhline(exact_s, color="black", linestyle="--", linewidth=1.0, label="exact retarded")
    axes[1].plot(x, [row["residual_vs_exact"] for row in rows], marker="o", label="vs exact")
    axes[1].plot(x, [row["residual_vs_previous_order"] for row in rows], marker="s", label="vs previous order")
    axes[1].axhline(threshold, color="black", linestyle="--", linewidth=1.0, label="1% threshold")
    axes[0].set_xlabel("interpolation order")
    axes[0].set_ylabel("wall time [s]")
    axes[1].set_xlabel("interpolation order")
    axes[1].set_ylabel("relative link residual")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_title("interpolated JAX link wall time")
    axes[1].set_title("interpolation convergence")

    png_path = out_dir / "interpolation_order_sweep.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
