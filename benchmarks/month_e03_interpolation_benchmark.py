"""One-month e=0.3 JAX brute-force vs interpolated eccentric benchmark.

Run with:

    python benchmarks/month_e03_interpolation_benchmark.py

Outputs:

    benchmarks/month_e03_interpolation_benchmark.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jax_bruteforce_links_walltime import xyz_from_links

from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_links_interpolated_jax, eccentric_links_jax, precompute_jax_link_geometry
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


def elapsed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def link_relerr(model: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> float:
    numerator = sum(np.linalg.norm(model[label] - reference[label]) ** 2 for label in reference)
    denominator = sum(np.linalg.norm(reference[label]) ** 2 for label in reference)
    return float(np.sqrt(numerator / denominator))


def xyz_relerr(model: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> float:
    numerator = sum(np.linalg.norm(model[channel] - reference[channel]) ** 2 for channel in ("X", "Y", "Z"))
    denominator = sum(np.linalg.norm(reference[channel]) ** 2 for channel in ("X", "Y", "Z"))
    return float(np.sqrt(numerator / denominator))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    n_sources = 64
    eccentricity = 0.3
    batch_size = 8
    interpolation_orders = [1, 3]
    duration = 30.0 * 24.0 * 3600.0
    dt = 10.0
    samples = int(duration / dt)
    times = np.arange(samples, dtype=np.float64) * dt

    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)
    sources = make_sources(n_sources, eccentricity)

    print(
        f"sources={n_sources}, e={eccentricity}, duration_days=30, samples={samples}, "
        f"dt={dt:.3f} s, batch_size={batch_size}, interpolation_orders={interpolation_orders}"
    )

    # Compile both fixed-shape kernels before timing.
    eccentric_links_jax(sources[:batch_size], geometry, batch_size=batch_size)
    for interpolation_order in interpolation_orders:
        eccentric_links_interpolated_jax(
            sources[:batch_size],
            geometry,
            batch_size=batch_size,
            interpolation_order=interpolation_order,
        )

    exact_links, exact_links_s = elapsed(lambda: eccentric_links_jax(sources, geometry, batch_size=batch_size))
    print(f"exact-retarded JAX links: {exact_links_s:.3f} s")
    exact_xyz, exact_xyz_s = elapsed(lambda: xyz_from_links(state, exact_links))
    print(f"exact-retarded JAX links + pyTDI XYZ: {exact_links_s + exact_xyz_s:.3f} s")

    rows = []
    previous_links = None
    for interpolation_order in interpolation_orders:
        interp_links, interp_links_s = elapsed(
            lambda interpolation_order=interpolation_order: eccentric_links_interpolated_jax(
                sources,
                geometry,
                batch_size=batch_size,
                interpolation_order=interpolation_order,
            )
        )
        print(f"interpolated order {interpolation_order} JAX links: {interp_links_s:.3f} s")
        interp_xyz, interp_xyz_s = elapsed(lambda links=interp_links: xyz_from_links(state, links))
        print(f"interpolated order {interpolation_order} JAX links + pyTDI XYZ: {interp_links_s + interp_xyz_s:.3f} s")

        row = {
            "sources": n_sources,
            "eccentricity": eccentricity,
            "duration_days": 30.0,
            "samples": samples,
            "dt_s": dt,
            "batch_size": batch_size,
            "interpolation_order": interpolation_order,
            "exact_links_s": exact_links_s,
            "exact_links_plus_xyz_s": exact_links_s + exact_xyz_s,
            "interpolated_links_s": interp_links_s,
            "interpolated_links_plus_xyz_s": interp_links_s + interp_xyz_s,
            "speedup_exact_links_over_interpolated_links": exact_links_s / interp_links_s,
            "speedup_exact_xyz_over_interpolated_xyz": (exact_links_s + exact_xyz_s) / (interp_links_s + interp_xyz_s),
            "link_relative_residual": link_relerr(interp_links, exact_links),
            "xyz_relative_residual": xyz_relerr(interp_xyz, exact_xyz),
            "link_relative_residual_vs_previous_order": np.nan if previous_links is None else link_relerr(interp_links, previous_links),
        }
        rows.append(row)
        previous_links = interp_links

    csv_path = out_dir / "month_e03_interpolation_benchmark.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"order {row['interpolation_order']} | link residual {row['link_relative_residual']:.3e} | "
            f"XYZ residual {row['xyz_relative_residual']:.3e} | "
            f"links speedup {row['speedup_exact_links_over_interpolated_links']:.2f}x | "
            f"links+XYZ speedup {row['speedup_exact_xyz_over_interpolated_xyz']:.2f}x"
        )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
