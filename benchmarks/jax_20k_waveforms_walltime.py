"""Large-batch wall time for 20k eccentric brute-force waveforms.

Run with:

    python benchmarks/jax_20k_waveforms_walltime.py

Outputs:

    benchmarks/jax_20k_waveforms_walltime.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eccentric_carrier_savings import brute_force_eccentric_links, brute_force_eccentric_xyz
from jax_bruteforce_links_walltime import xyz_from_links

from egb_multi.constants import JULIAN_YEAR_S
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry
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


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    n_sources = 20_000
    n_samples = 2048
    batch_size = 512
    eccentricity = 0.1

    times = np.linspace(0.0, JULIAN_YEAR_S, n_samples, endpoint=False)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)
    sources = make_sources(n_sources, eccentricity)

    print(f"sources={n_sources}, samples={n_samples}, batch_size={batch_size}, eccentricity={eccentricity}")

    _, jax_compile_s = elapsed(lambda: eccentric_links_jax(sources[:batch_size], geometry, batch_size=batch_size))
    print(f"JAX compile/warmup links ({batch_size} sources padded): {jax_compile_s:.3f} s")

    jax_links, jax_links_s = elapsed(lambda: eccentric_links_jax(sources, geometry, batch_size=batch_size))
    print(f"JAX links 20k: {jax_links_s:.3f} s")

    _, jax_xyz_s = elapsed(lambda: xyz_from_links(state, jax_links))
    print(f"pyTDI XYZ from JAX links: {jax_xyz_s:.3f} s")

    _, numpy_links_s = elapsed(lambda: brute_force_eccentric_links(state, sources))
    print(f"NumPy links 20k: {numpy_links_s:.3f} s")

    _, numpy_xyz_s = elapsed(lambda: brute_force_eccentric_xyz(state, sources))
    print(f"NumPy links + pyTDI XYZ 20k: {numpy_xyz_s:.3f} s")

    row = {
        "sources": n_sources,
        "samples": n_samples,
        "batch_size": batch_size,
        "eccentricity": eccentricity,
        "jax_compile_warmup_links_s": jax_compile_s,
        "jax_links_s": jax_links_s,
        "jax_links_plus_pytdi_xyz_s": jax_links_s + jax_xyz_s,
        "numpy_links_s": numpy_links_s,
        "numpy_links_plus_pytdi_xyz_s": numpy_xyz_s,
        "links_speedup": numpy_links_s / jax_links_s,
        "xyz_speedup": numpy_xyz_s / (jax_links_s + jax_xyz_s),
    }

    csv_path = out_dir / "jax_20k_waveforms_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    print(f"links speedup: {row['links_speedup']:.2f}x")
    print(f"links+XYZ speedup: {row['xyz_speedup']:.2f}x")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
