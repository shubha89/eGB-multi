"""Benchmark NumPy vs CPU-JAX eccentric source generation.

Run with:

    python benchmarks/jax_eccentric_walltime.py

Outputs:

    benchmarks/jax_eccentric_walltime.csv
    benchmarks/jax_eccentric_walltime.png
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
from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain
from egb_multi.eccentric_jax import eccentric_complex_strain_batch_jax_array, pack_eccentric_sources

import jax
import jax.numpy as jnp


def make_sources(n_sources: int, eccentricity: float = 0.1, seed: int = 1234) -> list[EccentricBinaryParams]:
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


def numpy_batch(sources: list[EccentricBinaryParams], times: np.ndarray) -> np.ndarray:
    return np.stack([eccentric_complex_strain(source, times) for source in sources], axis=0)


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
    times = np.linspace(0.0, JULIAN_YEAR_S, 2048, endpoint=False)
    times_jax = jnp.asarray(times)

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    rows = []
    for count in counts:
        sources = make_sources(count)
        params = pack_eccentric_sources(sources)
        params_jax = jnp.asarray(params)

        numpy_s = median_elapsed(lambda: numpy_batch(sources, times), repeats)

        start = time.perf_counter()
        compiled_result = eccentric_complex_strain_batch_jax_array(params_jax, times_jax)
        compiled_result.block_until_ready()
        jax_first_s = time.perf_counter() - start

        jax_device_s = median_elapsed(
            lambda: eccentric_complex_strain_batch_jax_array(params_jax, times_jax).block_until_ready(),
            repeats,
        )
        jax_host_s = median_elapsed(
            lambda: np.asarray(eccentric_complex_strain_batch_jax_array(params_jax, times_jax)),
            repeats,
        )

        rows.append(
            {
                "sources": count,
                "samples": times.size,
                "numpy_s": numpy_s,
                "jax_first_compile_and_run_s": jax_first_s,
                "jax_steady_device_s": jax_device_s,
                "jax_steady_host_s": jax_host_s,
                "speedup_device_vs_numpy": numpy_s / jax_device_s,
                "speedup_host_vs_numpy": numpy_s / jax_host_s,
            }
        )
        print(
            f"{count:4d} sources | NumPy {numpy_s:.6f} s | "
            f"JAX first {jax_first_s:.6f} s | "
            f"JAX steady device {jax_device_s:.6f} s | "
            f"JAX steady host {jax_host_s:.6f} s | "
            f"speedup {numpy_s / jax_device_s:.2f}x"
        )

    csv_path = out_dir / "jax_eccentric_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["sources"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].plot(x, [row["numpy_s"] for row in rows], marker="o", label="NumPy")
    axes[0].plot(x, [row["jax_steady_device_s"] for row in rows], marker="s", label="JAX steady device")
    axes[0].plot(x, [row["jax_steady_host_s"] for row in rows], marker="^", label="JAX steady host")
    axes[1].plot(x, [row["speedup_device_vs_numpy"] for row in rows], marker="s", label="device")
    axes[1].plot(x, [row["speedup_host_vs_numpy"] for row in rows], marker="^", label="host")

    axes[0].set_ylabel("median wall time [s]")
    axes[1].set_ylabel("NumPy / JAX speedup")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of sources")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_yscale("log")
    axes[0].set_title("eccentric source generation")
    axes[1].set_title("steady-state speedup")

    png_path = out_dir / "jax_eccentric_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
