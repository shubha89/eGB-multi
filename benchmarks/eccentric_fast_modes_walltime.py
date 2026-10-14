"""Wall times and residuals for accelerated eccentric GB response modes.

Run with:

    python benchmarks/eccentric_fast_modes_walltime.py

Outputs:

    benchmarks/eccentric_fast_modes_walltime.csv
    benchmarks/eccentric_fast_modes_walltime.png
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

from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_harmonic_links_grouped_jax, eccentric_harmonic_links_jax, eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import FastGBBackend, default_lisaorbits, state_from_lisaorbits
from egb_multi.sources import GalacticBinaryParams


def make_eccentric_sources(n_sources: int, eccentricity: float, seed: int = 1234) -> list[EccentricBinaryParams]:
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


def as_circular_fastgb_sources(sources: list[EccentricBinaryParams]) -> list[GalacticBinaryParams]:
    return [
        GalacticBinaryParams(
            f0=source.mean_motion / np.pi,
            amplitude=1.0e-22,
            beta=source.beta,
            lambda_=source.lambda_,
            psi=source.psi,
            inclination=source.inclination,
            phi0=source.phi0,
        )
        for source in sources
    ]


def elapsed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def combined_xyz_residual(model: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> float:
    numerator = 0.0
    denominator = 0.0
    for channel in ("X", "Y", "Z"):
        numerator += np.linalg.norm(model[channel] - reference[channel]) ** 2
        denominator += np.linalg.norm(reference[channel]) ** 2
    return float(np.sqrt(numerator / denominator))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    counts = [32, 128, 512, 2048]
    eccentricity = 0.05
    eccentric_order = 2
    harmonic_samples = 512
    link_batch_size = 128
    source_batch_size = 128
    duration = 1.0 * 24.0 * 3600.0
    samples = 2048
    times = np.linspace(0.0, duration, samples, endpoint=False)

    orbits = default_lisaorbits("esa-trailing")
    state = state_from_lisaorbits(orbits, times)
    geometry = precompute_jax_link_geometry(state)
    fastgb = FastGBBackend(delta_t=times[1] - times[0], duration=duration, n_samples=128, orbits=orbits)

    warmup = make_eccentric_sources(source_batch_size, eccentricity)
    eccentric_links_jax(warmup, geometry, batch_size=source_batch_size)
    for mode in ("triliear_beta", "fastgb_transfer"):
        eccentric_harmonic_links_grouped_jax(
            warmup,
            geometry,
            eccentric_order=eccentric_order,
            harmonic_samples=harmonic_samples,
            batch_size=link_batch_size,
            response_mode=mode,
            barycentric_phase=False,
        )
    eccentric_harmonic_links_grouped_jax(
        warmup,
        geometry,
        eccentric_order=1,
        harmonic_samples=harmonic_samples,
        batch_size=link_batch_size,
        response_mode="fastgb_transfer",
        barycentric_phase=False,
    )
    fastgb.frequency_domain_xyz(as_circular_fastgb_sources(warmup))

    rows = []
    for count in counts:
        sources = make_eccentric_sources(count, eccentricity)
        circular_sources = as_circular_fastgb_sources(sources)

        _, fastgb_s = elapsed(lambda s=circular_sources: fastgb.frequency_domain_xyz(s))

        brute_links, brute_links_s = elapsed(lambda s=sources: eccentric_links_jax(s, geometry, batch_size=source_batch_size))
        brute_xyz, brute_xyz_s = elapsed(lambda: xyz_from_links(state, brute_links))
        brute_total_s = brute_links_s + brute_xyz_s

        beta_links, beta_links_s = elapsed(
            lambda s=sources: eccentric_harmonic_links_grouped_jax(
                s,
                geometry,
                eccentric_order=eccentric_order,
                harmonic_samples=harmonic_samples,
                batch_size=link_batch_size,
                response_mode="triliear_beta",
                barycentric_phase=False,
            )
        )
        beta_xyz, beta_xyz_s = elapsed(lambda: xyz_from_links(state, beta_links))
        beta_total_s = beta_links_s + beta_xyz_s

        fast_transfer_links, fast_transfer_links_s = elapsed(
            lambda s=sources: eccentric_harmonic_links_grouped_jax(
                s,
                geometry,
                eccentric_order=eccentric_order,
                harmonic_samples=harmonic_samples,
                batch_size=link_batch_size,
                response_mode="fastgb_transfer",
                barycentric_phase=False,
            )
        )
        fast_transfer_xyz, fast_transfer_xyz_s = elapsed(lambda: xyz_from_links(state, fast_transfer_links))
        fast_transfer_total_s = fast_transfer_links_s + fast_transfer_xyz_s

        order1_links, order1_links_s = elapsed(
            lambda s=sources: eccentric_harmonic_links_grouped_jax(
                s,
                geometry,
                eccentric_order=1,
                harmonic_samples=harmonic_samples,
                batch_size=link_batch_size,
                response_mode="fastgb_transfer",
                barycentric_phase=False,
            )
        )
        order1_xyz, order1_xyz_s = elapsed(lambda: xyz_from_links(state, order1_links))
        order1_total_s = order1_links_s + order1_xyz_s

        _, python_component_s = elapsed(
            lambda s=sources: xyz_from_links(
                state,
                eccentric_harmonic_links_jax(
                    s,
                    geometry,
                    eccentric_order=eccentric_order,
                    harmonic_samples=harmonic_samples,
                    batch_size=link_batch_size,
                    response_mode="fastgb_transfer",
                    barycentric_phase=False,
                    component_backend="python",
                ),
            )
        )

        row = {
            "sources": count,
            "eccentricity": eccentricity,
            "samples": samples,
            "dt_s": times[1] - times[0],
            "eccentric_order": eccentric_order,
            "harmonic_samples": harmonic_samples,
            "source_batch_size": source_batch_size,
            "link_batch_size": link_batch_size,
            "fastgb_circular_fd_xyz_s": fastgb_s,
            "bruteforce_jax_xyz_s": brute_total_s,
            "harmonic_beta_jax_xyz_s": beta_total_s,
            "harmonic_order1_fast_transfer_jax_xyz_s": order1_total_s,
            "harmonic_fast_transfer_jax_xyz_s": fast_transfer_total_s,
            "harmonic_fast_transfer_python_components_xyz_s": python_component_s,
            "beta_xyz_residual": combined_xyz_residual(beta_xyz, brute_xyz),
            "order1_fast_transfer_xyz_residual": combined_xyz_residual(order1_xyz, brute_xyz),
            "fast_transfer_xyz_residual": combined_xyz_residual(fast_transfer_xyz, brute_xyz),
            "fast_transfer_over_bruteforce_time": fast_transfer_total_s / brute_total_s,
            "fast_transfer_over_fastgb_time": fast_transfer_total_s / fastgb_s,
            "jax_components_over_python_components_time": fast_transfer_total_s / python_component_s,
        }
        rows.append(row)
        print(
            f"{count:5d} sources | FastGB circular {fastgb_s:.4f}s | brute {brute_total_s:.4f}s | "
            f"beta {beta_total_s:.4f}s r={row['beta_xyz_residual']:.3e} | "
            f"order1 {order1_total_s:.4f}s r={row['order1_fast_transfer_xyz_residual']:.3e} | "
            f"sinc/exact {fast_transfer_total_s:.4f}s r={row['fast_transfer_xyz_residual']:.3e} | "
            f"python-components {python_component_s:.4f}s"
        )

    csv_path = out_dir / "eccentric_fast_modes_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["sources"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].plot(x, [row["fastgb_circular_fd_xyz_s"] for row in rows], marker="o", label="FastGB circular FD XYZ")
    axes[0].plot(x, [row["bruteforce_jax_xyz_s"] for row in rows], marker="s", label="JAX brute-force TD XYZ")
    axes[0].plot(x, [row["harmonic_beta_jax_xyz_s"] for row in rows], marker="D", label="grouped JAX harmonic beta XYZ")
    axes[0].plot(x, [row["harmonic_order1_fast_transfer_jax_xyz_s"] for row in rows], marker="P", label="grouped JAX harmonic order1 sinc/exact XYZ")
    axes[0].plot(x, [row["harmonic_fast_transfer_jax_xyz_s"] for row in rows], marker="^", label="grouped JAX harmonic sinc/exact XYZ")
    axes[0].plot(
        x,
        [row["harmonic_fast_transfer_python_components_xyz_s"] for row in rows],
        marker="v",
        label="sinc/exact with Python components",
    )
    axes[1].plot(x, [row["beta_xyz_residual"] for row in rows], marker="D", label="beta")
    axes[1].plot(x, [row["order1_fast_transfer_xyz_residual"] for row in rows], marker="P", label="order1 sinc/exact")
    axes[1].plot(x, [row["fast_transfer_xyz_residual"] for row in rows], marker="^", label="sinc/exact")
    axes[0].set_ylabel("wall time [s]")
    axes[1].set_ylabel("relative XYZ residual vs brute force")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("number of sources")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_title("accelerated eccentric response wall time")
    axes[1].set_title("accuracy against brute-force time domain")

    png_path = out_dir / "eccentric_fast_modes_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
