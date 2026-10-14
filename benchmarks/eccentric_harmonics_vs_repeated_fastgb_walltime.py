"""Compare time-domain eccentric harmonic generation to repeated FastGB calls.

This is a deliberately narrow timing study.  It asks how the current JAX
time-domain eccentric-harmonic path compares with the optimistic frequency-
domain proxy where FastGB is called once for each retained positive eccentric
harmonic.

Run with:

    python benchmarks/eccentric_harmonics_vs_repeated_fastgb_walltime.py

Outputs:

    benchmarks/eccentric_harmonics_vs_repeated_fastgb_walltime.csv
    benchmarks/eccentric_harmonics_vs_repeated_fastgb_walltime.png
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

from egb_multi.eccentric import EccentricBinaryParams, eccentric_harmonic_angular_frequency, eccentric_harmonic_indices
from egb_multi.eccentric_jax import eccentric_harmonic_links_jax, precompute_jax_link_geometry
from egb_multi.external import FastGBBackend, default_lisaorbits, state_from_lisaorbits
from egb_multi.sources import GalacticBinaryParams


def make_sources(n_sources: int, eccentricity: float, seed: int = 1234) -> list[EccentricBinaryParams]:
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


def as_fastgb_sources(sources: list[EccentricBinaryParams], harmonic: int = 2) -> list[GalacticBinaryParams]:
    return [
        GalacticBinaryParams(
            f0=eccentric_harmonic_angular_frequency(source, harmonic) / (2.0 * np.pi),
            amplitude=1.0e-22,
            beta=source.beta,
            lambda_=source.lambda_,
            psi=source.psi,
            inclination=source.inclination,
            phi0=source.phi0,
        )
        for source in sources
    ]


def median_elapsed(func, repeats: int = 3) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def median_jax_links_and_xyz_elapsed(func, state, repeats: int = 3) -> tuple[float, float]:
    link_timings = []
    total_timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        links = func()
        links_done = time.perf_counter()
        xyz_from_links(state, links)
        total_done = time.perf_counter()
        link_timings.append(links_done - start)
        total_timings.append(total_done - start)
    return float(np.median(link_timings)), float(np.median(total_timings))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    source_counts = [32, 128, 512]
    tiers = [
        {"label": "order 0, k=2", "eccentricity": 0.0, "order": 0},
        {"label": "order 1, k=1..3", "eccentricity": 0.02, "order": 1},
        {"label": "order 2, k=1..4", "eccentricity": 0.05, "order": 2},
    ]
    repeats = 3
    duration = 24.0 * 3600.0
    samples = 2048
    times = np.linspace(0.0, duration, samples, endpoint=False)
    harmonic_samples = 512
    link_batch_size = 64
    source_batch_size = 128

    orbits = default_lisaorbits("esa-trailing")
    state = state_from_lisaorbits(orbits, times)
    geometry = precompute_jax_link_geometry(state)
    fastgb = FastGBBackend(delta_t=times[1] - times[0], duration=duration, n_samples=128, orbits=orbits)

    # Compile the three static JAX harmonic tiers before measuring steady-state times.
    for tier in tiers:
        warm_sources = make_sources(source_batch_size, float(tier["eccentricity"]))
        eccentric_harmonic_links_jax(
            warm_sources,
            geometry,
            eccentric_order=int(tier["order"]),
            harmonic_samples=harmonic_samples,
            batch_size=link_batch_size,
            response_mode="fastgb_transfer",
            barycentric_phase=False,
            component_source_batch_size=source_batch_size,
        )
    fastgb.frequency_domain_xyz(as_fastgb_sources(make_sources(source_batch_size, 0.0), harmonic=2))

    rows: list[dict[str, float | int | str]] = []
    for tier in tiers:
        eccentricity = float(tier["eccentricity"])
        order = int(tier["order"])
        harmonics = eccentric_harmonic_indices(order)
        positive_harmonic_count = len(harmonics)

        for source_count in source_counts:
            sources = make_sources(source_count, eccentricity)
            fastgb_sources_by_harmonic = {harmonic: as_fastgb_sources(sources, harmonic=harmonic) for harmonic in harmonics}

            def run_repeated_fastgb() -> None:
                for harmonic in harmonics:
                    fastgb.frequency_domain_xyz(fastgb_sources_by_harmonic[harmonic])

            def run_jax_links() -> dict[str, np.ndarray]:
                return eccentric_harmonic_links_jax(
                    sources,
                    geometry,
                    eccentric_order=order,
                    harmonic_samples=harmonic_samples,
                    batch_size=link_batch_size,
                    response_mode="fastgb_transfer",
                    barycentric_phase=False,
                    component_source_batch_size=source_batch_size,
                )

            single_fastgb_s = median_elapsed(lambda: fastgb.frequency_domain_xyz(fastgb_sources_by_harmonic[2]), repeats)
            repeated_fastgb_s = median_elapsed(run_repeated_fastgb, repeats)
            jax_links_s, jax_xyz_s = median_jax_links_and_xyz_elapsed(run_jax_links, state, repeats)

            row = {
                "tier": str(tier["label"]),
                "sources": source_count,
                "samples": samples,
                "duration_s": duration,
                "dt_s": float(times[1] - times[0]),
                "eccentricity": eccentricity,
                "eccentric_order": order,
                "positive_harmonic_count": positive_harmonic_count,
                "positive_harmonics": " ".join(str(harmonic) for harmonic in harmonics),
                "jax_internal_carriers_per_source": 2 * positive_harmonic_count,
                "harmonic_samples": harmonic_samples,
                "source_batch_size": source_batch_size,
                "link_batch_size": link_batch_size,
                "single_fastgb_xyz_s": single_fastgb_s,
                "repeated_fastgb_xyz_s": repeated_fastgb_s,
                "jax_time_domain_links_s": jax_links_s,
                "jax_time_domain_links_plus_pytdi_xyz_s": jax_xyz_s,
                "jax_links_over_repeated_fastgb": jax_links_s / repeated_fastgb_s,
                "jax_xyz_over_repeated_fastgb": jax_xyz_s / repeated_fastgb_s,
            }
            rows.append(row)
            print(
                f"{tier['label']:16s} | {source_count:4d} sources | "
                f"FastGB x{positive_harmonic_count}: {repeated_fastgb_s:.4f}s | "
                f"JAX TD links: {jax_links_s:.4f}s | "
                f"JAX TD links+XYZ: {jax_xyz_s:.4f}s"
            )

    csv_path = out_dir / "eccentric_harmonics_vs_repeated_fastgb_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    colors = {"order 0, k=2": "#0072B2", "order 1, k=1..3": "#D55E00", "order 2, k=1..4": "#009E73"}
    for tier in tiers:
        label = str(tier["label"])
        subset = [row for row in rows if row["tier"] == label]
        x = np.array([int(row["sources"]) for row in subset])
        color = colors[label]
        axes[0].plot(
            x,
            [float(row["repeated_fastgb_xyz_s"]) for row in subset],
            marker="o",
            linestyle="-",
            color=color,
            label=f"FastGB repeated, {label}",
        )
        axes[0].plot(
            x,
            [float(row["jax_time_domain_links_s"]) for row in subset],
            marker="s",
            linestyle="--",
            color=color,
            label=f"JAX TD links, {label}",
        )
        axes[0].plot(
            x,
            [float(row["jax_time_domain_links_plus_pytdi_xyz_s"]) for row in subset],
            marker="^",
            linestyle=":",
            color=color,
            label=f"JAX TD links+XYZ, {label}",
        )
        axes[1].plot(
            x,
            [float(row["jax_links_over_repeated_fastgb"]) for row in subset],
            marker="s",
            linestyle="--",
            color=color,
            label=f"links / repeated FastGB, {label}",
        )
        axes[1].plot(
            x,
            [float(row["jax_xyz_over_repeated_fastgb"]) for row in subset],
            marker="^",
            linestyle=":",
            color=color,
            label=f"links+XYZ / repeated FastGB, {label}",
        )

    axes[0].set_ylabel("median wall time [s]")
    axes[1].set_ylabel("time ratio")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(source_counts)
        ax.set_xticklabels([str(count) for count in source_counts])
        ax.set_xlabel("number of sources")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_title("FastGB repeated per harmonic vs JAX time domain")
    axes[1].set_title("JAX / repeated-FastGB cost")

    png_path = out_dir / "eccentric_harmonics_vs_repeated_fastgb_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
