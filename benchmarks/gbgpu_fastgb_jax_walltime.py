"""Benchmark GBGPU-style FastGB generation against fastgb and JAX links.

The GBGPU package is a GPU/CPU reimplementation and extension of the FastGB
waveform used in GPU-accelerated LISA global-fit work.  This benchmark uses
the CPU backend, because CUDA/CuPy are not available on this machine.

Run with:

    python benchmarks/gbgpu_fastgb_jax_walltime.py

Outputs:

    benchmarks/gbgpu_fastgb_jax_walltime.csv
    benchmarks/gbgpu_fastgb_jax_walltime.png
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
from egb_multi.eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import FastGBBackend, default_lisaorbits, state_from_lisaorbits
from egb_multi.sources import GalacticBinaryParams


def make_circular_sources(n_sources: int, seed: int = 1234) -> list[GalacticBinaryParams]:
    rng = np.random.default_rng(seed)
    return [
        GalacticBinaryParams(
            f0=float(rng.uniform(5.0e-5, 3.0e-4)),
            fdot=0.0,
            amplitude=1.0e-22,
            beta=float(rng.uniform(-0.9, 0.9)),
            lambda_=float(rng.uniform(-np.pi, np.pi)),
            psi=float(rng.uniform(0.0, np.pi)),
            inclination=float(rng.uniform(0.0, np.pi)),
            phi0=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        for _ in range(n_sources)
    ]


def as_eccentric_zero_sources(sources: list[GalacticBinaryParams]) -> list[EccentricBinaryParams]:
    return [
        EccentricBinaryParams(
            mean_motion=np.pi * source.f0,
            eccentricity=0.0,
            m1_solar=0.5,
            m2_solar=0.5,
            beta=source.beta,
            lambda_=source.lambda_,
            psi=source.psi,
            inclination=source.inclination,
            phi0=source.phi0,
            fdot=source.fdot,
        )
        for source in sources
    ]


def as_gbgpu_arrays(sources: list[GalacticBinaryParams]) -> tuple[np.ndarray, ...]:
    amp = np.array([source.amplitude for source in sources], dtype=np.float64)
    f0 = np.array([source.f0 for source in sources], dtype=np.float64)
    fdot = np.array([source.fdot for source in sources], dtype=np.float64)
    fddot = np.zeros_like(f0)
    phi0 = np.array([source.phi0 for source in sources], dtype=np.float64)
    iota = np.array([source.inclination for source in sources], dtype=np.float64)
    psi = np.array([source.psi for source in sources], dtype=np.float64)
    lam = np.array([source.lambda_ for source in sources], dtype=np.float64)
    beta = np.array([source.beta for source in sources], dtype=np.float64)
    return amp, f0, fdot, fddot, phi0, iota, psi, lam, beta


def median_elapsed(func, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def main() -> None:
    try:
        from gbgpu.gbgpu import GBGPU
        from gbgpu.thirdbody import GBGPUThirdBody
    except Exception as exc:
        raise SystemExit(f"GBGPU is required for this benchmark: {exc}") from exc

    out_dir = Path(__file__).resolve().parent
    counts = [32, 128, 512, 2048]
    repeats = 5
    duration = 24.0 * 3600.0
    time_samples = 2048
    dt = duration / time_samples
    fd_bins = 128
    batch_size = 128

    orbits = default_lisaorbits("equal")
    times = np.linspace(0.0, duration, time_samples, endpoint=False)
    state = state_from_lisaorbits(orbits, times)
    geometry = precompute_jax_link_geometry(state)
    fastgb = FastGBBackend(delta_t=dt, duration=duration, n_samples=fd_bins, orbits=orbits)
    gbgpu = GBGPU(force_backend="cpu")
    gbgpu_third = GBGPUThirdBody(force_backend="cpu")

    warm_sources = make_circular_sources(batch_size)
    fastgb.frequency_domain_xyz(warm_sources)
    gbgpu.run_wave(*as_gbgpu_arrays(warm_sources), T=duration, dt=dt, N=fd_bins, tdi2=False)
    third_args = (
        np.full(batch_size, 1.0e-4),
        np.zeros(batch_size),
        np.full(batch_size, 0.1),
        np.ones(batch_size),
        np.zeros(batch_size),
    )
    gbgpu_third.run_wave(*as_gbgpu_arrays(warm_sources), *third_args, T=duration, dt=dt, N=fd_bins, tdi2=False)
    eccentric_links_jax(as_eccentric_zero_sources(warm_sources), geometry, batch_size=batch_size, physics_mode="newtonian")

    rows = []
    for count in counts:
        sources = make_circular_sources(count)
        eccentric_sources = as_eccentric_zero_sources(sources)
        gbgpu_args = as_gbgpu_arrays(sources)
        third_body_args = (
            np.full(count, 1.0e-4),
            np.zeros(count),
            np.full(count, 0.1),
            np.ones(count),
            np.zeros(count),
        )

        fastgb_s = median_elapsed(lambda: fastgb.frequency_domain_xyz(sources), repeats)
        fastgb_tdi2_s = median_elapsed(lambda: fastgb.frequency_domain_xyz(sources, tdi2=True), repeats)
        gbgpu_s = median_elapsed(lambda: gbgpu.run_wave(*gbgpu_args, T=duration, dt=dt, N=fd_bins, tdi2=False), repeats)
        gbgpu_tdi2_s = median_elapsed(lambda: gbgpu.run_wave(*gbgpu_args, T=duration, dt=dt, N=fd_bins, tdi2=True), repeats)
        gbgpu_third_s = median_elapsed(
            lambda: gbgpu_third.run_wave(
                *gbgpu_args,
                *third_body_args,
                T=duration,
                dt=dt,
                N=fd_bins,
                tdi2=False,
            ),
            repeats,
        )
        jax_links_s = median_elapsed(
            lambda: eccentric_links_jax(eccentric_sources, geometry, batch_size=batch_size, physics_mode="newtonian"),
            repeats,
        )

        row = {
            "sources": count,
            "duration_s": duration,
            "time_samples": time_samples,
            "dt_s": dt,
            "fd_bins": fd_bins,
            "fastgb_fd_xyz_s": fastgb_s,
            "fastgb_fd_xyz_tdi2_s": fastgb_tdi2_s,
            "gbgpu_cpu_fd_xae_s": gbgpu_s,
            "gbgpu_cpu_fd_xae_tdi2_s": gbgpu_tdi2_s,
            "gbgpu_cpu_third_body_fd_xae_s": gbgpu_third_s,
            "jax_time_domain_links_s": jax_links_s,
            "gbgpu_over_fastgb": gbgpu_s / fastgb_s,
            "gbgpu_third_body_over_gbgpu": gbgpu_third_s / gbgpu_s,
            "jax_links_over_gbgpu": jax_links_s / gbgpu_s,
        }
        rows.append(row)
        print(
            f"{count:5d} sources | FastGB {fastgb_s:.5f}s | GBGPU CPU {gbgpu_s:.5f}s | "
            f"GBGPU third {gbgpu_third_s:.5f}s | "
            f"FastGB TDI2 {fastgb_tdi2_s:.5f}s | GBGPU TDI2 {gbgpu_tdi2_s:.5f}s | "
            f"JAX TD links {jax_links_s:.5f}s"
        )

    csv_path = out_dir / "gbgpu_fastgb_jax_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["sources"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    axes[0].plot(x, [row["fastgb_fd_xyz_s"] for row in rows], marker="o", label="fastgb FD XYZ")
    axes[0].plot(x, [row["fastgb_fd_xyz_tdi2_s"] for row in rows], marker="P", label="fastgb FD XYZ, TDI2 factor")
    axes[0].plot(x, [row["gbgpu_cpu_fd_xae_s"] for row in rows], marker="s", label="GBGPU CPU FD XAE")
    axes[0].plot(x, [row["gbgpu_cpu_fd_xae_tdi2_s"] for row in rows], marker="D", label="GBGPU CPU FD XAE, TDI2")
    axes[0].plot(x, [row["gbgpu_cpu_third_body_fd_xae_s"] for row in rows], marker="v", label="GBGPU CPU third-body FD XAE")
    axes[0].plot(x, [row["jax_time_domain_links_s"] for row in rows], marker="^", label="JAX TD exact links")
    axes[1].plot(x, [row["gbgpu_over_fastgb"] for row in rows], marker="s", label="GBGPU CPU / fastgb")
    axes[1].plot(x, [row["gbgpu_third_body_over_gbgpu"] for row in rows], marker="v", label="GBGPU third-body / GBGPU")
    axes[1].plot(x, [row["jax_links_over_gbgpu"] for row in rows], marker="^", label="JAX TD links / GBGPU CPU")
    axes[0].set_ylabel("median wall time [s]")
    axes[1].set_ylabel("time ratio")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(counts)
        ax.set_xticklabels([str(count) for count in counts])
        ax.set_xlabel("number of circular sources")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("FastGB-family frequency-domain generators")
    axes[1].set_title("relative cost")

    png_path = out_dir / "gbgpu_fastgb_jax_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
