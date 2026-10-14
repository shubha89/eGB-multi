"""Detector-level residuals and wall times vs eccentricity.

This compares final pyTDI X/Y/Z from:

1. full time-domain eccentric JAX brute-force links,
2. JAX harmonic eccentric triliear links.

Run with:

    python benchmarks/eccentric_xyz_residual_walltime.py

Outputs:

    benchmarks/eccentric_xyz_residual_walltime.csv
    benchmarks/eccentric_xyz_residual_walltime.png
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
from egb_multi.eccentric_jax import eccentric_harmonic_links_jax, eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits


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


def elapsed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def relative_l2(model: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(model - reference) / np.linalg.norm(reference))


def best_fit_relative_l2(model: np.ndarray, reference: np.ndarray) -> float:
    denom = np.vdot(model, model)
    scale = 0.0j if denom == 0 else np.vdot(model, reference) / denom
    return relative_l2(scale * model, reference)


def combined_xyz_residual(model: dict[str, np.ndarray], reference: dict[str, np.ndarray], *, scaled: bool) -> float:
    numerator = 0.0
    denominator = 0.0
    for channel in ("X", "Y", "Z"):
        model_channel = model[channel]
        reference_channel = reference[channel]
        if scaled:
            denom = np.vdot(model_channel, model_channel)
            scale = 0.0j if denom == 0 else np.vdot(model_channel, reference_channel) / denom
            model_channel = scale * model_channel
        numerator += np.linalg.norm(model_channel - reference_channel) ** 2
        denominator += np.linalg.norm(reference_channel) ** 2
    return float(np.sqrt(numerator / denominator))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    n_sources = 32
    eccentricities = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
    eccentric_order = 2
    harmonic_samples = 512
    batch_size = 128
    duration = 1.0 * 24.0 * 3600.0
    samples = 8192
    times = np.linspace(0.0, duration, samples, endpoint=False)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)

    rows = []
    print(
        f"sources={n_sources}, samples={samples}, dt={times[1] - times[0]:.3f} s, "
        f"harmonic_order={eccentric_order}, harmonic_samples={harmonic_samples}"
    )
    warmup_sources = make_sources(n_sources, eccentricities[0])
    warmup_brute_links = eccentric_links_jax(warmup_sources, geometry, batch_size=batch_size)
    warmup_harmonic_links = eccentric_harmonic_links_jax(
        warmup_sources,
        geometry,
        eccentric_order=eccentric_order,
        harmonic_samples=harmonic_samples,
        batch_size=batch_size,
        barycentric_phase=False,
    )
    xyz_from_links(state, warmup_brute_links)
    xyz_from_links(state, warmup_harmonic_links)
    for eccentricity in eccentricities:
        sources = make_sources(n_sources, eccentricity)
        brute_links, brute_links_s = elapsed(lambda s=sources: eccentric_links_jax(s, geometry, batch_size=batch_size))
        brute_xyz, brute_xyz_s = elapsed(lambda: xyz_from_links(state, brute_links))
        harmonic_links, harmonic_links_s = elapsed(
            lambda s=sources: eccentric_harmonic_links_jax(
                s,
                geometry,
                eccentric_order=eccentric_order,
                harmonic_samples=harmonic_samples,
                batch_size=batch_size,
                barycentric_phase=False,
            )
        )
        harmonic_xyz, harmonic_xyz_s = elapsed(lambda: xyz_from_links(state, harmonic_links))

        row = {
            "eccentricity": eccentricity,
            "sources": n_sources,
            "samples": samples,
            "dt_s": times[1] - times[0],
            "eccentric_order": eccentric_order,
            "harmonic_samples": harmonic_samples,
            "brute_force_links_s": brute_links_s,
            "brute_force_xyz_total_s": brute_links_s + brute_xyz_s,
            "harmonic_links_s": harmonic_links_s,
            "harmonic_xyz_total_s": harmonic_links_s + harmonic_xyz_s,
            "harmonic_over_bruteforce_time": (harmonic_links_s + harmonic_xyz_s) / (brute_links_s + brute_xyz_s),
            "X_raw_relerr": relative_l2(harmonic_xyz["X"], brute_xyz["X"]),
            "Y_raw_relerr": relative_l2(harmonic_xyz["Y"], brute_xyz["Y"]),
            "Z_raw_relerr": relative_l2(harmonic_xyz["Z"], brute_xyz["Z"]),
            "XYZ_raw_relerr": combined_xyz_residual(harmonic_xyz, brute_xyz, scaled=False),
            "X_scaled_relerr": best_fit_relative_l2(harmonic_xyz["X"], brute_xyz["X"]),
            "Y_scaled_relerr": best_fit_relative_l2(harmonic_xyz["Y"], brute_xyz["Y"]),
            "Z_scaled_relerr": best_fit_relative_l2(harmonic_xyz["Z"], brute_xyz["Z"]),
            "XYZ_scaled_relerr": combined_xyz_residual(harmonic_xyz, brute_xyz, scaled=True),
        }
        rows.append(row)
        print(
            f"e={eccentricity:.2f} | brute XYZ {row['brute_force_xyz_total_s']:.3f} s | "
            f"harmonic XYZ {row['harmonic_xyz_total_s']:.3f} s | "
            f"raw XYZ residual {row['XYZ_raw_relerr']:.3e} | "
            f"scaled XYZ residual {row['XYZ_scaled_relerr']:.3e}"
        )

    csv_path = out_dir / "eccentric_xyz_residual_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["eccentricity"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    axes[0].plot(x, [row["XYZ_raw_relerr"] for row in rows], marker="o", label="raw XYZ residual")
    axes[0].plot(x, [row["XYZ_scaled_relerr"] for row in rows], marker="s", label="best-fit scaled XYZ residual")
    axes[1].plot(x, [row["brute_force_xyz_total_s"] for row in rows], marker="o", label="full time-domain JAX brute force")
    axes[1].plot(x, [row["harmonic_xyz_total_s"] for row in rows], marker="s", label="JAX harmonic triliear")
    axes[0].set_xlabel("eccentricity")
    axes[0].set_ylabel("relative L2 residual")
    axes[0].set_yscale("log")
    axes[1].set_xlabel("eccentricity")
    axes[1].set_ylabel("wall time [s]")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_title("final X/Y/Z residual")
    axes[1].set_title("final X/Y/Z wall time")

    png_path = out_dir / "eccentric_xyz_residual_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
