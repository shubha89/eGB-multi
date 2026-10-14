"""Residuals for harmonic JAX response modes against full time-domain JAX brute force.

Run with:

    python benchmarks/fastgb_transfer_residual.py

Outputs:

    benchmarks/fastgb_transfer_residual.csv
    benchmarks/fastgb_transfer_residual.png
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
    eccentricities = [0.0, 0.01, 0.05]
    response_modes = ["triliear_beta", "fastgb_transfer"]
    eccentric_order = 2
    harmonic_samples = 512
    batch_size = 128
    duration = 1.0 * 24.0 * 3600.0
    samples = 8192
    times = np.linspace(0.0, duration, samples, endpoint=False)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)

    warmup = make_sources(n_sources, 0.0)
    xyz_from_links(state, eccentric_links_jax(warmup, geometry, batch_size=batch_size))
    for mode in response_modes:
        xyz_from_links(
            state,
            eccentric_harmonic_links_jax(
                warmup,
                geometry,
                eccentric_order=eccentric_order,
                harmonic_samples=harmonic_samples,
                batch_size=batch_size,
                response_mode=mode,
                barycentric_phase=False,
            ),
        )

    rows = []
    for eccentricity in eccentricities:
        sources = make_sources(n_sources, eccentricity)
        brute_links, brute_links_s = elapsed(lambda s=sources: eccentric_links_jax(s, geometry, batch_size=batch_size))
        brute_xyz, brute_xyz_s = elapsed(lambda: xyz_from_links(state, brute_links))
        for mode in response_modes:
            model_links, model_links_s = elapsed(
                lambda s=sources, m=mode: eccentric_harmonic_links_jax(
                    s,
                    geometry,
                    eccentric_order=eccentric_order,
                    harmonic_samples=harmonic_samples,
                    batch_size=batch_size,
                    response_mode=m,
                    barycentric_phase=False,
                )
            )
            model_xyz, model_xyz_s = elapsed(lambda links=model_links: xyz_from_links(state, links))
            row = {
                "eccentricity": eccentricity,
                "response_mode": mode,
                "sources": n_sources,
                "samples": samples,
                "dt_s": times[1] - times[0],
                "brute_force_xyz_total_s": brute_links_s + brute_xyz_s,
                "model_xyz_total_s": model_links_s + model_xyz_s,
                "model_over_bruteforce_time": (model_links_s + model_xyz_s) / (brute_links_s + brute_xyz_s),
                "XYZ_raw_relerr": combined_xyz_residual(model_xyz, brute_xyz, scaled=False),
                "XYZ_scaled_relerr": combined_xyz_residual(model_xyz, brute_xyz, scaled=True),
            }
            rows.append(row)
            print(
                f"e={eccentricity:.2f} {mode}: time {row['model_xyz_total_s']:.3f}s "
                f"raw {row['XYZ_raw_relerr']:.3e} scaled {row['XYZ_scaled_relerr']:.3e}"
            )

    csv_path = out_dir / "fastgb_transfer_residual.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    for mode in response_modes:
        subset = [row for row in rows if row["response_mode"] == mode]
        x = [row["eccentricity"] for row in subset]
        axes[0].plot(x, [row["XYZ_scaled_relerr"] for row in subset], marker="o", label=mode)
        axes[1].plot(x, [row["model_xyz_total_s"] for row in subset], marker="o", label=mode)
    axes[0].set_xlabel("eccentricity")
    axes[0].set_ylabel("best-fit scaled XYZ residual")
    axes[0].set_yscale("log")
    axes[1].set_xlabel("eccentricity")
    axes[1].set_ylabel("wall time [s]")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_title("response residual vs brute force")
    axes[1].set_title("response wall time")

    png_path = out_dir / "fastgb_transfer_residual.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
