"""Compare nearly circular eccentric brute-force XYZ against FastGB circular XYZ.

This compares the same sparse frequency bins returned by FastGB:

1. compute eccentric JAX brute-force links in the time domain with e=1e-4,
2. compute pyTDI X/Y/Z,
3. FFT the time-domain X/Y/Z,
4. select FastGB's bins and compare complex spectra.

The eccentric model uses physical masses/distance while FastGB accepts a direct
strain amplitude, so the scaled residual is the useful convention check.

Run with:

    python benchmarks/eccentric_bruteforce_vs_fastgb_circular.py
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

from egb_multi.constants import G_M3_KG_S2, KILOPARSEC_M, SOLAR_MASS_KG
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import FastGBBackend, default_lisaorbits, state_from_lisaorbits
from egb_multi.sources import GalacticBinaryParams


def circular_strain_amplitude(source: EccentricBinaryParams) -> float:
    """Approximate dominant circular amplitude for FastGB convention."""

    total_mass = (source.m1_solar + source.m2_solar) * SOLAR_MASS_KG
    eta = source.symmetric_mass_ratio
    return float(
        2.0
        * eta
        * (G_M3_KG_S2 * total_mass) ** (5.0 / 3.0)
        * source.mean_motion ** (2.0 / 3.0)
        / (299792458.0**4 * source.distance_m)
    )


def as_fastgb_source(source: EccentricBinaryParams) -> GalacticBinaryParams:
    return GalacticBinaryParams(
        f0=source.mean_motion / np.pi,
        fdot=0.0,
        amplitude=circular_strain_amplitude(source),
        beta=source.beta,
        lambda_=source.lambda_,
        psi=source.psi,
        inclination=source.inclination,
        phi0=source.phi0,
    )


def best_fit_scale(model: np.ndarray, reference: np.ndarray) -> complex:
    denom = np.vdot(model, model)
    if denom == 0:
        return 0.0j
    return complex(np.vdot(model, reference) / denom)


def relative_norm(model: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(model - reference) / np.linalg.norm(reference))


def combined_xyz_residual(model: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> float:
    numerator = 0.0
    denominator = 0.0
    for channel in ("X", "Y", "Z"):
        numerator += np.linalg.norm(model[channel] - reference[channel]) ** 2
        denominator += np.linalg.norm(reference[channel]) ** 2
    return float(np.sqrt(numerator / denominator))


def select_fft_bins(series: np.ndarray, dt: float, kmin: int, n_bins: int) -> np.ndarray:
    spectrum = np.fft.fft(series) * dt
    if kmin + n_bins > spectrum.size:
        raise ValueError(f"FastGB bins [{kmin}, {kmin + n_bins}) exceed FFT size {spectrum.size}")
    return spectrum[kmin : kmin + n_bins]


def elapsed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def compare_frequency(f0: float, *, out_dir: Path) -> dict[str, float | str]:
    eccentricity = 1.0e-4
    duration = 90.0 * 24.0 * 3600.0
    dt = 100.0
    n_time = int(duration / dt)
    duration = n_time * dt
    times = np.arange(n_time, dtype=np.float64) * dt
    fastgb_bins = 128

    eccentric_source = EccentricBinaryParams(
        mean_motion=np.pi * f0,
        eccentricity=eccentricity,
        m1_solar=0.5,
        m2_solar=0.5,
        distance_m=KILOPARSEC_M,
        beta=0.3,
        lambda_=-0.7,
        psi=0.4,
        inclination=1.0,
        phi0=0.2,
    )
    circular_eccentric_source = EccentricBinaryParams(
        **{**eccentric_source.__dict__, "eccentricity": 0.0},
    )
    fastgb_source = as_fastgb_source(eccentric_source)

    orbits = default_lisaorbits("equal")
    fastgb = FastGBBackend(delta_t=dt, duration=duration, n_samples=fastgb_bins, orbits=orbits)
    fast, fastgb_s = elapsed(lambda: fastgb.frequency_domain_xyz(fastgb_source))
    kmin = int(fast.kmin[0])
    n_bins = fast.x.shape[-1]
    fast_xyz = {"X": fast.x[0], "Y": fast.y[0], "Z": fast.z[0]}

    state = state_from_lisaorbits(orbits, times)
    geometry = precompute_jax_link_geometry(state)
    eccentric_links_jax([eccentric_source], geometry, batch_size=1)
    circular_links = eccentric_links_jax([circular_eccentric_source], geometry, batch_size=1)
    circular_xyz = xyz_from_links(state, circular_links)
    links, links_s = elapsed(lambda: eccentric_links_jax([eccentric_source], geometry, batch_size=1))
    xyz, xyz_s = elapsed(lambda: xyz_from_links(state, links))

    row: dict[str, float | str] = {
        "f0": f0,
        "eccentricity": eccentricity,
        "duration_days": duration / 86400.0,
        "samples": n_time,
        "dt_s": dt,
        "fastgb_bins": n_bins,
        "kmin": kmin,
        "fastgb_s": fastgb_s,
        "bruteforce_links_s": links_s,
        "pytdi_xyz_s": xyz_s,
        "bruteforce_total_s": links_s + xyz_s,
        "eccentric_vs_e0_bruteforce_xyz_relerr": combined_xyz_residual(xyz, circular_xyz),
    }

    freq = np.arange(kmin, kmin + n_bins, dtype=np.float64) / duration
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 7.0), sharex="col", constrained_layout=True)
    combined_num = 0.0
    combined_den = 0.0
    combined_num_scaled = 0.0

    for channel_index, channel in enumerate(("X", "Y", "Z")):
        model = select_fft_bins(xyz[channel], dt, kmin, n_bins)
        reference = fast_xyz[channel]
        alpha = best_fit_scale(model, reference)
        scaled = alpha * model
        residual = reference - scaled

        row[f"{channel}_raw_relerr"] = relative_norm(model, reference)
        row[f"{channel}_scaled_relerr"] = relative_norm(scaled, reference)
        row[f"{channel}_scale_abs"] = abs(alpha)
        row[f"{channel}_scale_phase"] = float(np.angle(alpha))
        row[f"{channel}_max_peak_residual"] = float(np.max(np.abs(residual)) / np.max(np.abs(reference)))

        combined_num += np.linalg.norm(model - reference) ** 2
        combined_num_scaled += np.linalg.norm(residual) ** 2
        combined_den += np.linalg.norm(reference) ** 2

        axes[channel_index, 0].plot(freq, np.abs(reference), label="FastGB circular", linewidth=1.4)
        axes[channel_index, 0].plot(freq, np.abs(scaled), label="scaled eccentric brute-force FFT", linestyle="--")
        axes[channel_index, 0].set_ylabel(f"|{channel}(f)|")
        axes[channel_index, 0].grid(True, alpha=0.3)

        axes[channel_index, 1].plot(freq, np.abs(residual) / np.max(np.abs(reference)), color="tab:red")
        axes[channel_index, 1].set_ylabel(f"{channel} residual")
        axes[channel_index, 1].grid(True, alpha=0.3)

    row["XYZ_raw_relerr"] = float(np.sqrt(combined_num / combined_den))
    row["XYZ_scaled_relerr"] = float(np.sqrt(combined_num_scaled / combined_den))

    axes[-1, 0].set_xlabel("frequency [Hz]")
    axes[-1, 1].set_xlabel("frequency [Hz]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("amplitude overlay")
    axes[0, 1].set_title("|FastGB - scaled brute force| / peak(|FastGB|)")
    fig.suptitle(f"e=1e-4 brute-force eccentric vs FastGB circular, f0={f0:.1e} Hz")
    plot = out_dir / f"eccentric_bruteforce_vs_fastgb_circular_f0_{f0:.0e}.png"
    fig.savefig(plot, dpi=180)
    plt.close(fig)
    row["plot"] = str(plot)
    return row


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    frequencies = [1.0e-4, 3.0e-4, 1.0e-3]
    rows = [compare_frequency(f0, out_dir=out_dir) for f0 in frequencies]

    csv_path = out_dir / "eccentric_bruteforce_vs_fastgb_circular.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"f0={row['f0']:.1e} Hz | e={row['eccentricity']:.1e} | "
            f"FastGB {row['fastgb_s']:.3f}s | brute XYZ {row['bruteforce_total_s']:.3f}s | "
            f"XYZ raw={row['XYZ_raw_relerr']:.3e} | XYZ scaled={row['XYZ_scaled_relerr']:.3e} | "
            f"e-vs-e0={row['eccentric_vs_e0_bruteforce_xyz_relerr']:.3e}"
        )
        for channel in ("X", "Y", "Z"):
            print(
                f"  {channel}: raw={row[f'{channel}_raw_relerr']:.3e}, "
                f"scaled={row[f'{channel}_scaled_relerr']:.3e}, "
                f"peak-resid={row[f'{channel}_max_peak_residual']:.3e}, "
                f"|scale|={row[f'{channel}_scale_abs']:.3e}, "
                f"phase={row[f'{channel}_scale_phase']:.3f}"
            )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
