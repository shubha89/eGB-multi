"""Monte Carlo circular vs eccentric Galactic-binary foreground comparison.

This is a toy unresolved foreground experiment. It draws one population of
quasi-monochromatic Galactic binaries, computes the circular foreground and an
eccentric foreground with fixed eccentricity, and compares binned A/E PSDs.

Run with:

    python benchmarks/monte_carlo_circular_eccentric_confusion.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.constants import KILOPARSEC_M
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import FastGBBackend, default_lisaorbits
from egb_multi.fastgb_convention import (
    circular_component_from_fastgb_source,
    eccentric_components_for_fastgb_response,
    frequency_domain_xyz_from_components,
)
from egb_multi.sources import GalacticBinaryParams


def draw_population(n_sources: int, *, seed: int = 1234) -> list[GalacticBinaryParams]:
    rng = np.random.default_rng(seed)
    log_f = rng.uniform(np.log10(3.0e-4), np.log10(5.0e-3), n_sources)
    f0 = 10.0**log_f
    distance_factor = 10.0 ** rng.uniform(-0.5, 0.5, n_sources)
    amplitude = 1.0e-22 * (f0 / 1.0e-3) ** (2.0 / 3.0) / distance_factor
    return [
        GalacticBinaryParams(
            f0=float(f0[i]),
            amplitude=float(amplitude[i]),
            beta=float(np.arcsin(rng.uniform(-1.0, 1.0))),
            lambda_=float(rng.uniform(-np.pi, np.pi)),
            psi=float(rng.uniform(0.0, np.pi)),
            inclination=float(np.arccos(rng.uniform(-1.0, 1.0))),
            phi0=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        for i in range(n_sources)
    ]


def circular_to_eccentric_source(source: GalacticBinaryParams, eccentricity: float) -> EccentricBinaryParams:
    # Equal 0.5 Msun masses at a variable effective distance chosen so the
    # circular-limit calibration maps back to the drawn FastGB amplitude.
    # The calibration is applied component-wise below, so distance only sets the
    # raw PN scale before calibration.
    return EccentricBinaryParams(
        mean_motion=np.pi * source.f0,
        eccentricity=eccentricity,
        m1_solar=0.5,
        m2_solar=0.5,
        distance_m=KILOPARSEC_M,
        beta=source.beta,
        lambda_=source.lambda_,
        psi=source.psi,
        inclination=source.inclination,
        phi0=source.phi0,
    )


def xyz_to_aet(result) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = result.x[0]
    y = result.y[0]
    z = result.z[0]
    a = (z - x) / np.sqrt(2.0)
    e = (x - 2.0 * y + z) / np.sqrt(6.0)
    t = (x + y + z) / np.sqrt(3.0)
    frequencies = (int(result.kmin[0]) + np.arange(x.size, dtype=np.float64)) / result.x.size
    return a, e, t, frequencies


def fd_psd(values: np.ndarray, duration: float) -> np.ndarray:
    return 2.0 * np.abs(values) ** 2 / duration


def binned_mean(frequency: np.ndarray, values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    out = np.full(bins.size - 1, np.nan, dtype=np.float64)
    for i in range(bins.size - 1):
        mask = (frequency >= bins[i]) & (frequency < bins[i + 1]) & np.isfinite(values)
        if np.any(mask):
            out[i] = float(np.mean(values[mask]))
    return out


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    n_sources = 2000
    eccentricity = 0.1
    eccentric_order = 2
    harmonic_samples = 2048
    duration = 365.25 * 24.0 * 3600.0
    response_bins = 256

    backend = FastGBBackend(
        delta_t=10.0,
        duration=duration,
        n_samples=response_bins,
        orbits=default_lisaorbits("equal"),
    )
    sources = draw_population(n_sources)

    start = time.perf_counter()
    circular_components = [circular_component_from_fastgb_source(source) for source in sources]
    circular = frequency_domain_xyz_from_components(backend, circular_components)
    circular_s = time.perf_counter() - start

    start = time.perf_counter()
    eccentric_components = []
    for source in sources:
        eccentric_components.extend(
            eccentric_components_for_fastgb_response(
                circular_to_eccentric_source(source, eccentricity),
                order=eccentric_order,
                samples=harmonic_samples,
                circular_reference=source,
            )
        )
    eccentric = frequency_domain_xyz_from_components(backend, eccentric_components)
    eccentric_s = time.perf_counter() - start

    df = 1.0 / duration
    circular_freq = (int(circular.kmin[0]) + np.arange(circular.x.shape[-1], dtype=np.float64)) * df
    eccentric_freq = (int(eccentric.kmin[0]) + np.arange(eccentric.x.shape[-1], dtype=np.float64)) * df
    circular_a = (circular.z[0] - circular.x[0]) / np.sqrt(2.0)
    circular_e = (circular.x[0] - 2.0 * circular.y[0] + circular.z[0]) / np.sqrt(6.0)
    eccentric_a = (eccentric.z[0] - eccentric.x[0]) / np.sqrt(2.0)
    eccentric_e = (eccentric.x[0] - 2.0 * eccentric.y[0] + eccentric.z[0]) / np.sqrt(6.0)

    bins = np.logspace(np.log10(2.0e-4), np.log10(1.2e-2), 120)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    circ_a_psd = binned_mean(circular_freq, fd_psd(circular_a, duration), bins)
    circ_e_psd = binned_mean(circular_freq, fd_psd(circular_e, duration), bins)
    ecc_a_psd = binned_mean(eccentric_freq, fd_psd(eccentric_a, duration), bins)
    ecc_e_psd = binned_mean(eccentric_freq, fd_psd(eccentric_e, duration), bins)
    circ_ae_psd = 0.5 * (circ_a_psd + circ_e_psd)
    ecc_ae_psd = 0.5 * (ecc_a_psd + ecc_e_psd)
    ratio = ecc_ae_psd / circ_ae_psd

    rows = []
    for i, frequency in enumerate(bin_centers):
        rows.append(
            {
                "frequency_hz": frequency,
                "circular_A_psd": circ_a_psd[i],
                "circular_E_psd": circ_e_psd[i],
                "circular_AE_mean_psd": circ_ae_psd[i],
                "eccentric_A_psd": ecc_a_psd[i],
                "eccentric_E_psd": ecc_e_psd[i],
                "eccentric_AE_mean_psd": ecc_ae_psd[i],
                "eccentric_over_circular_AE": ratio[i],
            }
        )

    csv_path = out_dir / "monte_carlo_circular_eccentric_confusion.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True)
    axes[0].loglog(bin_centers, circ_ae_psd, label="circular AE mean", linewidth=1.6)
    axes[0].loglog(bin_centers, ecc_ae_psd, label=f"eccentric AE mean, e={eccentricity}", linewidth=1.6)
    axes[0].set_ylabel("binned foreground PSD [arb.]")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[0].set_title(f"Monte Carlo GB foreground, N={n_sources}")

    axes[1].semilogx(bin_centers, ratio, color="tab:red")
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("eccentric / circular")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].set_ylim(0.0, np.nanpercentile(ratio, 95) * 1.2)

    png_path = out_dir / "monte_carlo_circular_eccentric_confusion.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"sources={n_sources}, eccentricity={eccentricity}, eccentric_order={eccentric_order}")
    print(f"circular response time: {circular_s:.3f} s")
    print(f"eccentric response time: {eccentric_s:.3f} s")
    print(f"median eccentric/circular AE ratio: {np.nanmedian(ratio):.3f}")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
