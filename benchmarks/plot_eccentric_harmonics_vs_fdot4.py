"""Overlay eccentric source harmonics with a circular fdot=4*GR reference.

This is a standalone diagnostic plot and is not copied into the manuscript.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.constants import C_M_PER_S, G_M3_KG_S2, SOLAR_MASS_KG
from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain


def chirp_mass_solar(m1_solar: float, m2_solar: float) -> float:
    total = m1_solar + m2_solar
    return (m1_solar * m2_solar) ** (3.0 / 5.0) / total ** (1.0 / 5.0)


def circular_gr_fdot_hz_s(f0_hz: float, m1_solar: float, m2_solar: float) -> float:
    chirp_mass_kg = chirp_mass_solar(m1_solar, m2_solar) * SOLAR_MASS_KG
    return float(
        (96.0 / 5.0)
        * np.pi ** (8.0 / 3.0)
        * (G_M3_KG_S2 * chirp_mass_kg / C_M_PER_S**3) ** (5.0 / 3.0)
        * f0_hz ** (11.0 / 3.0)
    )


def source(f0_hz: float, eccentricity: float, *, fdot: float = 0.0) -> EccentricBinaryParams:
    return EccentricBinaryParams(
        mean_motion=np.pi * f0_hz,
        eccentricity=eccentricity,
        m1_solar=0.5,
        m2_solar=0.5,
        inclination=0.8,
        beta=0.2,
        lambda_=-1.0,
        psi=0.4,
        phi0=0.2,
        fdot=fdot,
    )


def positive_fft(series: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(series.size)
    spectrum = np.fft.rfft((series - np.mean(series)) * window) * dt
    freq = np.fft.rfftfreq(series.size, dt)
    return freq, np.abs(spectrum)


def monochromatic_reference(times: np.ndarray, f0_hz: float, fdot: float = 0.0) -> np.ndarray:
    phase = 2.0 * np.pi * f0_hz * times + np.pi * fdot * times * times + 0.2
    return np.cos(phase)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    f0_hz = 1.0e-2
    m1 = m2 = 0.5
    gr_fdot = circular_gr_fdot_hz_s(f0_hz, m1, m2)
    fdot4 = 4.0 * gr_fdot
    duration = 365.25 * 86400.0
    dt = 5.0
    times = np.arange(0.0, duration, dt)

    cases = [
        ("monochromatic, $\\dot f=0$", monochromatic_reference(times, f0_hz), "#D62728", 0.72, 1.4),
        (
            "monochromatic, $\\dot f=4\\dot f_{\\rm GR}$",
            monochromatic_reference(times, f0_hz, fdot4),
            "#000000",
            0.92,
            1.7,
        ),
        ("eccentric source, $e=0.1$", np.real(eccentric_complex_strain(source(f0_hz, 0.1), times, physics_mode="newtonian")), "#0072B2", 0.78, 1.15),
        ("eccentric source, $e=0.3$", np.real(eccentric_complex_strain(source(f0_hz, 0.3), times, physics_mode="newtonian")), "#009E73", 0.78, 1.15),
    ]

    curves = []
    for label, h_plus, color, alpha, linewidth in cases:
        freq, amp = positive_fft(h_plus, dt)
        curves.append((label, freq / f0_hz, amp / np.max(amp), color, alpha, linewidth))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    ax, zoom_ax = axes
    for label, x, amp, color, alpha, linewidth in curves:
        mask = (x >= 0.25) & (x <= 5.0)
        ax.plot(
            x[mask],
            np.maximum(amp[mask], 1.0e-12),
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )
        zoom = (x >= 0.99988) & (x <= 1.00014)
        zoom_ax.plot(
            x[zoom],
            np.maximum(amp[zoom], 1.0e-12),
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )
    for harmonic in range(1, 11):
        ax.axvline(0.5 * harmonic, color="0.75", linewidth=0.65, alpha=0.35)
    ax.set_yscale("log")
    ax.set_xlim(0.25, 5.0)
    ax.set_ylim(1.0e-9, 2.0)
    ax.set_xlabel("$f/f_0$")
    ax.set_ylabel("self-normalized FFT amplitude")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(fontsize=8, frameon=True)
    ax.set_title("harmonic content")

    zoom_ax.set_yscale("log")
    zoom_ax.set_xlim(0.99988, 1.00014)
    zoom_ax.set_ylim(1.0e-6, 2.0)
    zoom_ax.set_xlabel("$f/f_0$")
    zoom_ax.set_ylabel("self-normalized FFT amplitude")
    zoom_ax.ticklabel_format(axis="x", useOffset=False)
    zoom_ax.grid(True, which="both", alpha=0.28)
    zoom_ax.set_title("carrier zoom")
    ax.text(
        0.98,
        0.04,
        f"$f_0={f0_hz:g}\\,\\mathrm{{Hz}}$, $T=1\\,\\mathrm{{yr}}$, $\\dot f_{{\\rm GR}}={gr_fdot:.2e}\\,\\mathrm{{Hz/s}}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    path = out_dir / "eccentric_harmonics_vs_circular_fdot4.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
