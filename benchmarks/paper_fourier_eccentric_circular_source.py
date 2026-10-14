"""Fourier-domain source comparison for circular and eccentric GB waveforms."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain


PAPER_FIGURE_DIR = Path("/Users/shubhanshutiwari/Downloads/Effect_of_eccentricity_for_GB_in_LISA-2/figures")


def source(eccentricity: float, f0_hz: float) -> EccentricBinaryParams:
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
    )


def positive_fft(series: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.fft.fft(series) * dt
    freq = np.fft.fftfreq(series.size, dt)
    mask = freq > 0.0
    return freq[mask], np.abs(spectrum[mask])


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    f0_hz = 1.0e-3
    orbital_period = 2.0 / f0_hz
    periods = 512
    dt = 5.0
    times = np.arange(0.0, periods * orbital_period, dt)
    eccentricities = [0.0, 0.05, 0.1, 0.3]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    reference_peak = None
    offsets = [-0.045, -0.015, 0.015, 0.045]
    harmonic_positions = np.arange(1, 9, dtype=np.float64) / 2.0
    for eccentricity, color, offset in zip(eccentricities, colors, offsets, strict=True):
        strain = eccentric_complex_strain(source(eccentricity, f0_hz), times, physics_mode="newtonian")
        freq, amp = positive_fft(strain, dt)
        if reference_peak is None:
            reference_peak = np.max(amp)
        amp = amp / reference_peak
        harmonic_amp = []
        for x in harmonic_positions:
            index = int(np.argmin(np.abs(freq / f0_hz - x)))
            harmonic_amp.append(max(float(amp[index]), 1.0e-12))
        x_values = harmonic_positions + offset
        ax.vlines(x_values, 1.0e-7, harmonic_amp, color=color, linewidth=2.0, alpha=0.85)
        ax.scatter(
            x_values,
            harmonic_amp,
            s=36,
            color=color,
            label=f"$e={eccentricity:g}$",
            zorder=3,
        )

    for harmonic in range(1, 9):
        ax.axvline(0.5 * harmonic, color="0.75", linewidth=0.7, alpha=0.45)
    ax.set_yscale("log")
    ax.set_ylim(1.0e-7, 2.0)
    ax.set_xlim(0.25, 4.0)
    ax.set_xticks(harmonic_positions)
    ax.set_xticklabels([f"{x:g}" for x in harmonic_positions])
    ax.set_xlabel("frequency / circular $f_0$")
    ax.set_ylabel("normalized Fourier amplitude")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=True, fontsize=9)

    path = out_dir / "paper_fourier_eccentric_circular_source.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    target = PAPER_FIGURE_DIR / path.name
    shutil.copy2(path, target)
    print(f"wrote {path}")
    print(f"copied {target}")


if __name__ == "__main__":
    main()
