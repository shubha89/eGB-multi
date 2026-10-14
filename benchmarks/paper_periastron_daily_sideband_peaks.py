"""Binned Fourier sideband diagnostic for a 1PN+periastron eccentric source."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import (  # noqa: E402
    EccentricBinaryParams,
    eccentric_complex_strain,
    eccentric_harmonic_angular_frequency,
)


PAPER_FIGURE_DIR = Path(__file__).resolve().parents[1] / "reports" / "Effect_of_eccentricity_for_GB_in_LISA" / "figures"
PHYSICS_MODE = "1pn+periastron"
EVOLUTION_MODE = "fixed"
MASS_SOLAR = 1.4


def source(eccentricity: float, f0_hz: float) -> EccentricBinaryParams:
    return EccentricBinaryParams(
        mean_motion=np.pi * f0_hz,
        eccentricity=eccentricity,
        m1_solar=MASS_SOLAR,
        m2_solar=MASS_SOLAR,
        inclination=0.8,
        beta=0.2,
        lambda_=-1.0,
        psi=0.4,
        phi0=0.2,
    )


def binned_spectrum(segment: np.ndarray, dt: float, *, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(segment.size)
    spectrum = np.fft.rfft((segment - np.mean(segment)) * window, n=n_fft) * dt
    freq = np.fft.rfftfreq(n_fft, dt)
    power = np.abs(spectrum) ** 2
    return freq, power


def sideband_peak_table(
    spectra: list[tuple[np.ndarray, np.ndarray]],
    *,
    src: EccentricBinaryParams,
    f0_hz: float,
    global_peak: float,
    segment_days: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for bin_index, (freq, power) in enumerate(spectra, start=1):
        start_day = (bin_index - 1) * segment_days
        stop_day = bin_index * segment_days
        midpoint_day = 0.5 * (start_day + stop_day)
        for harmonic in range(1, 8):
            expected_hz = eccentric_harmonic_angular_frequency(src, harmonic, physics_mode=PHYSICS_MODE) / (2.0 * np.pi)
            search_half_width_hz = 0.03 * f0_hz
            mask = np.abs(freq - expected_hz) <= search_half_width_hz
            if not np.any(mask):
                continue
            local_indices = np.nonzero(mask)[0]
            peak_index = local_indices[int(np.argmax(power[mask]))]
            rows.append(
                {
                    "bin": bin_index,
                    "start_day": start_day,
                    "stop_day": stop_day,
                    "midpoint_day": midpoint_day,
                    "harmonic": harmonic,
                    "expected_frequency_hz": float(expected_hz),
                    "expected_frequency_over_f0": float(expected_hz / f0_hz),
                    "measured_peak_frequency_hz": float(freq[peak_index]),
                    "measured_peak_frequency_over_f0": float(freq[peak_index] / f0_hz),
                    "relative_power": float(power[peak_index] / global_peak),
                }
            )
    return rows


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    f0_hz = 1.0e-3
    eccentricity = 0.5
    dt = 5.0
    total_days = 100
    segment_days = 5
    day_seconds = 24.0 * 3600.0
    times = np.arange(0.0, total_days * day_seconds, dt)
    src = source(eccentricity, f0_hz)
    strain = eccentric_complex_strain(src, times, physics_mode=PHYSICS_MODE, evolution_mode=EVOLUTION_MODE)
    h_plus = np.real(strain)

    samples_per_segment = int(round(segment_days * day_seconds / dt))
    segment_count = total_days // segment_days
    n_fft = 8 * samples_per_segment
    spectra = []
    for segment_index in range(segment_count):
        start = segment_index * samples_per_segment
        stop = start + samples_per_segment
        spectra.append(binned_spectrum(h_plus[start:stop], dt, n_fft=n_fft))
    global_peak = max(float(np.max(power)) for _, power in spectra)

    nrows, ncols = 5, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 10.0), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.ravel()
    expected_ratios = [
        eccentric_harmonic_angular_frequency(src, harmonic, physics_mode=PHYSICS_MODE) / (2.0 * np.pi * f0_hz)
        for harmonic in range(1, 8)
    ]
    newtonian_ratios = [0.5 * harmonic for harmonic in range(1, 8)]

    for segment_index, (ax, (freq, power)) in enumerate(zip(axes_flat, spectra, strict=True), start=1):
        ratio = freq / f0_hz
        mask = (ratio >= 0.25) & (ratio <= 3.75)
        ax.plot(ratio[mask], np.maximum(power[mask] / global_peak, 1.0e-14), color="#0072B2", linewidth=0.85)
        for x in newtonian_ratios:
            ax.axvline(x, color="0.78", linewidth=0.5, alpha=0.42)
        for x in expected_ratios:
            ax.axvline(x, color="#D55E00", linewidth=0.7, alpha=0.76)
        start_day = (segment_index - 1) * segment_days
        stop_day = segment_index * segment_days
        ax.text(0.04, 0.80, f"{start_day}-{stop_day} d", transform=ax.transAxes, fontsize=7, fontweight="bold")
        ax.set_yscale("log")
        ax.set_ylim(1.0e-12, 2.0)
        ax.grid(True, which="both", alpha=0.25)

    for ax in axes[:, 0]:
        ax.set_ylabel("rel. power", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("$f/f_0$")
    axes_flat[-1].set_xlim(0.25, 3.75)
    fig.suptitle(
        "$e=0.5$, $1.4M_\\odot+1.4M_\\odot$, 100-day waveform in 5-day FFT bins, fixed 1PN+periastron source",
        fontsize=10,
    )

    png_path = out_dir / "paper_periastron_sideband_peaks_e05_m14_m14_100days_5daybins.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    rows = sideband_peak_table(spectra, src=src, f0_hz=f0_hz, global_peak=global_peak, segment_days=segment_days)
    csv_path = out_dir / "paper_periastron_sideband_peaks_e05_m14_m14_100days_5daybins.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    target = PAPER_FIGURE_DIR / png_path.name
    shutil.copy2(png_path, target)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    print(f"copied {target}")


if __name__ == "__main__":
    main()
