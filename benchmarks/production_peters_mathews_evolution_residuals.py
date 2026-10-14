"""Residual and mismatch decomposition for Peters-Mathews source evolution.

This diagnostic asks which part of Peters-Mathews evolution matters for the
eccentric Galactic-binary waveforms used in the paper:

* orbital-only: evolve mean motion and accumulated mean anomaly, hold e fixed;
* eccentricity-only: evolve e, hold the orbital clock fixed;
* full PM: evolve both.

The calculation is source-level, not full LISA response, so that it can scan a
small production grid quickly. The noise-weighted mismatch uses the measured
LISA-instrument A-channel PSD as a weighting proxy.

Run with:

    python benchmarks/production_peters_mathews_evolution_residuals.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
MODES = {
    "fixed": "fixed",
    "orbital_only": "peters_mathews_orbital_only",
    "eccentricity_only": "peters_mathews_eccentricity_only",
    "full_pm": "peters_mathews",
}
COMPARISONS = {
    "full_pm_vs_fixed": ("full_pm", "fixed"),
    "orbital_only_vs_fixed": ("orbital_only", "fixed"),
    "eccentricity_only_vs_fixed": ("eccentricity_only", "fixed"),
    "full_pm_vs_orbital_only": ("full_pm", "orbital_only"),
    "full_pm_vs_eccentricity_only": ("full_pm", "eccentricity_only"),
}


def load_a_channel_psd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    return (
        np.array([float(row["frequency_hz"]) for row in rows], dtype=np.float64),
        np.array([float(row["PSD_A"]) for row in rows], dtype=np.float64),
    )


def rfft_real(series: np.ndarray, dt: float) -> np.ndarray:
    return np.fft.rfft(np.asarray(series, dtype=np.float64)) * dt


def complex_inner(left: np.ndarray, right: np.ndarray, psd: np.ndarray, df: float, mask: np.ndarray) -> complex:
    valid = mask & np.isfinite(psd) & (psd > 0.0)
    if valid.size:
        valid[0] = False
    return complex(4.0 * np.sum(left[valid] * np.conjugate(right[valid]) / psd[valid]) * df)


def fd_polarizations(strain: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    return rfft_real(np.real(strain), dt), rfft_real(-np.imag(strain), dt)


def two_pol_inner(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
    psd: np.ndarray,
    df: float,
    mask: np.ndarray,
) -> complex:
    return complex_inner(left[0], right[0], psd, df, mask.copy()) + complex_inner(left[1], right[1], psd, df, mask.copy())


def compare_strains(reference: np.ndarray, template: np.ndarray, psd: np.ndarray, dt: float, mask: np.ndarray) -> dict[str, float]:
    df = 1.0 / (reference.size * dt)
    ref_fd = fd_polarizations(reference, dt)
    tmp_fd = fd_polarizations(template, dt)
    diff_fd = (ref_fd[0] - tmp_fd[0], ref_fd[1] - tmp_fd[1])
    rr = two_pol_inner(ref_fd, ref_fd, psd, df, mask).real
    tt = two_pol_inner(tmp_fd, tmp_fd, psd, df, mask).real
    rt = two_pol_inner(ref_fd, tmp_fd, psd, df, mask)
    dd = two_pol_inner(diff_fd, diff_fd, psd, df, mask).real
    ff = float(abs(rt) / np.sqrt(rr * tt)) if rr > 0.0 and tt > 0.0 else 0.0
    ff = min(ff, 1.0)
    return {
        "relative_l2_residual": float(np.linalg.norm(reference - template) / np.linalg.norm(reference)),
        "noise_weighted_residual": float(np.sqrt(max(dd, 0.0) / rr)) if rr > 0.0 else np.nan,
        "fitting_factor": ff,
        "mismatch": max(1.0 - ff, 0.0),
    }


def make_source(f0_hz: float, eccentricity: float, m1_solar: float, m2_solar: float) -> EccentricBinaryParams:
    return EccentricBinaryParams(
        mean_motion=np.pi * f0_hz,
        eccentricity=eccentricity,
        m1_solar=m1_solar,
        m2_solar=m2_solar,
        beta=0.3,
        lambda_=-0.7,
        psi=0.4,
        inclination=0.8,
        phi0=0.2,
    )


def chirp_mass_solar(m1_solar: float, m2_solar: float) -> float:
    total = m1_solar + m2_solar
    return (m1_solar * m2_solar) ** (3.0 / 5.0) / total ** (1.0 / 5.0)


def run_grid() -> list[dict[str, object]]:
    out_dir = Path(__file__).resolve().parent
    psd_freq, psd_a = load_a_channel_psd(out_dir / "lisainstrument_welch_psd.csv")

    duration = 4.0 * SECONDS_PER_YEAR
    dt = 50.0
    times = np.arange(0.0, duration, dt, dtype=np.float64)
    freq = np.fft.rfftfreq(times.size, dt)
    psd = np.interp(freq, psd_freq, psd_a, left=np.inf, right=np.inf)

    frequencies = [2.0e-4, 1.0e-3, 3.0e-3]
    eccentricities = [0.1, 0.3]
    masses = [(0.5, 0.5), (0.8, 0.6), (1.4, 1.4)]

    rows: list[dict[str, object]] = []
    total = len(frequencies) * len(eccentricities) * len(masses)
    case_index = 0
    for f0_hz in frequencies:
        band_mask = (freq >= 0.20 * f0_hz) & (freq <= min(freq[-1], 10.0 * f0_hz))
        for eccentricity in eccentricities:
            for m1_solar, m2_solar in masses:
                case_index += 1
                source = make_source(f0_hz, eccentricity, m1_solar, m2_solar)
                started = time.perf_counter()
                strains = {
                    label: eccentric_complex_strain(source, times, evolution_mode=evolution_mode)
                    for label, evolution_mode in MODES.items()
                }
                waveform_wall_s = time.perf_counter() - started
                for comparison, (reference_label, template_label) in COMPARISONS.items():
                    metrics = compare_strains(strains[reference_label], strains[template_label], psd, dt, band_mask.copy())
                    rows.append(
                        {
                            "duration_years": 4.0,
                            "dt_s": dt,
                            "samples": times.size,
                            "f0_hz": f0_hz,
                            "eccentricity": eccentricity,
                            "m1_solar": m1_solar,
                            "m2_solar": m2_solar,
                            "chirp_mass_solar": chirp_mass_solar(m1_solar, m2_solar),
                            "comparison": comparison,
                            "waveform_wall_s": waveform_wall_s,
                            **metrics,
                        }
                    )
                print(
                    f"[{case_index:02d}/{total}] f0={f0_hz:.1e} e={eccentricity:.1f} "
                    f"m={m1_solar:.1f}+{m2_solar:.1f} wall={waveform_wall_s:.2f}s",
                    flush=True,
                )
    return rows


def plot_rows(rows: list[dict[str, object]], path: Path) -> None:
    comparison_order = [
        "full_pm_vs_fixed",
        "orbital_only_vs_fixed",
        "eccentricity_only_vs_fixed",
        "full_pm_vs_orbital_only",
        "full_pm_vs_eccentricity_only",
    ]
    labels = {
        "full_pm_vs_fixed": "full PM - fixed",
        "orbital_only_vs_fixed": "orbital only - fixed",
        "eccentricity_only_vs_fixed": "e only - fixed",
        "full_pm_vs_orbital_only": "full PM - orbital only",
        "full_pm_vs_eccentricity_only": "full PM - e only",
    }
    colors = {
        "full_pm_vs_fixed": "#000000",
        "orbital_only_vs_fixed": "#0072B2",
        "eccentricity_only_vs_fixed": "#D55E00",
        "full_pm_vs_orbital_only": "#CC79A7",
        "full_pm_vs_eccentricity_only": "#009E73",
    }
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})
    fig, axes = plt.subplots(2, len(eccentricities), figsize=(6.2 * len(eccentricities), 8.0), constrained_layout=True, squeeze=False)
    for col, eccentricity in enumerate(eccentricities):
        subset_e = [row for row in rows if float(row["eccentricity"]) == eccentricity]
        frequencies = np.array(sorted({float(row["f0_hz"]) for row in subset_e}))
        for comparison in comparison_order:
            med_mismatch = []
            med_residual = []
            for f0_hz in frequencies:
                values = [
                    row
                    for row in subset_e
                    if float(row["f0_hz"]) == f0_hz and row["comparison"] == comparison
                ]
                med_mismatch.append(np.median([max(float(row["mismatch"]), 1.0e-16) for row in values]))
                med_residual.append(np.median([max(float(row["relative_l2_residual"]), 1.0e-16) for row in values]))
            axes[0, col].plot(frequencies * 1.0e3, med_mismatch, marker="o", color=colors[comparison], label=labels[comparison])
            axes[1, col].plot(frequencies * 1.0e3, med_residual, marker="o", color=colors[comparison], label=labels[comparison])
        for ax in axes[:, col]:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("dominant GW frequency f0 [mHz]")
            ax.grid(True, which="both", alpha=0.25)
        axes[0, col].text(0.04, 0.92, f"e0={eccentricity:g}", transform=axes[0, col].transAxes)
    axes[0, 0].set_ylabel("noise-weighted mismatch, 1 - FF")
    axes[1, 0].set_ylabel("relative L2 residual")
    axes[0, -1].legend(fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows = run_grid()
    csv_path = out_dir / "production_peters_mathews_evolution_residuals.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    png_path = out_dir / "production_peters_mathews_evolution_residuals.png"
    plot_rows(rows, png_path)
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
