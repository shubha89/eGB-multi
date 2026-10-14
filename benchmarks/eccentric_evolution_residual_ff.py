"""Residuals and fitting factors for eccentric source evolution.

This source-level diagnostic compares eccentric waveforms against quasi-circular
waveforms on the same 4-year grid. It does not include the LISA response; it is
meant to isolate the source eccentricity and Peters-Mathews evolution effects.

Run with:

    python benchmarks/eccentric_evolution_residual_ff.py
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


def load_a_channel_psd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return (
        np.array([float(row["frequency_hz"]) for row in rows], dtype=np.float64),
        np.array([float(row["PSD_A"]) for row in rows], dtype=np.float64),
    )


def rfft_real(series: np.ndarray, dt: float) -> np.ndarray:
    return np.fft.rfft(np.asarray(series, dtype=np.float64)) * dt


def complex_inner(a_fd: np.ndarray, b_fd: np.ndarray, psd: np.ndarray, df: float, mask: np.ndarray) -> complex:
    valid = mask & np.isfinite(psd) & (psd > 0.0)
    valid[0] = False
    return complex(4.0 * np.sum(a_fd[valid] * np.conjugate(b_fd[valid]) / psd[valid]) * df)


def fitting_factor(reference_fd: np.ndarray, template_fd: np.ndarray, psd: np.ndarray, df: float, mask: np.ndarray) -> float:
    rr = complex_inner(reference_fd, reference_fd, psd, df, mask).real
    tt = complex_inner(template_fd, template_fd, psd, df, mask).real
    if rr <= 0.0 or tt <= 0.0:
        return 0.0
    rt = complex_inner(reference_fd, template_fd, psd, df, mask)
    return float(abs(rt) / np.sqrt(rr * tt))


def two_polarization_inner(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
    psd: np.ndarray,
    df: float,
    mask: np.ndarray,
) -> complex:
    return complex_inner(left[0], right[0], psd, df, mask.copy()) + complex_inner(left[1], right[1], psd, df, mask.copy())


def two_polarization_fitting_factor(
    reference_fd: tuple[np.ndarray, np.ndarray],
    template_fd: tuple[np.ndarray, np.ndarray],
    psd: np.ndarray,
    df: float,
    mask: np.ndarray,
) -> float:
    rr = two_polarization_inner(reference_fd, reference_fd, psd, df, mask).real
    tt = two_polarization_inner(template_fd, template_fd, psd, df, mask).real
    if rr <= 0.0 or tt <= 0.0:
        return 0.0
    rt = two_polarization_inner(reference_fd, template_fd, psd, df, mask)
    return float(abs(rt) / np.sqrt(rr * tt))


def residuals(reference: np.ndarray, template: np.ndarray, psd: np.ndarray, dt: float, mask: np.ndarray) -> dict[str, float]:
    df = 1.0 / (reference.size * dt)
    reference_fd = (rfft_real(np.real(reference), dt), rfft_real(-np.imag(reference), dt))
    template_fd = (rfft_real(np.real(template), dt), rfft_real(-np.imag(template), dt))
    diff_fd = (reference_fd[0] - template_fd[0], reference_fd[1] - template_fd[1])
    rr = two_polarization_inner(reference_fd, reference_fd, psd, df, mask).real
    dd = two_polarization_inner(diff_fd, diff_fd, psd, df, mask).real
    return {
        "relative_l2_residual": float(np.linalg.norm(reference - template) / np.linalg.norm(reference)),
        "noise_weighted_residual": float(np.sqrt(max(dd, 0.0) / rr)) if rr > 0.0 else np.nan,
        "fitting_factor": two_polarization_fitting_factor(reference_fd, template_fd, psd, df, mask),
    }


def make_source(f0: float, eccentricity: float) -> EccentricBinaryParams:
    return EccentricBinaryParams(
        mean_motion=np.pi * f0,
        eccentricity=eccentricity,
        m1_solar=0.8,
        m2_solar=0.6,
        beta=0.3,
        lambda_=-0.7,
        psi=0.4,
        inclination=1.0,
        phi0=0.2,
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    psd_freq, psd_a = load_a_channel_psd(out_dir / "lisainstrument_welch_psd.csv")

    duration = 4.0 * SECONDS_PER_YEAR
    dt = 50.0
    t = np.arange(0.0, duration, dt)
    freq = np.fft.rfftfreq(t.size, dt)
    psd = np.interp(freq, psd_freq, psd_a, left=np.inf, right=np.inf)

    frequencies = [2.0e-4, 1.0e-3]
    eccentricities = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    rows: list[dict[str, float | str]] = []

    start_all = time.perf_counter()
    for f0 in frequencies:
        band_mask = (freq >= 0.25 * f0) & (freq <= min(freq[-1], 8.0 * f0))
        circular_fixed = eccentric_complex_strain(make_source(f0, 0.0), t, evolution_mode="fixed")
        circular_evolved = eccentric_complex_strain(make_source(f0, 0.0), t, evolution_mode="peters_mathews")

        for eccentricity in eccentricities:
            source = make_source(f0, eccentricity)
            fixed = eccentric_complex_strain(source, t, evolution_mode="fixed")
            evolved = eccentric_complex_strain(source, t, evolution_mode="peters_mathews")

            comparisons = {
                "fixed_eccentric_vs_fixed_circular": (fixed, circular_fixed),
                "evolved_eccentric_vs_evolved_circular": (evolved, circular_evolved),
                "evolved_eccentric_vs_fixed_circular": (evolved, circular_fixed),
            }
            for label, (reference, template) in comparisons.items():
                values = residuals(reference, template, psd, dt, band_mask.copy())
                row = {
                    "f0_hz": f0,
                    "eccentricity": eccentricity,
                    "comparison": label,
                    **values,
                    "mismatch": 1.0 - values["fitting_factor"],
                }
                rows.append(row)
            print(f"f0={f0:.1e} e={eccentricity:.2f} done")

    csv_path = out_dir / "eccentric_evolution_residual_ff.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True, sharex="col")
    styles = {
        "fixed_eccentric_vs_fixed_circular": ("fixed e vs fixed circular", "o"),
        "evolved_eccentric_vs_evolved_circular": ("evolved e vs evolved circular", "s"),
        "evolved_eccentric_vs_fixed_circular": ("evolved e vs fixed circular", "^"),
    }
    for col, f0 in enumerate(frequencies):
        subset_f = [row for row in rows if row["f0_hz"] == f0]
        for comparison, (label, marker) in styles.items():
            subset = [row for row in subset_f if row["comparison"] == comparison]
            x = np.array([float(row["eccentricity"]) for row in subset])
            axes[0, col].plot(
                x,
                [max(float(row["relative_l2_residual"]), 1.0e-16) for row in subset],
                marker=marker,
                label=label,
            )
            axes[1, col].plot(
                x,
                [max(float(row["mismatch"]), 1.0e-16) for row in subset],
                marker=marker,
                label=label,
            )
        axes[0, col].set_title(f"f0={f0:.1e} Hz")
        axes[1, col].set_xlabel("initial eccentricity")
        for row_axis in range(2):
            axes[row_axis, col].set_xscale("symlog", linthresh=1.0e-3)
            axes[row_axis, col].set_yscale("log")
            axes[row_axis, col].grid(True, which="both", alpha=0.3)
    axes[0, 0].set_ylabel("relative L2 residual")
    axes[1, 0].set_ylabel("noise-weighted mismatch, 1 - FF")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Eccentric source residuals and fitting factors, 4 yr, dt=50 s")
    png_path = out_dir / "eccentric_evolution_residual_ff.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wall_time_s={time.perf_counter() - start_all:.3f}")
    print(f"samples={t.size}, duration_years=4.0, dt={dt}")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
