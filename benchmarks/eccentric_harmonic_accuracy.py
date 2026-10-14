"""Source-side residuals for eccentric harmonic truncation.

Run with:

    python benchmarks/eccentric_harmonic_accuracy.py

Outputs:

    benchmarks/eccentric_harmonic_accuracy.csv
    benchmarks/eccentric_harmonic_accuracy.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import (
    EccentricBinaryParams,
    eccentric_harmonic_angular_frequency,
    eccentric_harmonic_indices,
    eccentric_harmonic_projection_angular_frequency,
    eccentric_polarizations,
)


def analytic_reconstruction(signal: np.ndarray, harmonics: tuple[int, ...]) -> np.ndarray:
    coeffs = np.fft.fft(signal) / signal.size
    j = np.arange(signal.size)
    out = np.zeros(signal.size, dtype=np.complex128)
    for harmonic in harmonics:
        out += 2.0 * coeffs[harmonic] * np.exp(2j * np.pi * harmonic * j / signal.size)
    return np.real(out)


def sideband_reconstruction(source: EccentricBinaryParams, signal: np.ndarray, times: np.ndarray, harmonics: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(signal.size, dtype=np.complex128)
    for harmonic in harmonics:
        angular_frequency = eccentric_harmonic_angular_frequency(source, harmonic)
        coeff = np.mean(signal * np.exp(-1.0j * angular_frequency * times))
        out += coeff * np.exp(1.0j * angular_frequency * times)
    return 2.0 * np.real(out)


def relative_l2(residual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(residual) / np.linalg.norm(reference))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    eccentricities = [0.01, 0.03, 0.1, 0.3]
    orders = [0, 1, 2]
    samples = 4096
    base = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.0,
        m1_solar=0.6,
        m2_solar=0.4,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
        phi0=0.2,
    )
    rows = []
    for eccentricity in eccentricities:
        source = EccentricBinaryParams(**{**base.__dict__, "eccentricity": eccentricity})
        projection_frequency = eccentric_harmonic_projection_angular_frequency(source)
        period = 2.0 * np.pi / projection_frequency
        t = source.t0 + np.linspace(0.0, period, samples, endpoint=False)
        hplus, hcross = eccentric_polarizations(source, t)
        for order in orders:
            harmonics = eccentric_harmonic_indices(order)
            hplus_approx = sideband_reconstruction(source, hplus, t, harmonics)
            hcross_approx = sideband_reconstruction(source, hcross, t, harmonics)
            rows.append(
                {
                    "eccentricity": eccentricity,
                    "eccentric_order": order,
                    "hplus_relative_l2_residual": relative_l2(hplus - hplus_approx, hplus),
                    "hcross_relative_l2_residual": relative_l2(hcross - hcross_approx, hcross),
                    "harmonics": " ".join(str(h) for h in harmonics),
                }
            )

    csv_path = out_dir / "eccentric_harmonic_accuracy.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True, constrained_layout=True)
    for order in orders:
        subset = [row for row in rows if row["eccentric_order"] == order]
        x = [row["eccentricity"] for row in subset]
        axes[0].plot(x, [row["hplus_relative_l2_residual"] for row in subset], marker="o", label=f"order={order}")
        axes[1].plot(x, [row["hcross_relative_l2_residual"] for row in subset], marker="o", label=f"order={order}")
    axes[0].set_title("plus")
    axes[1].set_title("cross")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("eccentricity")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("relative L2 residual")

    png_path = out_dir / "eccentric_harmonic_accuracy.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    for row in rows:
        print(
            f"e={row['eccentricity']:.2f}, order={row['eccentric_order']}: "
            f"h_plus={row['hplus_relative_l2_residual']:.3e}, "
            f"h_cross={row['hcross_relative_l2_residual']:.3e}"
        )
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
