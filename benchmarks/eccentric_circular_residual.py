"""Plot eccentric source waveforms against the circular limit.

Run with:

    python benchmarks/eccentric_circular_residual.py

Outputs:

    benchmarks/eccentric_circular_residual.csv
    benchmarks/eccentric_circular_residual.png
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import EccentricBinaryParams, eccentric_polarizations


def relative_l2(residual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(residual) / np.linalg.norm(reference))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    eccentricities = [0.1, 0.3, 0.5, 0.8]
    base = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.0,
        m1_solar=0.6,
        m2_solar=0.4,
        inclination=0.8,
        phi0=0.2,
    )
    orbital_period = 2.0 * np.pi / base.mean_motion
    t = np.linspace(0.0, 5.0 * orbital_period, 4096, endpoint=False)
    x = t / orbital_period

    hplus_circular, hcross_circular = eccentric_polarizations(base, t)
    circular_norm = max(np.max(np.abs(hplus_circular)), np.max(np.abs(hcross_circular)))

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True, constrained_layout=True)
    ax_hp, ax_hc, ax_rp, ax_rc = axes.ravel()

    ax_hp.plot(x, hplus_circular / circular_norm, color="black", linewidth=1.5, label="circular")
    ax_hc.plot(x, hcross_circular / circular_norm, color="black", linewidth=1.5, label="circular")

    for eccentricity in eccentricities:
        source = replace(base, eccentricity=eccentricity)
        hplus, hcross = eccentric_polarizations(source, t)
        rplus = hplus - hplus_circular
        rcross = hcross - hcross_circular
        rows.append(
            {
                "eccentricity": eccentricity,
                "hplus_relative_l2_residual": relative_l2(rplus, hplus_circular),
                "hcross_relative_l2_residual": relative_l2(rcross, hcross_circular),
                "hplus_max_abs_residual": float(np.max(np.abs(rplus))),
                "hcross_max_abs_residual": float(np.max(np.abs(rcross))),
            }
        )

        label = f"e={eccentricity:.1f}"
        ax_hp.plot(x, hplus / circular_norm, linewidth=1.0, label=label)
        ax_hc.plot(x, hcross / circular_norm, linewidth=1.0, label=label)
        ax_rp.plot(x, rplus / circular_norm, linewidth=1.0, label=label)
        ax_rc.plot(x, rcross / circular_norm, linewidth=1.0, label=label)

    ax_hp.set_title("plus polarization")
    ax_hc.set_title("cross polarization")
    ax_rp.set_title("plus residual: eccentric - circular")
    ax_rc.set_title("cross residual: eccentric - circular")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("strain / max circular strain")
    ax_rp.set_xlabel("time / orbital period")
    ax_rc.set_xlabel("time / orbital period")
    ax_hp.legend(fontsize=8, ncols=2)
    ax_hc.legend(fontsize=8, ncols=2)

    csv_path = out_dir / "eccentric_circular_residual.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    png_path = out_dir / "eccentric_circular_residual.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print("relative L2 residuals against e=0 circular waveform")
    for row in rows:
        print(
            f"  e={row['eccentricity']:.1f}: "
            f"h_plus={row['hplus_relative_l2_residual']:.3f}, "
            f"h_cross={row['hcross_relative_l2_residual']:.3f}"
        )
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
