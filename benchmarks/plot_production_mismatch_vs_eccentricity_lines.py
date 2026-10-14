"""Single-panel mismatch-vs-eccentricity summary for the production grid."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def chirp_mass(m1: float, m2: float) -> float:
    total = m1 + m2
    eta = m1 * m2 / total**2
    return total * eta ** (3.0 / 5.0)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "production_eccentric_circular_mismatch_grid_production.csv"
    rows = list(csv.DictReader(csv_path.open()))

    frequencies = sorted({float(row["f0_hz"]) for row in rows})
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})
    mass_pairs = sorted({(float(row["m1_solar"]), float(row["m2_solar"])) for row in rows}, key=lambda pair: chirp_mass(*pair))
    chirp_masses = [chirp_mass(*pair) for pair in mass_pairs]

    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    # Okabe-Ito color-blind-friendly palette.
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

    for color, f0 in zip(colors[: len(frequencies)], frequencies, strict=True):
        medians = []
        lows = []
        highs = []
        for eccentricity in eccentricities:
            values = np.array(
                [
                    max(float(row["mismatch_AE"]), 1.0e-16)
                    for row in rows
                    if float(row["f0_hz"]) == f0 and float(row["eccentricity"]) == eccentricity
                ],
                dtype=np.float64,
            )
            medians.append(np.median(values))
            lows.append(np.percentile(values, 16.0))
            highs.append(np.percentile(values, 84.0))

        x = np.array(eccentricities)
        med = np.array(medians)
        lo = np.array(lows)
        hi = np.array(highs)
        label = f"$f_0={f0:.0e}\\,\\mathrm{{Hz}}$"
        ax.plot(x, med, marker="o", linewidth=2.0, markersize=4.4, color=color, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)

    ax.set_xscale("symlog", linthresh=1.0e-3)
    ax.set_yscale("log")
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("A/E mismatch, $1-\\mathrm{FF}$")
    ax.set_title("Full-response eccentric vs circular mismatch")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, frameon=True)

    png_path = out_dir / "production_mismatch_vs_eccentricity_by_f0.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
