"""Mismatch-vs-eccentricity with color=f0 and style=chirp mass."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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

    colors = dict(zip(frequencies, plt.cm.viridis(np.linspace(0.08, 0.92, len(frequencies))), strict=True))
    linestyles = ["-", "--", ":"]
    markers = ["o", "s", "^"]
    style_by_mass = {
        pair: (linestyles[index % len(linestyles)], markers[index % len(markers)])
        for index, pair in enumerate(mass_pairs)
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.6), constrained_layout=True)

    for f0 in frequencies:
        for pair in mass_pairs:
            m1, m2 = pair
            medians = []
            for eccentricity in eccentricities:
                values = np.array(
                    [
                        max(float(row["mismatch_AE"]), 1.0e-16)
                        for row in rows
                        if float(row["f0_hz"]) == f0
                        and float(row["eccentricity"]) == eccentricity
                        and float(row["m1_solar"]) == m1
                        and float(row["m2_solar"]) == m2
                    ],
                    dtype=np.float64,
                )
                medians.append(np.median(values))
            linestyle, marker = style_by_mass[pair]
            ax.plot(
                eccentricities,
                medians,
                color=colors[f0],
                linestyle=linestyle,
                marker=marker,
                linewidth=1.35,
                markersize=3.5,
                alpha=0.95,
            )

    ax.set_xscale("symlog", linthresh=1.0e-3)
    ax.set_yscale("log")
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("A/E mismatch, $1-\\mathrm{FF}$")
    ax.set_title("Full-response eccentric vs circular mismatch")
    ax.grid(True, which="both", alpha=0.3)

    color_handles = [
        Line2D([0], [0], color=colors[f0], linewidth=2.0, label=f"$f_0={f0:.0e}\\,\\mathrm{{Hz}}$")
        for f0 in frequencies
    ]
    mass_handles = []
    for pair in mass_pairs:
        linestyle, marker = style_by_mass[pair]
        cm = chirp_mass(*pair)
        mass_handles.append(
            Line2D(
                [0],
                [0],
                color="0.25",
                linestyle=linestyle,
                marker=marker,
                linewidth=1.5,
                markersize=4.0,
                label=f"$\\mathcal{{M}}_c={cm:.2f}M_\\odot$ ({pair[0]:.1f}+{pair[1]:.1f})",
            )
        )

    legend_f0 = ax.legend(handles=color_handles, title="frequency", fontsize=8, title_fontsize=8, loc="upper left", ncol=2)
    ax.add_artist(legend_f0)
    ax.legend(handles=mass_handles, title="mass", fontsize=8, title_fontsize=8, loc="lower right")

    png_path = out_dir / "production_mismatch_vs_eccentricity_f0_mass_styles.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
