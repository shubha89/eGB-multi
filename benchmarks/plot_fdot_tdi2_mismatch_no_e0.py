"""Regenerate the paper mismatch diagnostic without the e=0 point."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open()))


def plot_mismatch_lines_no_e0(rows: list[dict[str, str]], path: Path) -> None:
    rows = [row for row in rows if float(row["eccentricity"]) > 0.0]
    frequencies = sorted({float(row["f0_hz"]) for row in rows})
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    column = "mismatch_AE"

    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    for color, f0_hz in zip(colors, frequencies, strict=True):
        medians = []
        for eccentricity in eccentricities:
            values = np.array(
                [
                    max(float(row[column]), 1.0e-16)
                    for row in rows
                    if float(row["f0_hz"]) == f0_hz and float(row["eccentricity"]) == eccentricity
                ],
                dtype=np.float64,
            )
            medians.append(np.median(values))

        x = np.array(eccentricities)
        ax.plot(
            x,
            medians,
            marker="o",
            linewidth=2.0,
            markersize=4.4,
            color=color,
            label=f"$f_0={f0_hz:.0e}\\,\\mathrm{{Hz}}$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("A/E mismatch, $1-\\mathrm{FF}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, frameon=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = load_rows(root / "production_eccentric_circular_fdot_tdi2_signal_gr_template_grid_production.csv")
    benchmark_path = root / "wide_fdot_tdi2_mismatch_vs_eccentricity_by_f0_no_e0.png"
    manuscript_path = (
        Path("/Users/shubhanshutiwari/Downloads/Effect_of_eccentricity_for_GB_in_LISA-2/figures")
        / "wide_fdot_tdi2_mismatch_vs_eccentricity_by_f0_no_e0.png"
    )
    plot_mismatch_lines_no_e0(rows, benchmark_path)
    plot_mismatch_lines_no_e0(rows, manuscript_path)
    print(f"wrote {benchmark_path}")
    print(f"wrote {manuscript_path}")


if __name__ == "__main__":
    main()
