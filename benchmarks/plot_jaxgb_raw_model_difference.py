"""Plot raw JAXGBaccurate/local time-domain model differences.

This figure intentionally keeps the circular-limit offset in the raw
JAXGBaccurate comparison. It contrasts that raw model difference with the
local eccentric-vs-local-circular time-domain mismatch.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
FIGURES = ROOT / "reports" / "Effect_of_eccentricity_for_GB_in_LISA-new" / "figures"
INPUT = BENCHMARKS / "jaxgbaccurate_calibrated_eccentric_study_1pn_periastron.csv"
OUTPUT = BENCHMARKS / "paper_jaxgb_raw_model_difference.png"
PAPER_OUTPUT = FIGURES / OUTPUT.name


COLORS = {
    1.0e-4: "#0072B2",
    3.0e-4: "#009E73",
    1.0e-3: "#D55E00",
    3.0e-3: "#CC79A7",
}


def x_position(eccentricity: float) -> float:
    return 2.0e-5 if np.isclose(eccentricity, 0.0) else eccentricity


def main() -> None:
    rows = list(csv.DictReader(INPUT.open()))
    frequencies = sorted({float(row["f0_hz"]) for row in rows})

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)

    for f0 in frequencies:
        subset = sorted(
            [row for row in rows if np.isclose(float(row["f0_hz"]), f0)],
            key=lambda row: float(row["eccentricity"]),
        )
        x = np.array([x_position(float(row["eccentricity"])) for row in subset])
        raw = np.array([max(float(row["mismatch_AE"]), 1.0e-12) for row in subset])
        local = np.array([max(float(row["local_e0_mismatch_AE"]), 1.0e-12) for row in subset])
        color = COLORS.get(f0, None)
        label = f"{f0:.0e} Hz"

        axes[0].plot(x, raw, marker="o", linewidth=1.9, markersize=4.5, color=color, label=label)
        axes[0].axhline(raw[0], color=color, linewidth=1.0, alpha=0.35, linestyle=":")
        axes[1].plot(x, local, marker="o", linewidth=1.9, markersize=4.5, color=color, label=label)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.28)
        ax.set_xlabel(r"eccentricity $e_0$")
        ax.set_xlim(1.5e-5, 4.0e-1)
        ax.set_xticks([2.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 3.0e-1])
        ax.set_xticklabels([r"$0$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$0.3$"])

    axes[0].set_ylabel(r"mismatch $1-\mathrm{FF}$")
    axes[0].set_ylim(5.0e-3, 1.0)
    axes[1].set_ylim(5.0e-8, 1.0)
    axes[0].text(
        0.04,
        0.94,
        r"local TD $h(e_0)$ vs. circular JAXGBaccurate",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8.5,
    )
    axes[1].text(
        0.04,
        0.94,
        r"local TD $h(e_0)$ vs. local TD $h(0)$",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8.5,
    )
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")

    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)
    FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT, PAPER_OUTPUT)
    print(f"wrote {OUTPUT}")
    print(f"wrote {PAPER_OUTPUT}")


if __name__ == "__main__":
    main()
