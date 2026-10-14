"""Plot circular fdot absorption for the auto-PM wide-grid run."""

from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CSV_NAME = "production_eccentric_circular_fdot_tdi2_signal_none_auto_pm1em02_template_wide_grid_production.csv"
FIGURE_NAME = "auto_pm_wide_fdot_absorption_summary.png"
PAPER_FIGURE_DIR = Path(__file__).resolve().parents[1] / "reports" / "Effect_of_eccentricity_for_GB_in_LISA-new" / "figures"
PAPER_FIGURE_NAME = "paper_fdot_absorption_summary.png"

FDOT_FACTORS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
DISPLAYED_ECCENTRICITIES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
COLORS = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#999999", "#6A3D9A", "#111111"]


def load_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open()))


def best_fdot_factor(row: dict[str, str]) -> float:
    gr_fdot = float(row["circular_gr_fdot_hz_s"])
    best_fdot = float(row["best_circular_fdot_hz_s"])
    if gr_fdot == 0.0:
        return 0.0
    factor = best_fdot / gr_fdot
    return min(FDOT_FACTORS, key=lambda candidate: abs(candidate - factor))


def fdot_label(factor: float) -> str:
    if factor == 0.0:
        return "$0$"
    if factor == 0.25:
        return "$0.25$"
    if factor == 0.5:
        return "$0.5$"
    return f"${factor:g}$"


def plot_fdot_absorption(rows: list[dict[str, str]], path: Path) -> dict[float, Counter[float]]:
    counts_by_eccentricity: dict[float, Counter[float]] = {}
    for eccentricity in DISPLAYED_ECCENTRICITIES:
        subset = [row for row in rows if float(row["eccentricity"]) == eccentricity]
        counts_by_eccentricity[eccentricity] = Counter(best_fdot_factor(row) for row in subset)

    fig, ax = plt.subplots(figsize=(7.6, 4.9), constrained_layout=True)
    y = np.arange(len(DISPLAYED_ECCENTRICITIES), dtype=np.float64)
    left = np.zeros(len(DISPLAYED_ECCENTRICITIES), dtype=np.float64)

    for factor, color in zip(FDOT_FACTORS, COLORS, strict=True):
        values = np.array([counts_by_eccentricity[eccentricity].get(factor, 0) for eccentricity in DISPLAYED_ECCENTRICITIES])
        ax.barh(
            y,
            values,
            left=left,
            height=0.68,
            color=color,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.45,
            label=fdot_label(factor),
        )
        for row_index, value in enumerate(values):
            if value >= 14:
                ax.text(
                    left[row_index] + 0.5 * value,
                    y[row_index],
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if factor in (16.0, 32.0) else "black",
                )
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels([f"${eccentricity:g}$" for eccentricity in DISPLAYED_ECCENTRICITIES])
    ax.set_xlabel("number of production-grid cases")
    ax.set_ylabel("eccentricity $e_0$")
    ax.set_xlim(0.0, max(left) * 1.02)
    ax.grid(True, which="major", axis="x", alpha=0.28)
    ax.legend(
        title="$\\dot f_{\\rm circ}/\\dot f_{\\rm GR}$",
        fontsize=7.6,
        title_fontsize=8.0,
        ncol=5,
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
    )

    fig.savefig(path, dpi=220)
    plt.close(fig)
    return counts_by_eccentricity


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = load_rows(root / CSV_NAME)
    benchmark_path = root / FIGURE_NAME
    manuscript_path = PAPER_FIGURE_DIR / FIGURE_NAME
    manuscript_current_path = PAPER_FIGURE_DIR / PAPER_FIGURE_NAME

    counts = plot_fdot_absorption(rows, benchmark_path)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(benchmark_path, manuscript_path)
    shutil.copy2(benchmark_path, manuscript_current_path)

    print(f"wrote {benchmark_path}")
    print(f"wrote {manuscript_path}")
    print(f"wrote {manuscript_current_path}")
    for eccentricity in DISPLAYED_ECCENTRICITIES:
        formatted = ", ".join(f"{factor:g}:{counts[eccentricity].get(factor, 0)}" for factor in FDOT_FACTORS)
        print(f"e={eccentricity:g}: {formatted}")


if __name__ == "__main__":
    main()
