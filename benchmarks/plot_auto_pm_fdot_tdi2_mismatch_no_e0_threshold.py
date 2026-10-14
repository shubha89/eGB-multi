"""Combined paper plot for auto-PM eccentric-vs-circular TDI mismatch."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import ANNOTATION_SIZE, LEGEND_SIZE, apply_tick_style, use_paper_plot_style


THRESHOLD = 0.1
CSV_NAME = "production_eccentric_circular_fdot_tdi2_signal_none_auto_pm1em02_template_wide_grid_production.csv"
FIGURE_NAME = "auto_pm_wide_fdot_tdi2_mismatch_vs_eccentricity_by_f0_no_e0_threshold.png"
MANUSCRIPT_FIGURE_NAME = "wide_fdot_tdi2_mismatch_vs_eccentricity_by_f0_no_e0_threshold.png"
PAPER_FIGURE_DIRS = (
    Path(__file__).resolve().parents[1] / "reports" / "Effect_of_eccentricity_for_GB_in_LISA-new" / "figures",
    Path(__file__).resolve().parents[1] / "reports" / "Effect_of_eccentricity_for_GB_in_LISA-final" / "figures",
)


def load_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open()))


def log_interpolated_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    above = y >= threshold
    if not np.any(above):
        return None
    first = int(np.argmax(above))
    if first == 0:
        return float(x[0])

    x0, x1 = x[first - 1], x[first]
    y0, y1 = y[first - 1], y[first]
    if y0 <= 0.0 or y1 <= 0.0:
        return float(x1)
    frac = (np.log(threshold) - np.log(y0)) / (np.log(y1) - np.log(y0))
    return float(np.exp(np.log(x0) + frac * (np.log(x1) - np.log(x0))))


def frequency_label(f0_hz: float) -> str:
    exponent = int(np.floor(np.log10(f0_hz)))
    mantissa = f0_hz / 10.0**exponent
    if np.isclose(mantissa, 1.0):
        return rf"$f_0=10^{{{exponent}}}\,\mathrm{{Hz}}$"
    return rf"$f_0={mantissa:g}\times10^{{{exponent}}}\,\mathrm{{Hz}}$"


def plot_mismatch_lines(rows: list[dict[str, str]], path: Path) -> list[tuple[float, float | None]]:
    use_paper_plot_style()
    rows = [row for row in rows if float(row["eccentricity"]) > 0.0]
    frequencies = sorted({float(row["f0_hz"]) for row in rows})
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    crossings: list[tuple[float, float | None]] = []
    ymax = THRESHOLD
    for color, f0_hz in zip(colors, frequencies, strict=True):
        medians = []
        lows = []
        highs = []
        for eccentricity in eccentricities:
            values = np.array(
                [
                    max(float(row["mismatch_AE"]), 1.0e-16)
                    for row in rows
                    if float(row["f0_hz"]) == f0_hz and float(row["eccentricity"]) == eccentricity
                ],
                dtype=np.float64,
            )
            medians.append(np.median(values))
            lows.append(np.percentile(values, 16.0))
            highs.append(np.percentile(values, 84.0))

        x = np.array(eccentricities)
        y = np.array(medians)
        lo = np.array(lows)
        hi = np.array(highs)
        ymax = max(ymax, float(np.max(hi)) * 1.25)
        crossing = log_interpolated_crossing(x, y, THRESHOLD)
        crossings.append((f0_hz, crossing))

        ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0.0)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4.4,
            color=color,
            label=frequency_label(f0_hz),
        )

    ax.axhline(THRESHOLD, color="0.2", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.text(
        0.00108,
        THRESHOLD * 1.08,
        "$1-\\mathrm{FF}=0.1$",
        ha="left",
        va="bottom",
        color="0.2",
        fontsize=ANNOTATION_SIZE,
    )

    axis_ymin = 8.0e-7
    label_y = 1.8e-5
    for index, (color, (f0_hz, crossing)) in enumerate(zip(colors, crossings, strict=True)):
        if crossing is None:
            continue
        ax.vlines(
            crossing,
            axis_ymin,
            THRESHOLD,
            color=color,
            linestyle=":",
            linewidth=1.35,
            alpha=0.72,
        )
        ax.text(
            crossing,
            label_y,
            f"{crossing:.2g}",
            rotation=90,
            ha="center",
            va="center",
            color=color,
            fontsize=ANNOTATION_SIZE,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(eccentricities) * 0.82, max(eccentricities) * 1.18)
    ax.set_ylim(axis_ymin, ymax)
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("A/E mismatch, $1-\\mathrm{FF}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE, ncol=2, frameon=True, loc="upper left")
    apply_tick_style(ax)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return crossings


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = load_rows(root / CSV_NAME)
    benchmark_path = root / FIGURE_NAME
    benchmark_manuscript_path = root / MANUSCRIPT_FIGURE_NAME

    crossings = plot_mismatch_lines(rows, benchmark_path)
    shutil.copy2(benchmark_path, benchmark_manuscript_path)
    for figure_dir in PAPER_FIGURE_DIRS:
        figure_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(benchmark_path, figure_dir / FIGURE_NAME)
        shutil.copy2(benchmark_path, figure_dir / MANUSCRIPT_FIGURE_NAME)

    print(f"wrote {benchmark_path}")
    print(f"wrote {benchmark_manuscript_path}")
    for figure_dir in PAPER_FIGURE_DIRS:
        print(f"wrote {figure_dir / FIGURE_NAME}")
        print(f"wrote {figure_dir / MANUSCRIPT_FIGURE_NAME}")
    for f0_hz, crossing in crossings:
        if crossing is None:
            print(f"f0={f0_hz:.6g} Hz: mismatch does not reach {THRESHOLD}")
        else:
            print(f"f0={f0_hz:.6g} Hz: mismatch={THRESHOLD} at e≈{crossing:.6g}")


if __name__ == "__main__":
    main()
