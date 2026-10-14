"""Mismatch-vs-eccentricity plot without e=0, marking the 0.1 mismatch crossings."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THRESHOLD = 0.1


def load_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open()))


def log_interpolated_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    """Return the first x where y crosses threshold, interpolating in log-log space."""

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
    logx0, logx1 = np.log(x0), np.log(x1)
    logy0, logy1 = np.log(y0), np.log(y1)
    frac = (np.log(threshold) - logy0) / (logy1 - logy0)
    return float(np.exp(logx0 + frac * (logx1 - logx0)))


def plot_mismatch_lines_no_e0_threshold(rows: list[dict[str, str]], path: Path) -> list[tuple[float, float | None]]:
    rows = [row for row in rows if float(row["eccentricity"]) > 0.0]
    frequencies = sorted({float(row["f0_hz"]) for row in rows})
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    column = "mismatch_AE"

    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    crossings: list[tuple[float, float | None]] = []
    ymax = 2.0
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
        y = np.array(medians)
        ymax = max(ymax, float(np.max(y)) * 1.35)
        crossing = log_interpolated_crossing(x, y, THRESHOLD)
        crossings.append((f0_hz, crossing))

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4.4,
            color=color,
            label=f"$f_0={f0_hz:.0e}\\,\\mathrm{{Hz}}$",
        )
        if crossing is not None:
            ax.vlines(
                crossing,
                THRESHOLD,
                ymax,
                color=color,
                linestyle=":",
                linewidth=1.45,
                alpha=0.78,
            )
            ax.text(
                crossing,
                ymax / 1.08,
                f"{crossing:.2g}",
                rotation=90,
                ha="right",
                va="top",
                color=color,
                fontsize=7.4,
            )

    ax.axhline(THRESHOLD, color="0.2", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.text(
        0.00105,
        THRESHOLD * 1.08,
        "$1-\\mathrm{FF}=0.1$",
        ha="left",
        va="bottom",
        color="0.2",
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(6.0e-7, ymax)
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("A/E mismatch, $1-\\mathrm{FF}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, frameon=True, loc="upper left")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return crossings


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = load_rows(root / "production_eccentric_circular_fdot_tdi2_signal_gr_template_grid_production.csv")
    benchmark_path = root / "wide_fdot_tdi2_mismatch_vs_eccentricity_by_f0_no_e0_threshold.png"
    manuscript_path = (
        Path("/Users/shubhanshutiwari/Downloads/Effect_of_eccentricity_for_GB_in_LISA-2/figures")
        / "wide_fdot_tdi2_mismatch_vs_eccentricity_by_f0_no_e0_threshold.png"
    )
    crossings = plot_mismatch_lines_no_e0_threshold(rows, benchmark_path)
    plot_mismatch_lines_no_e0_threshold(rows, manuscript_path)
    print(f"wrote {benchmark_path}")
    print(f"wrote {manuscript_path}")
    for f0_hz, crossing in crossings:
        if crossing is None:
            print(f"f0={f0_hz:.6g} Hz: mismatch does not reach {THRESHOLD}")
        else:
            print(f"f0={f0_hz:.6g} Hz: mismatch={THRESHOLD} at e≈{crossing:.6g}")


if __name__ == "__main__":
    main()
