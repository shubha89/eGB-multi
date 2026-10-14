"""JAXGBaccurate vs eGB-multi raw mismatch and timing comparison."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import LEGEND_SIZE, TITLE_SIZE, apply_tick_style, use_paper_plot_style

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
PAPER_FIGURE_DIRS = (
    ROOT / "reports" / "Effect_of_eccentricity_for_GB_in_LISA-new" / "figures",
    ROOT / "reports" / "Effect_of_eccentricity_for_GB_in_LISA-final" / "figures",
)
INPUT = BENCHMARKS / "jaxgbaccurate_calibrated_eccentric_study_1pn_periastron.csv"
OUTPUT = BENCHMARKS / "paper_jaxgb_comprehensive_comparison.png"


COLORS = {
    1.0e-4: "#0072B2",
    3.0e-4: "#009E73",
    1.0e-3: "#D55E00",
    3.0e-3: "#CC79A7",
}


def scientific_hz_label(f0_hz: float) -> str:
    exponent = int(np.floor(np.log10(f0_hz)))
    mantissa = f0_hz / 10.0**exponent
    if np.isclose(mantissa, 1.0):
        return rf"$10^{{{exponent}}}\,\mathrm{{Hz}}$"
    return rf"${mantissa:g}\times10^{{{exponent}}}\,\mathrm{{Hz}}$"


def x_position(eccentricity: float) -> float:
    return eccentricity


def main() -> None:
    use_paper_plot_style()
    rows = list(csv.DictReader(INPUT.open()))
    frequencies = sorted({float(row["f0_hz"]) for row in rows})

    fig, (ax_raw, ax_time) = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.15),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    for f0 in frequencies:
        subset = sorted(
            [row for row in rows if np.isclose(float(row["f0_hz"]), f0)],
            key=lambda row: float(row["eccentricity"]),
        )
        x = np.array([x_position(float(row["eccentricity"])) for row in subset])
        raw = np.array([max(float(row["mismatch_AE"]), 1.0e-12) for row in subset])
        color = COLORS.get(f0, None)
        label = scientific_hz_label(f0)

        ax_raw.plot(x, raw, marker="o", linewidth=1.8, markersize=4.2, color=color, label=label)
        ax_raw.axhline(raw[0], color=color, linewidth=0.9, alpha=0.35, linestyle=":")

    ax_raw.set_xscale("symlog", linthresh=1.0e-4, linscale=0.5)
    ax_raw.set_yscale("log")
    ax_raw.grid(True, which="both", alpha=0.28)
    ax_raw.set_xlabel(r"eccentricity $e_0$")
    ax_raw.set_xlim(-2.0e-5, 4.0e-1)
    ax_raw.set_xticks([0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1])
    ax_raw.set_xticklabels([r"$0$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
    ax_raw.set_ylabel(r"mismatch $1-\mathrm{FF}$")
    ax_raw.set_ylim(5.0e-3, 1.0)
    ax_raw.set_title(r"raw response mismatch", fontsize=TITLE_SIZE)
    ax_raw.legend(fontsize=LEGEND_SIZE, frameon=True, loc="upper left")

    e0_rows = sorted([row for row in rows if np.isclose(float(row["eccentricity"]), 0.0)], key=lambda row: float(row["f0_hz"]))
    f0 = np.array([float(row["f0_hz"]) for row in e0_rows])
    jaxgb_time = np.array([float(row["jaxgbaccurate_xyz_s"]) for row in e0_rows])
    egb_min = []
    egb_med = []
    egb_max = []
    for frequency in f0:
        subset = [row for row in rows if np.isclose(float(row["f0_hz"]), frequency)]
        values = np.array([float(row["bruteforce_xyz_s"]) for row in subset])
        egb_min.append(float(np.min(values)))
        egb_med.append(float(np.median(values)))
        egb_max.append(float(np.max(values)))
    egb_min = np.array(egb_min)
    egb_med = np.array(egb_med)
    egb_max = np.array(egb_max)

    ax_time.fill_between(f0, egb_min, egb_max, color="#D55E00", alpha=0.22, label="eGB-multi TD range")
    ax_time.plot(f0, egb_med, marker="s", color="#D55E00", linewidth=1.8, label="eGB-multi TD median")
    ax_time.plot(f0, jaxgb_time, marker="o", color="#0072B2", linewidth=1.8, label="JAXGBaccurate")
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel(r"frequency $f_0$ [Hz]")
    ax_time.set_ylabel("wall time [s]")
    ax_time.set_title("response wall time", fontsize=TITLE_SIZE)
    ax_time.grid(True, which="both", alpha=0.28)
    ax_time.set_ylim(3.0e-3, 3.0e-2)
    ax_time.legend(fontsize=LEGEND_SIZE, frameon=True, loc="lower right")
    apply_tick_style((ax_raw, ax_time))

    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)
    print(f"wrote {OUTPUT}")
    for figure_dir in PAPER_FIGURE_DIRS:
        figure_dir.mkdir(parents=True, exist_ok=True)
        paper_output = figure_dir / OUTPUT.name
        shutil.copy2(OUTPUT, paper_output)
        print(f"wrote {paper_output}")


if __name__ == "__main__":
    main()
