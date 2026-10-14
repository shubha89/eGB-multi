"""Circular-vs-eccentric mismatch for fixed 1PN vs full Peters-Mathews evolution."""

from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paper_plot_style import LEGEND_SIZE, apply_tick_style, use_paper_plot_style

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain
from production_subdominant_corrections_summary import SECONDS_PER_YEAR, load_a_channel_psd, mismatch


PAPER_FIGURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "Effect_of_eccentricity_for_GB_in_LISA-new"
    / "figures"
)


def scientific_hz_label(f0_hz: float) -> str:
    exponent = int(np.floor(np.log10(f0_hz)))
    mantissa = f0_hz / 10.0**exponent
    if np.isclose(mantissa, 1.0):
        return rf"$f_0=10^{{{exponent}}}\,\mathrm{{Hz}}$"
    return rf"$f_0={mantissa:g}\times10^{{{exponent}}}\,\mathrm{{Hz}}$"


def make_source(f0_hz: float, eccentricity: float, m1_solar: float, m2_solar: float) -> EccentricBinaryParams:
    return EccentricBinaryParams(
        mean_motion=np.pi * f0_hz,
        eccentricity=eccentricity,
        m1_solar=m1_solar,
        m2_solar=m2_solar,
        beta=0.3,
        lambda_=-0.7,
        psi=0.4,
        inclination=0.8,
        phi0=0.2,
    )


def strain(
    f0_hz: float,
    eccentricity: float,
    m1_solar: float,
    m2_solar: float,
    times: np.ndarray,
    *,
    model: str,
) -> np.ndarray:
    source = make_source(f0_hz, eccentricity, m1_solar, m2_solar)
    if model == "fixed_1pn":
        return eccentric_complex_strain(
            source,
            times,
            physics_mode="1pn",
            evolution_mode="fixed",
        )
    if model == "full_pm":
        return eccentric_complex_strain(
            source,
            times,
            physics_mode="1pn",
            evolution_mode="peters_mathews",
        )
    raise ValueError(f"unknown model {model!r}")


def run_grid() -> list[dict[str, object]]:
    out_dir = Path(__file__).resolve().parent
    psd_freq, psd_a = load_a_channel_psd(out_dir / "lisainstrument_welch_psd.csv")

    duration = 4.0 * SECONDS_PER_YEAR
    dt = 50.0
    times = np.arange(0.0, duration, dt, dtype=np.float64)
    freq = np.fft.rfftfreq(times.size, dt)
    psd = np.interp(freq, psd_freq, psd_a, left=np.inf, right=np.inf)

    frequencies = [2.0e-4, 1.0e-3, 2.0e-3, 3.0e-3, 5.0e-3]
    eccentricities = [0.0, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 5.0e-2, 0.1, 0.2, 0.3]
    masses = [(0.5, 0.5), (0.8, 0.6), (1.4, 1.4)]
    models = ["fixed_1pn", "full_pm"]

    rows: list[dict[str, object]] = []
    total = len(frequencies) * len(eccentricities) * len(masses)
    case_index = 0
    for f0_hz in frequencies:
        band_mask = (freq >= 0.20 * f0_hz) & (freq <= min(freq[-1], 10.0 * f0_hz))
        for eccentricity in eccentricities:
            for m1_solar, m2_solar in masses:
                case_index += 1
                started = time.perf_counter()
                for model in models:
                    circular = strain(f0_hz, 0.0, m1_solar, m2_solar, times, model=model)
                    eccentric = strain(f0_hz, eccentricity, m1_solar, m2_solar, times, model=model)
                    rows.append(
                        {
                            "model": model,
                            "duration_years": 4.0,
                            "dt_s": dt,
                            "samples": times.size,
                            "f0_hz": f0_hz,
                            "eccentricity": eccentricity,
                            "m1_solar": m1_solar,
                            "m2_solar": m2_solar,
                            "mismatch": mismatch(eccentric, circular, psd, dt, band_mask.copy()),
                        }
                    )
                print(
                    f"[{case_index:02d}/{total}] f0={f0_hz:.1e} e={eccentricity:.3g} "
                    f"m={m1_solar:.1f}+{m2_solar:.1f} wall={time.perf_counter() - started:.2f}s",
                    flush=True,
                )
    return rows


def plot_rows(rows: list[dict[str, object]], path: Path) -> None:
    use_paper_plot_style()
    frequencies = sorted({float(row["f0_hz"]) for row in rows})
    eccentricities = sorted({float(row["eccentricity"]) for row in rows if float(row["eccentricity"]) > 0.0})
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    labels = {
        "fixed_1pn": "fixed 1PN",
        "full_pm": "1PN + full PM",
    }
    styles = {
        "fixed_1pn": ("--", 0.65, None),
        "full_pm": ("-", 1.0, "o"),
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for color, f0_hz in zip(colors, frequencies, strict=True):
        for model in ["fixed_1pn", "full_pm"]:
            linestyle, alpha, marker = styles[model]
            medians = []
            lows = []
            highs = []
            for eccentricity in eccentricities:
                values = np.array(
                    [
                        max(float(row["mismatch"]), 1.0e-16)
                        for row in rows
                        if row["model"] == model
                        and float(row["f0_hz"]) == f0_hz
                        and float(row["eccentricity"]) == eccentricity
                    ],
                    dtype=np.float64,
                )
                medians.append(np.median(values))
                lows.append(np.min(values))
                highs.append(np.max(values))
            ax.plot(
                eccentricities,
                medians,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0 if model == "full_current" else 1.5,
                alpha=alpha,
            )
            if model == "full_pm":
                ax.fill_between(eccentricities, lows, highs, color=color, alpha=0.12, linewidth=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(eccentricities) * 0.8, max(eccentricities) * 1.05)
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("source-level eccentric-vs-circular mismatch, $1-\\mathrm{FF}$")
    ax.grid(True, which="both", alpha=0.3)

    frequency_handles = [
        Line2D([0], [0], color=color, linewidth=2.0, marker="o", label=scientific_hz_label(f0_hz))
        for color, f0_hz in zip(colors, frequencies, strict=True)
    ]
    model_handles = [
        Line2D([0], [0], color="0.25", linewidth=2.0, linestyle=styles[model][0], marker=styles[model][2], label=labels[model])
        for model in ["fixed_1pn", "full_pm"]
    ]
    first_legend = ax.legend(handles=frequency_handles, fontsize=LEGEND_SIZE, loc="upper left", frameon=True)
    ax.add_artist(first_legend)
    ax.legend(handles=model_handles, fontsize=LEGEND_SIZE, loc="lower right", frameon=True)
    apply_tick_style(ax)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize(rows: list[dict[str, object]]) -> None:
    for model in ["fixed_1pn", "full_pm"]:
        for eccentricity in [0.1, 0.3]:
            values = [
                float(row["mismatch"])
                for row in rows
                if row["model"] == model and abs(float(row["eccentricity"]) - eccentricity) < 1.0e-15
            ]
            print(model, f"e={eccentricity:g}", f"median={np.median(values):.3e}", f"max={np.max(values):.3e}")


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows = run_grid()
    csv_path = out_dir / "production_full_model_vs_fixed_circular.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    png_path = out_dir / "production_full_model_vs_fixed_circular.png"
    plot_rows(rows, png_path)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png_path, PAPER_FIGURE_DIR / png_path.name)
    shutil.copy2(png_path, PAPER_FIGURE_DIR / "full_model_vs_fixed_circular.png")
    summarize(rows)
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"copied {PAPER_FIGURE_DIR / png_path.name}")
    print(f"copied {PAPER_FIGURE_DIR / 'full_model_vs_fixed_circular.png'}")


if __name__ == "__main__":
    main()
