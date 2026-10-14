"""Noise-weighted source-level mismatch summary for source-physics switches.

The fixed Newtonian eccentric source is used as the common lower-physics
baseline. The figure compares this baseline to a fixed 1PN eccentric source,
a Newtonian source with Peters-Mathews evolution, and a 1PN source with full
Peters-Mathews evolution.
This is intentionally source-level, not full-response, so that it isolates
source-physics effects before the LISA response is applied.

The mismatch uses the measured lisainstrument A-channel PSD as a proxy
weighting curve for source-level comparisons.
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from paper_plot_style import ANNOTATION_SIZE, LEGEND_SIZE, apply_tick_style, use_paper_plot_style

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
PAPER_FIGURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "Effect_of_eccentricity_for_GB_in_LISA-new"
    / "figures"
)


def scientific_hz_tick(f0_hz: float) -> str:
    exponent = int(np.floor(np.log10(f0_hz)))
    mantissa = f0_hz / 10.0**exponent
    if np.isclose(mantissa, 1.0):
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa:g}\times10^{{{exponent}}}$"


def load_a_channel_psd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    return (
        np.array([float(row["frequency_hz"]) for row in rows], dtype=np.float64),
        np.array([float(row["PSD_A"]) for row in rows], dtype=np.float64),
    )


def rfft_real(series: np.ndarray, dt: float) -> np.ndarray:
    return np.fft.rfft(np.asarray(series, dtype=np.float64)) * dt


def complex_inner(left: np.ndarray, right: np.ndarray, psd: np.ndarray, df: float, mask: np.ndarray) -> complex:
    valid = mask & np.isfinite(psd) & (psd > 0.0)
    if valid.size:
        valid[0] = False
    return complex(4.0 * np.sum(left[valid] * np.conjugate(right[valid]) / psd[valid]) * df)


def fd_polarizations(strain: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    return rfft_real(np.real(strain), dt), rfft_real(-np.imag(strain), dt)


def two_pol_inner(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
    psd: np.ndarray,
    df: float,
    mask: np.ndarray,
) -> complex:
    return complex_inner(left[0], right[0], psd, df, mask.copy()) + complex_inner(left[1], right[1], psd, df, mask.copy())


def mismatch(reference: np.ndarray, template: np.ndarray, psd: np.ndarray, dt: float, mask: np.ndarray) -> float:
    df = 1.0 / (reference.size * dt)
    ref_fd = fd_polarizations(reference, dt)
    tmp_fd = fd_polarizations(template, dt)
    rr = two_pol_inner(ref_fd, ref_fd, psd, df, mask).real
    tt = two_pol_inner(tmp_fd, tmp_fd, psd, df, mask).real
    rt = two_pol_inner(ref_fd, tmp_fd, psd, df, mask)
    ff = float(abs(rt) / np.sqrt(rr * tt)) if rr > 0.0 and tt > 0.0 else 0.0
    return max(1.0 - min(ff, 1.0), 0.0)


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


def run_grid() -> list[dict[str, object]]:
    out_dir = Path(__file__).resolve().parent
    psd_freq, psd_a = load_a_channel_psd(out_dir / "lisainstrument_welch_psd.csv")

    duration = 4.0 * SECONDS_PER_YEAR
    dt = 50.0
    times = np.arange(0.0, duration, dt, dtype=np.float64)
    freq = np.fft.rfftfreq(times.size, dt)
    psd = np.interp(freq, psd_freq, psd_a, left=np.inf, right=np.inf)

    frequencies = [2.0e-4, 1.0e-3, 2.0e-3, 3.0e-3, 5.0e-3]
    eccentricities = [0.1, 0.3, 0.5]
    masses = [(0.5, 0.5), (0.8, 0.6), (1.4, 1.4)]
    comparisons = {
        "fixed_1pn_vs_fixed_newtonian": ("fixed_1pn", "fixed_newtonian"),
        "newtonian_pm_vs_fixed_newtonian": ("newtonian_pm", "fixed_newtonian"),
        "pm_full_vs_fixed_newtonian": ("full_pm", "fixed_newtonian"),
    }

    rows: list[dict[str, object]] = []
    total = len(frequencies) * len(eccentricities) * len(masses)
    case_index = 0
    for f0_hz in frequencies:
        band_mask = (freq >= 0.20 * f0_hz) & (freq <= min(freq[-1], 10.0 * f0_hz))
        for eccentricity in eccentricities:
            for m1_solar, m2_solar in masses:
                case_index += 1
                source = make_source(f0_hz, eccentricity, m1_solar, m2_solar)
                started = time.perf_counter()
                strains = {
                    "fixed_newtonian": eccentric_complex_strain(source, times, physics_mode="newtonian", evolution_mode="fixed"),
                    "fixed_1pn": eccentric_complex_strain(source, times, physics_mode="1pn", evolution_mode="fixed"),
                    "newtonian_pm": eccentric_complex_strain(
                        source,
                        times,
                        physics_mode="newtonian",
                        evolution_mode="peters_mathews",
                    ),
                    "full_pm": eccentric_complex_strain(source, times, physics_mode="1pn", evolution_mode="peters_mathews"),
                }
                wall_s = time.perf_counter() - started
                for comparison, (reference_label, template_label) in comparisons.items():
                    rows.append(
                        {
                            "duration_years": 4.0,
                            "dt_s": dt,
                            "samples": times.size,
                            "f0_hz": f0_hz,
                            "eccentricity": eccentricity,
                            "m1_solar": m1_solar,
                            "m2_solar": m2_solar,
                            "comparison": comparison,
                            "waveform_wall_s": wall_s,
                            "mismatch": mismatch(strains[reference_label], strains[template_label], psd, dt, band_mask.copy()),
                        }
                    )
                print(
                    f"[{case_index:02d}/{total}] f0={f0_hz:.1e} e={eccentricity:.1f} "
                    f"m={m1_solar:.1f}+{m2_solar:.1f} wall={wall_s:.2f}s",
                    flush=True,
                )
    return rows


def plot_rows(rows: list[dict[str, object]], path: Path, *, zoom_floor: float | None = None) -> None:
    use_paper_plot_style()
    comparison_order = [
        "fixed_1pn_vs_fixed_newtonian",
        "newtonian_pm_vs_fixed_newtonian",
        "pm_full_vs_fixed_newtonian",
    ]
    labels = {
        "fixed_1pn_vs_fixed_newtonian": "fixed 1PN vs fixed Newtonian",
        "newtonian_pm_vs_fixed_newtonian": "Newtonian+PM vs fixed Newtonian",
        "pm_full_vs_fixed_newtonian": "1PN+PM full vs fixed Newtonian",
    }
    colors = {
        "fixed_1pn_vs_fixed_newtonian": "#D55E00",
        "newtonian_pm_vs_fixed_newtonian": "#0072B2",
        "pm_full_vs_fixed_newtonian": "#009E73",
    }
    x_offsets = {
        "fixed_1pn_vs_fixed_newtonian": 0.90,
        "newtonian_pm_vs_fixed_newtonian": 1.00,
        "pm_full_vs_fixed_newtonian": 1.10,
    }
    linestyles = {
        "fixed_1pn_vs_fixed_newtonian": "-",
        "newtonian_pm_vs_fixed_newtonian": ":",
        "pm_full_vs_fixed_newtonian": "--",
    }
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})
    fig, axes = plt.subplots(1, len(eccentricities), figsize=(11.8, 4.6), constrained_layout=True, sharey=True)
    if len(eccentricities) == 1:
        axes = [axes]
    for ax, eccentricity in zip(axes, eccentricities, strict=True):
        subset_e = [row for row in rows if float(row["eccentricity"]) == eccentricity]
        frequencies = np.array(sorted({float(row["f0_hz"]) for row in subset_e}))
        for comparison in comparison_order:
            medians = []
            lows = []
            highs = []
            for f0_hz in frequencies:
                values = np.array(
                    [
                        max(float(row["mismatch"]), 1.0e-16)
                        for row in subset_e
                        if float(row["f0_hz"]) == f0_hz and row["comparison"] == comparison
                    ],
                    dtype=np.float64,
                )
                medians.append(np.median(values))
                lows.append(np.min(values))
                highs.append(np.max(values))
            medians = np.array(medians)
            lows = np.array(lows)
            highs = np.array(highs)
            x_values = frequencies * x_offsets[comparison]
            if zoom_floor is None:
                plot_medians = medians
                plot_lows = lows
                plot_highs = highs
            else:
                visible = medians >= zoom_floor
                plot_medians = np.where(visible, medians, np.nan)
                plot_lows = np.where(visible, np.maximum(lows, zoom_floor), np.nan)
                plot_highs = np.where(visible, highs, np.nan)
            ax.plot(
                x_values,
                plot_medians,
                marker="o",
                linewidth=2.0,
                linestyle=linestyles[comparison],
                color=colors[comparison],
                label=labels[comparison],
            )
            ax.fill_between(x_values, plot_lows, plot_highs, color=colors[comparison], alpha=0.12, linewidth=0)
            if zoom_floor is not None:
                for x_value, median in zip(x_values, medians, strict=True):
                    if median < zoom_floor:
                        ax.text(
                            x_value,
                            zoom_floor * 1.13,
                            r"$<10^{-4}$",
                            color=colors[comparison],
                            fontsize=ANNOTATION_SIZE - 1.0,
                            rotation=90,
                            ha="center",
                            va="bottom",
                        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(frequencies)
        ax.set_xticklabels([scientific_hz_tick(frequency) for frequency in frequencies], rotation=35, ha="right")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("$f_0$ [Hz]")
        ax.set_title(f"$e_0={eccentricity:g}$")
        ax.grid(True, which="both", alpha=0.25)
        if zoom_floor is not None:
            ax.set_ylim(zoom_floor, 2.0)
    axes[0].set_ylabel("source-level mismatch, $1-\\mathrm{FF}$")
    axes[-1].legend(fontsize=LEGEND_SIZE, frameon=True)
    apply_tick_style(axes)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize(rows: list[dict[str, object]]) -> None:
    for comparison in sorted({str(row["comparison"]) for row in rows}):
        values_low = [
            float(row["mismatch"])
            for row in rows
            if row["comparison"] == comparison and float(row["f0_hz"]) <= 1.0e-3
        ]
        values_all = [float(row["mismatch"]) for row in rows if row["comparison"] == comparison]
        print(
            comparison,
            f"median_low={np.median(values_low):.3e}",
            f"max_low={np.max(values_low):.3e}",
            f"median_all={np.median(values_all):.3e}",
            f"max_all={np.max(values_all):.3e}",
        )


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows = run_grid()
    csv_path = out_dir / "production_subdominant_corrections_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    png_path = out_dir / "production_subdominant_corrections_summary.png"
    plot_rows(rows, png_path, zoom_floor=1.0e-4)
    zoom_png_path = out_dir / "production_subdominant_corrections_summary_zoom_all.png"
    plot_rows(rows, zoom_png_path, zoom_floor=1.0e-4)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png_path, PAPER_FIGURE_DIR / png_path.name)
    shutil.copy2(png_path, PAPER_FIGURE_DIR / "source_corrections_summary.png")
    shutil.copy2(zoom_png_path, PAPER_FIGURE_DIR / zoom_png_path.name)
    shutil.copy2(zoom_png_path, PAPER_FIGURE_DIR / "source_corrections_summary_zoom_all.png")
    shutil.copy2(zoom_png_path, PAPER_FIGURE_DIR / "source_corrections_summary_zoom_pm.png")
    summarize(rows)
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"wrote {zoom_png_path}")
    print(f"copied {PAPER_FIGURE_DIR / png_path.name}")
    print(f"copied {PAPER_FIGURE_DIR / 'source_corrections_summary.png'}")
    print(f"copied {PAPER_FIGURE_DIR / zoom_png_path.name}")
    print(f"copied {PAPER_FIGURE_DIR / 'source_corrections_summary_zoom_all.png'}")
    print(f"copied {PAPER_FIGURE_DIR / 'source_corrections_summary_zoom_pm.png'}")


if __name__ == "__main__":
    main()
