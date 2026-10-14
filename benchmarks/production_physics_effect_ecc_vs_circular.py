"""Circular-vs-eccentric mismatch for optional source-physics switches.

This source-level diagnostic asks whether the optional eccentric-binary source
effects change the core circular-vs-eccentric conclusion.  For each switch we
compare an eccentric signal to the corresponding circular waveform at the same
frequency and masses. A fixed 1PN-no-periastron eccentric model is included as
a dashed baseline in every panel.
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain
from production_subdominant_corrections_summary import SECONDS_PER_YEAR, load_a_channel_psd, mismatch


PAPER_FIGURE_DIR = Path("/Users/shubhanshutiwari/Downloads/Effect_of_eccentricity_for_GB_in_LISA-2/figures")

CASES = {
    "periastron": {
        "label": "1PN periastron advance",
        "physics_mode": "1pn",
        "evolution_mode": "fixed",
    },
    "pm_orbital": {
        "label": "Peters-Mathews orbital evolution",
        "physics_mode": "1pn_no_periastron",
        "evolution_mode": "peters_mathews_orbital_only",
    },
    "pm_eccentricity": {
        "label": "Peters-Mathews eccentricity evolution",
        "physics_mode": "1pn_no_periastron",
        "evolution_mode": "peters_mathews_eccentricity_only",
    },
}


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


def strain_for(
    f0_hz: float,
    eccentricity: float,
    m1_solar: float,
    m2_solar: float,
    times: np.ndarray,
    *,
    physics_mode: str,
    evolution_mode: str,
) -> np.ndarray:
    source = make_source(f0_hz, eccentricity, m1_solar, m2_solar)
    return eccentric_complex_strain(source, times, physics_mode=physics_mode, evolution_mode=evolution_mode)


def run_grid() -> list[dict[str, object]]:
    out_dir = Path(__file__).resolve().parent
    psd_freq, psd_a = load_a_channel_psd(out_dir / "lisainstrument_welch_psd.csv")

    duration = 4.0 * SECONDS_PER_YEAR
    dt = 50.0
    times = np.arange(0.0, duration, dt, dtype=np.float64)
    freq = np.fft.rfftfreq(times.size, dt)
    psd = np.interp(freq, psd_freq, psd_a, left=np.inf, right=np.inf)

    frequencies = [2.0e-4, 1.0e-3, 3.0e-3]
    eccentricities = [0.0, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 5.0e-2, 0.1, 0.2, 0.3]
    masses = [(0.5, 0.5), (0.8, 0.6), (1.4, 1.4)]

    rows: list[dict[str, object]] = []
    total = len(frequencies) * len(eccentricities) * len(masses)
    case_index = 0
    for f0_hz in frequencies:
        band_mask = (freq >= 0.20 * f0_hz) & (freq <= min(freq[-1], 10.0 * f0_hz))
        for eccentricity in eccentricities:
            for m1_solar, m2_solar in masses:
                case_index += 1
                started = time.perf_counter()
                circular_baseline = strain_for(
                    f0_hz,
                    0.0,
                    m1_solar,
                    m2_solar,
                    times,
                    physics_mode="1pn_no_periastron",
                    evolution_mode="fixed",
                )
                eccentric_baseline = strain_for(
                    f0_hz,
                    eccentricity,
                    m1_solar,
                    m2_solar,
                    times,
                    physics_mode="1pn_no_periastron",
                    evolution_mode="fixed",
                )
                rows.append(
                    {
                        "case": "fixed_no_periastron",
                        "duration_years": 4.0,
                        "dt_s": dt,
                        "samples": times.size,
                        "f0_hz": f0_hz,
                        "eccentricity": eccentricity,
                        "m1_solar": m1_solar,
                        "m2_solar": m2_solar,
                        "mismatch": mismatch(eccentric_baseline, circular_baseline, psd, dt, band_mask.copy()),
                    }
                )

                for case_name, config in CASES.items():
                    circular = strain_for(
                        f0_hz,
                        0.0,
                        m1_solar,
                        m2_solar,
                        times,
                        physics_mode=str(config["physics_mode"]),
                        evolution_mode=str(config["evolution_mode"]),
                    )
                    eccentric = strain_for(
                        f0_hz,
                        eccentricity,
                        m1_solar,
                        m2_solar,
                        times,
                        physics_mode=str(config["physics_mode"]),
                        evolution_mode=str(config["evolution_mode"]),
                    )
                    rows.append(
                        {
                            "case": case_name,
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


def plot_case(rows: list[dict[str, object]], case_name: str, path: Path) -> None:
    label = CASES[case_name]["label"]
    frequencies = sorted({float(row["f0_hz"]) for row in rows})
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})
    colors = ["#0072B2", "#D55E00", "#009E73"]

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for color, f0_hz in zip(colors, frequencies, strict=True):
        for plotted_case, linestyle, alpha, plotted_label in [
            ("fixed_no_periastron", "--", 0.55, "fixed no-periastron baseline"),
            (case_name, "-", 1.0, label),
        ]:
            medians = []
            lows = []
            highs = []
            for eccentricity in eccentricities:
                values = np.array(
                    [
                        max(float(row["mismatch"]), 1.0e-16)
                        for row in rows
                        if row["case"] == plotted_case
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
                marker="o" if linestyle == "-" else None,
                linewidth=2.0 if linestyle == "-" else 1.4,
                linestyle=linestyle,
                color=color,
                alpha=alpha,
            )
            if linestyle == "-":
                ax.fill_between(eccentricities, lows, highs, color=color, alpha=0.12, linewidth=0)

    ax.set_xscale("symlog", linthresh=1.0e-3)
    ax.set_yscale("log")
    ax.set_xlim(0.0, max(eccentricities) * 1.05)
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("source-level eccentric-vs-circular mismatch, $1-\\mathrm{FF}$")
    ax.set_title(label)
    ax.grid(True, which="both", alpha=0.3)
    frequency_handles = [
        Line2D([0], [0], color=color, linewidth=2.0, marker="o", label=f"$f_0={f0_hz:.0e}\\,\\mathrm{{Hz}}$")
        for color, f0_hz in zip(colors, frequencies, strict=True)
    ]
    style_handles = [
        Line2D([0], [0], color="0.25", linewidth=2.0, linestyle="-", label="with physics switch"),
        Line2D([0], [0], color="0.25", linewidth=1.6, linestyle="--", alpha=0.7, label="fixed no-periastron baseline"),
    ]
    first_legend = ax.legend(handles=frequency_handles, fontsize=8, loc="upper left", frameon=True)
    ax.add_artist(first_legend)
    ax.legend(handles=style_handles, fontsize=8, loc="lower right", frameon=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize(rows: list[dict[str, object]]) -> None:
    for case in ["fixed_no_periastron", *CASES.keys()]:
        values_e01 = [float(row["mismatch"]) for row in rows if row["case"] == case and abs(float(row["eccentricity"]) - 0.1) < 1e-15]
        values_e03 = [float(row["mismatch"]) for row in rows if row["case"] == case and abs(float(row["eccentricity"]) - 0.3) < 1e-15]
        print(case, f"median_e01={np.median(values_e01):.3e}", f"median_e03={np.median(values_e03):.3e}")


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows = run_grid()
    csv_path = out_dir / "production_physics_effect_ecc_vs_circular.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for case_name in CASES:
        png_path = out_dir / f"production_ecc_vs_circular_{case_name}.png"
        plot_case(rows, case_name, png_path)
        shutil.copy2(png_path, PAPER_FIGURE_DIR / png_path.name)
        print(f"wrote {png_path}")
        print(f"copied {PAPER_FIGURE_DIR / png_path.name}")
    summarize(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
