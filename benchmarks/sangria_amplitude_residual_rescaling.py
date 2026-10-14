"""Use Sangria GB amplitudes to rescale the paper residual diagnostic.

The production residual grid was generated for physical binaries at 1 kpc.
This script uses the Sangria training catalog amplitudes to estimate how those
noise-weighted residuals would scale for population-like Galactic-binary
amplitudes in the same central-frequency bins.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.constants import C_M_PER_S, G_M3_KG_S2, KILOPARSEC_M, SOLAR_MASS_KG


SANGRIA_PATH = Path("/Users/shubhanshutiwari/Downloads/LDC2_sangria_training_v2.h5")
PRODUCTION_CSV = Path(__file__).resolve().parent / "production_eccentric_circular_fdot_tdi2_signal_gr_template_grid_production.csv"
OUT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_FIG_DIR = Path("/Users/shubhanshutiwari/Downloads/Effect_of_eccentricity_for_GB_in_LISA-2/figures")

F0_GRID = np.array([1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 3.0e-3])
AMPLITUDE_PERCENTILES = np.array([1.0, 5.0, 16.0, 50.0, 84.0, 90.0, 95.0, 99.0])
PLOT_PERCENTILES = (50.0, 90.0, 99.0)
CHUNK_SIZE = 1_000_000


def frequency_edges(centers: np.ndarray) -> np.ndarray:
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    edges[0] = centers[0] ** 2 / edges[1]
    edges[-1] = centers[-1] ** 2 / edges[-2]
    return edges


def circular_amplitude_at_1kpc(f0_hz: float, chirp_mass_solar: float) -> float:
    chirp_mass_kg = chirp_mass_solar * SOLAR_MASS_KG
    return float(
        2.0
        * (G_M3_KG_S2 * chirp_mass_kg) ** (5.0 / 3.0)
        * (np.pi * f0_hz) ** (2.0 / 3.0)
        / (C_M_PER_S**4 * KILOPARSEC_M)
    )


def read_catalog_bin_amplitudes(h5_path: Path, f0_grid: np.ndarray) -> dict[float, np.ndarray]:
    edges = frequency_edges(f0_grid)
    chunks: dict[float, list[np.ndarray]] = {float(f0): [] for f0 in f0_grid}
    with h5py.File(h5_path, "r") as h5:
        for dataset_name in ("sky/dgb/cat", "sky/igb/cat"):
            dset = h5[dataset_name]
            nrows = dset.shape[0]
            for start in range(0, nrows, CHUNK_SIZE):
                stop = min(start + CHUNK_SIZE, nrows)
                block = dset[start:stop, 0]
                freq = block["Frequency"]
                amp = block["Amplitude"]
                valid = np.isfinite(freq) & np.isfinite(amp) & (freq > 0.0) & (amp > 0.0)
                if not np.any(valid):
                    continue
                freq = freq[valid]
                amp = amp[valid]
                which = np.searchsorted(edges, freq, side="right") - 1
                in_range = (which >= 0) & (which < f0_grid.size)
                if not np.any(in_range):
                    continue
                which = which[in_range]
                amp = amp[in_range]
                for index, f0_hz in enumerate(f0_grid):
                    selected = amp[which == index]
                    if selected.size:
                        chunks[float(f0_hz)].append(selected.astype(np.float64, copy=False))

    return {
        f0_hz: np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        for f0_hz, parts in chunks.items()
    }


def write_amplitude_summary(amplitudes_by_f0: dict[float, np.ndarray], path: Path) -> dict[float, dict[float, float]]:
    summary: dict[float, dict[float, float]] = {}
    with path.open("w", newline="") as f:
        fieldnames = ["f0_hz", "frequency_bin_low_hz", "frequency_bin_high_hz", "count"] + [
            f"amp_p{p:g}" for p in AMPLITUDE_PERCENTILES
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        edges = frequency_edges(F0_GRID)
        for i, f0_hz in enumerate(F0_GRID):
            amps = amplitudes_by_f0[float(f0_hz)]
            row: dict[str, float | int] = {
                "f0_hz": float(f0_hz),
                "frequency_bin_low_hz": float(edges[i]),
                "frequency_bin_high_hz": float(edges[i + 1]),
                "count": int(amps.size),
            }
            if amps.size:
                values = np.percentile(amps, AMPLITUDE_PERCENTILES)
                summary[float(f0_hz)] = {
                    float(p): float(v) for p, v in zip(AMPLITUDE_PERCENTILES, values, strict=True)
                }
                row.update({f"amp_p{p:g}": float(v) for p, v in zip(AMPLITUDE_PERCENTILES, values, strict=True)})
            else:
                summary[float(f0_hz)] = {float(p): float("nan") for p in AMPLITUDE_PERCENTILES}
                row.update({f"amp_p{p:g}": float("nan") for p in AMPLITUDE_PERCENTILES})
            writer.writerow(row)
    return summary


def plot_amplitude_summary(summary: dict[float, dict[float, float]], counts: dict[float, int], path: Path) -> None:
    colors = {50.0: "#0072B2", 90.0: "#D55E00", 99.0: "#009E73"}
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for percentile in PLOT_PERCENTILES:
        values = np.array([summary[float(f0)][percentile] for f0 in F0_GRID])
        ax.plot(
            F0_GRID,
            values,
            marker="o",
            linewidth=2.0,
            markersize=4.4,
            color=colors[percentile],
            label=f"Sangria amplitude p{percentile:g}",
        )
    for f0_hz in F0_GRID:
        ax.text(
            f0_hz,
            summary[float(f0_hz)][99.0] * 1.16,
            f"{counts[float(f0_hz)]}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=30,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$f_0$ [Hz]")
    ax.set_ylabel("Sangria GB strain amplitude")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, frameon=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def load_production_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open()))


def scaled_residual_table(
    production_rows: list[dict[str, str]],
    amplitude_summary: dict[float, dict[float, float]],
    path: Path,
) -> dict[tuple[float, float, float], float]:
    grouped: dict[tuple[float, float, float], list[float]] = defaultdict(list)
    for row in production_rows:
        eccentricity = float(row["eccentricity"])
        if eccentricity <= 0.0:
            continue
        f0_hz = float(row["f0_hz"])
        residual = float(row["noise_weighted_residual_AE"])
        reference_amplitude = circular_amplitude_at_1kpc(f0_hz, float(row["chirp_mass_solar"]))
        if reference_amplitude <= 0.0:
            continue
        for percentile in PLOT_PERCENTILES:
            sangria_amplitude = amplitude_summary[f0_hz][percentile]
            grouped[(f0_hz, eccentricity, percentile)].append(residual * sangria_amplitude / reference_amplitude)

    medians = {key: float(np.median(values)) for key, values in grouped.items()}
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["f0_hz", "eccentricity", "sangria_amplitude_percentile", "median_scaled_residual_AE"],
        )
        writer.writeheader()
        for (f0_hz, eccentricity, percentile), value in sorted(medians.items()):
            writer.writerow(
                {
                    "f0_hz": f0_hz,
                    "eccentricity": eccentricity,
                    "sangria_amplitude_percentile": percentile,
                    "median_scaled_residual_AE": value,
                }
            )
    return medians


def plot_scaled_residuals(medians: dict[tuple[float, float, float], float], path: Path) -> None:
    colors = {50.0: "#0072B2", 90.0: "#D55E00", 99.0: "#009E73"}
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.8), constrained_layout=True, sharex=True, sharey=True)
    for ax, f0_hz in zip(axes.flat, F0_GRID, strict=True):
        eccentricities = sorted({ecc for f0, ecc, _ in medians if f0 == float(f0_hz)})
        for percentile in PLOT_PERCENTILES:
            y = np.array([medians[(float(f0_hz), ecc, percentile)] for ecc in eccentricities])
            ax.plot(
                eccentricities,
                y,
                marker="o",
                linewidth=1.9,
                markersize=3.7,
                color=colors[percentile],
                label=f"p{percentile:g} amplitude",
            )
        ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.text(0.05, 0.89, f"$f_0={f0_hz:.0e}$ Hz", transform=ax.transAxes, fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.28)
    for ax in axes[-1, :]:
        ax.set_xlabel("eccentricity")
    for ax in axes[:, 0]:
        ax.set_ylabel("Sangria-scaled residual in A/E")
    axes[0, 0].legend(fontsize=8, frameon=True, loc="lower right")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    amplitudes_by_f0 = read_catalog_bin_amplitudes(SANGRIA_PATH, F0_GRID)
    counts = {f0: int(values.size) for f0, values in amplitudes_by_f0.items()}
    summary_path = OUT_DIR / "sangria_gb_amplitude_percentiles_by_f0.csv"
    summary = write_amplitude_summary(amplitudes_by_f0, summary_path)

    amp_plot = OUT_DIR / "sangria_gb_amplitude_percentiles_by_f0.png"
    plot_amplitude_summary(summary, counts, amp_plot)

    rows = load_production_rows(PRODUCTION_CSV)
    scaled_csv = OUT_DIR / "sangria_scaled_noise_weighted_residuals.csv"
    medians = scaled_residual_table(rows, summary, scaled_csv)

    residual_plot = OUT_DIR / "sangria_scaled_noise_weighted_residuals.png"
    plot_scaled_residuals(medians, residual_plot)
    if MANUSCRIPT_FIG_DIR.exists():
        plot_amplitude_summary(summary, counts, MANUSCRIPT_FIG_DIR / amp_plot.name)
        plot_scaled_residuals(medians, MANUSCRIPT_FIG_DIR / residual_plot.name)

    print(f"wrote {summary_path}")
    print(f"wrote {amp_plot}")
    print(f"wrote {scaled_csv}")
    print(f"wrote {residual_plot}")
    for f0_hz in F0_GRID:
        values = summary[float(f0_hz)]
        print(
            f"f0={f0_hz:.0e} Hz: n={counts[float(f0_hz)]}, "
            f"amp p50={values[50.0]:.3e}, p90={values[90.0]:.3e}, p99={values[99.0]:.3e}"
        )


if __name__ == "__main__":
    main()
