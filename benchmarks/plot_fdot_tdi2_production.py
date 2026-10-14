"""Paper-style plots for the TDI-2, GR-fdot production grid."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def chirp_mass(m1: float, m2: float) -> float:
    total = m1 + m2
    eta = m1 * m2 / total**2
    return total * eta ** (3.0 / 5.0)


def load_rows(path: Path) -> list[dict[str, object]]:
    return list(csv.DictReader(path.open()))


def plot_lines(rows: list[dict[str, object]], path: Path) -> None:
    frequencies = sorted({float(row["f0_hz"]) for row in rows})
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    for color, f0_hz in zip(colors, frequencies, strict=True):
        medians = []
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

    ax.set_xscale("symlog", linthresh=1.0e-3)
    ax.set_yscale("log")
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("A/E mismatch, $1-\\mathrm{FF}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, frameon=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_residual_lines(rows: list[dict[str, object]], path: Path) -> None:
    frequencies = sorted({float(row["f0_hz"]) for row in rows})
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    column = "noise_weighted_residual_AE"
    for color, f0_hz in zip(colors, frequencies, strict=True):
        medians = []
        lows = []
        highs = []
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
            lows.append(np.percentile(values, 16.0))
            highs.append(np.percentile(values, 84.0))

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
        ax.fill_between(x, lows, highs, color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("symlog", linthresh=1.0e-3)
    ax.set_yscale("log")
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("relative noise-weighted residual in A/E")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, frameon=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_contours(rows: list[dict[str, object]], path: Path) -> None:
    frequencies = sorted({float(row["f0_hz"]) for row in rows})
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})
    mass_pairs = sorted(
        {(float(row["m1_solar"]), float(row["m2_solar"])) for row in rows},
        key=lambda pair: chirp_mass(*pair),
    )
    chirp_masses = np.array([chirp_mass(*pair) for pair in mass_pairs])
    positive = [float(row["mismatch_AE"]) for row in rows if float(row["mismatch_AE"]) > 0.0]
    vmin = max(min(positive), 1.0e-8)
    vmax = max(positive)
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 28)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.4), constrained_layout=True, sharex=True, sharey=True)
    last_mesh = None
    for ax, f0_hz in zip(axes.flat, frequencies, strict=True):
        z = np.zeros((len(eccentricities), len(chirp_masses)), dtype=np.float64)
        for i, eccentricity in enumerate(eccentricities):
            for j, pair in enumerate(mass_pairs):
                values = [
                    max(float(row["mismatch_AE"]), vmin)
                    for row in rows
                    if float(row["f0_hz"]) == f0_hz
                    and float(row["eccentricity"]) == eccentricity
                    and float(row["m1_solar"]) == pair[0]
                    and float(row["m2_solar"]) == pair[1]
                ]
                z[i, j] = np.median(values)

        last_mesh = ax.contourf(
            chirp_masses,
            eccentricities,
            np.maximum(z, vmin),
            levels=levels,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="viridis",
            extend="both",
        )
        ax.contour(
            chirp_masses,
            eccentricities,
            np.maximum(z, vmin),
            levels=levels[::4],
            colors="white",
            linewidths=0.45,
            alpha=0.55,
        )
        ax.text(0.05, 0.9, f"$f_0={f0_hz:.0e}$ Hz", transform=ax.transAxes)
        ax.set_yscale("symlog", linthresh=1.0e-3)
        ax.grid(True, alpha=0.18)
        ax.set_xticks(chirp_masses)
        ax.set_xticklabels([f"{cm:.2f}" for cm in chirp_masses])

    for ax in axes[:, 0]:
        ax.set_ylabel("eccentricity")
    for ax in axes[-1, :]:
        ax.set_xlabel("chirp mass [$M_\\odot$]")

    assert last_mesh is not None
    cbar = fig.colorbar(last_mesh, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("median A/E mismatch, $1-\\mathrm{FF}$")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "production_eccentric_circular_fdot_tdi2_signal_gr_template_grid_production.csv"
    rows = load_rows(csv_path)
    line_path = out_dir / "production_fdot_tdi2_mismatch_vs_eccentricity_by_f0.png"
    residual_path = out_dir / "production_fdot_tdi2_residual_vs_eccentricity_by_f0.png"
    contour_path = out_dir / "production_fdot_tdi2_mass_eccentricity_mismatch_contours.png"
    plot_lines(rows, line_path)
    plot_residual_lines(rows, residual_path)
    plot_contours(rows, contour_path)
    print(f"rows={len(rows)}")
    print(f"wrote {line_path}")
    print(f"wrote {residual_path}")
    print(f"wrote {contour_path}")


if __name__ == "__main__":
    main()
