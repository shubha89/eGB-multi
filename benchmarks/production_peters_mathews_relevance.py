"""Peters-Mathews relevance maps for Galactic-binary eccentricity studies.

The purpose is to show where secular vacuum-GR evolution is negligible over a
LISA observation, and where it should be retained. The diagnostic uses the
leading Peters-Mathews derivatives at the initial frequency/eccentricity:

    phase_drift_rad ~= ndot * T^2
    fractional_e_change ~= |edot| * T / e

Here ``f0`` is the dominant nearly circular GW frequency, so ``n = pi f0``.
The accumulated phase is the dominant GW phase drift from the changing mean
motion. This is the quantity most relevant for deciding whether evolution
matters in matched filtering.

Run with:

    python benchmarks/production_peters_mathews_relevance.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import ANNOTATION_SIZE, LEGEND_SIZE, apply_tick_style, use_paper_plot_style

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.constants import C_M_PER_S, G_M3_KG_S2, JULIAN_YEAR_S, SOLAR_MASS_KG


def peters_mathews_rates_from_chirp_mass(
    f0_hz: np.ndarray,
    eccentricity: float,
    chirp_mass_solar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ndot, edot)`` for arrays of f0 and chirp mass.

    The Peters-Mathews mass factor is ``eta (G M)^(5/3)``, equivalent to
    ``(G Mc)^(5/3)``. Therefore a chirp-mass grid is enough for the secular
    evolution rates at this order.
    """

    n = np.pi * np.asarray(f0_hz, dtype=np.float64)
    e = float(eccentricity)
    mc_kg = np.asarray(chirp_mass_solar, dtype=np.float64) * SOLAR_MASS_KG
    mass_factor = (G_M3_KG_S2 * mc_kg) ** (5.0 / 3.0) / C_M_PER_S**5
    one_minus_e2 = 1.0 - e * e
    frequency_boost = (1.0 + 73.0 * e * e / 24.0 + 37.0 * e**4 / 96.0) / one_minus_e2 ** (7.0 / 2.0)
    decay_boost = e * (1.0 + 121.0 * e * e / 304.0) / one_minus_e2 ** (5.0 / 2.0)
    ndot = (96.0 / 5.0) * mass_factor * n ** (11.0 / 3.0) * frequency_boost
    edot = -(304.0 / 15.0) * mass_factor * n ** (8.0 / 3.0) * decay_boost
    return ndot, edot


def chirp_mass_solar(m1_solar: float, m2_solar: float) -> float:
    total = m1_solar + m2_solar
    return (m1_solar * m2_solar) ** (3.0 / 5.0) / total ** (1.0 / 5.0)


def phase_and_echange(
    f0_hz: np.ndarray,
    eccentricity: float,
    chirp_mass_solar_value: np.ndarray,
    duration_years: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = duration_years * JULIAN_YEAR_S
    ndot, edot = peters_mathews_rates_from_chirp_mass(f0_hz, eccentricity, chirp_mass_solar_value)
    phase_rad = ndot * duration**2
    fdot_gw = ndot / np.pi
    if eccentricity == 0.0:
        fractional_e_change = np.zeros_like(phase_rad)
    else:
        fractional_e_change = np.abs(edot) * duration / eccentricity
    return phase_rad, fractional_e_change, fdot_gw


def write_reference_table(path: Path, duration_years: float) -> None:
    masses = [(0.3, 0.3), (0.5, 0.5), (0.8, 0.6), (1.0, 1.0), (1.4, 1.4)]
    frequencies = [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2]
    eccentricities = [0.0, 0.1, 0.3]
    rows = []
    for m1, m2 in masses:
        mc = chirp_mass_solar(m1, m2)
        for f0 in frequencies:
            for eccentricity in eccentricities:
                phase_rad, fractional_e_change, fdot_gw = phase_and_echange(
                    np.array(f0),
                    eccentricity,
                    np.array(mc),
                    duration_years,
                )
                rows.append(
                    {
                        "duration_years": duration_years,
                        "m1_solar": m1,
                        "m2_solar": m2,
                        "chirp_mass_solar": mc,
                        "f0_hz": f0,
                        "eccentricity": eccentricity,
                        "fdot_gw_hz_s": float(fdot_gw),
                        "phase_drift_rad": float(phase_rad),
                        "phase_drift_cycles": float(phase_rad / (2.0 * np.pi)),
                        "fractional_e_change": float(fractional_e_change),
                    }
                )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_threshold_contours(ax, f_grid_hz, mc_grid, quantity, *, phase: bool) -> None:
    if phase:
        levels = [0.1, 1.0, 10.0]
        labels = {0.1: "0.1 rad", 1.0: "1 rad", 10.0: "10 rad"}
    else:
        levels = [1.0e-4, 1.0e-3, 1.0e-2]
        labels = {1.0e-4: "1e-4", 1.0e-3: "1e-3", 1.0e-2: "1e-2"}
    contour = ax.contour(f_grid_hz, mc_grid, quantity, levels=levels, colors="white", linewidths=0.9)
    ax.clabel(contour, fmt=labels, fontsize=8)


def make_relevance_map(path: Path, duration_years: float, eccentricity: float) -> None:
    use_paper_plot_style()
    f0 = np.geomspace(1.0e-5, 2.0e-2, 260)
    mc = np.geomspace(0.15, 1.4, 180)
    f_grid, mc_grid = np.meshgrid(f0, mc)
    phase_rad, fractional_e_change, fdot_gw = phase_and_echange(f_grid, eccentricity, mc_grid, duration_years)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    cmap = "viridis"
    panels = [
        (phase_rad, "accumulated PM phase drift [rad]", 1.0e-4, 1.0e4, True),
        (fractional_e_change, "fractional eccentricity change |de|/e", 1.0e-8, 1.0e-1, False),
        (fdot_gw, "instantaneous GW fdot [Hz/s]", 1.0e-23, 1.0e-12, False),
    ]
    for ax, (quantity, label, vmin, vmax, is_phase) in zip(axes, panels, strict=True):
        mesh = ax.pcolormesh(
            f_grid,
            mc_grid,
            np.clip(quantity, vmin, vmax),
            shading="auto",
            norm="log",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(mesh, ax=ax)
        if is_phase:
            add_threshold_contours(ax, f_grid, mc_grid, quantity, phase=True)
        elif "eccentricity" in label:
            add_threshold_contours(ax, f_grid, mc_grid, quantity, phase=False)
        ax.axvspan(1.0e-5, 1.0e-2, color="white", alpha=0.10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"dominant GW frequency $f_0$ [Hz]")
        ax.grid(True, which="both", alpha=0.18)
    axes[0].set_ylabel("chirp mass [Msun]")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_frequency_curves(path: Path, duration_years: float) -> None:
    use_paper_plot_style()
    f0 = np.geomspace(1.0e-5, 2.0e-2, 500)
    mass_sets = [(0.5, 0.5), (1.4, 1.4)]
    eccentricities = [0.0, 0.1, 0.3, 0.5, 0.8]
    colors = ["#0072B2", "#D55E00"]
    linestyles = {
        0.0: "-",
        0.1: "--",
        0.3: ":",
        0.5: "-.",
        0.8: (0, (3.0, 1.2, 1.0, 1.2, 1.0, 1.2)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True, sharex=True)
    for color, (m1, m2) in zip(colors, mass_sets, strict=True):
        mc = chirp_mass_solar(m1, m2)
        for eccentricity in eccentricities:
            phase_rad, fractional_e_change, _ = phase_and_echange(f0, eccentricity, np.full_like(f0, mc), duration_years)
            axes[0].plot(f0, phase_rad, color=color, linestyle=linestyles[eccentricity], linewidth=1.8)
            if eccentricity > 0.0:
                axes[1].plot(f0, fractional_e_change, color=color, linestyle=linestyles[eccentricity], linewidth=1.8)

    axes[0].axhline(0.1, color="0.45", linewidth=0.9)
    axes[0].axhline(1.0, color="0.2", linewidth=1.0)
    axes[0].text(1.2e-5, 0.12, "0.1 rad", fontsize=ANNOTATION_SIZE, color="0.35")
    axes[0].text(1.2e-5, 1.2, "1 rad", fontsize=ANNOTATION_SIZE, color="0.2")
    axes[1].axhline(1.0e-3, color="0.35", linewidth=0.9)
    axes[1].text(1.2e-5, 1.2e-3, "|de|/e=1e-3", fontsize=ANNOTATION_SIZE, color="0.3")

    axes[0].set_ylabel("accumulated PM phase drift [rad]")
    axes[1].set_ylabel("fractional eccentricity change |de|/e")
    for ax in axes:
        ax.axvspan(1.0e-5, 1.0e-2, color="#56B4E9", alpha=0.10, label="LISA sensitivity band" if ax is axes[0] else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"dominant GW frequency $f_0$ [Hz]")
        ax.grid(True, which="both", alpha=0.25)
    mass_handles = [
        plt.Line2D([0], [0], color=color, linestyle="-", linewidth=2.0, label=f"{m1:g}+{m2:g} Msun")
        for color, (m1, m2) in zip(colors, mass_sets, strict=True)
    ]
    eccentricity_handles = [
        plt.Line2D([0], [0], color="0.25", linestyle=linestyles[eccentricity], linewidth=2.0, label=f"e={eccentricity:g}")
        for eccentricity in eccentricities
    ]
    nonzero_eccentricity_handles = [
        plt.Line2D([0], [0], color="0.25", linestyle=linestyles[eccentricity], linewidth=2.0, label=f"e={eccentricity:g}")
        for eccentricity in eccentricities
        if eccentricity > 0.0
    ]
    first_legend = axes[0].legend(handles=mass_handles, fontsize=LEGEND_SIZE, frameon=False, loc="lower right")
    axes[0].add_artist(first_legend)
    axes[0].legend(handles=eccentricity_handles, fontsize=LEGEND_SIZE, frameon=False, loc="upper left")
    axes[1].legend(handles=nonzero_eccentricity_handles, fontsize=LEGEND_SIZE, frameon=False, loc="lower right")
    apply_tick_style(axes)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    duration_years = 4.0
    csv_path = out_dir / "production_peters_mathews_relevance.csv"
    map_path = out_dir / "production_peters_mathews_relevance_map_e01.png"
    curves_path = out_dir / "production_peters_mathews_relevance_curves.png"

    write_reference_table(csv_path, duration_years)
    make_relevance_map(map_path, duration_years, eccentricity=0.1)
    make_frequency_curves(curves_path, duration_years)
    print(f"wrote {csv_path}")
    print(f"wrote {map_path}")
    print(f"wrote {curves_path}")


if __name__ == "__main__":
    main()
