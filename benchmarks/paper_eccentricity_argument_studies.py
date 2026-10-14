"""Diagnostics supporting the eccentric time-domain GB argument.

These plots are meant for the paper narrative rather than for a new production
response grid. They reuse the production CSV when possible and otherwise use
short source-level calculations or precomputed LISA orbit geometry.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import (
    EccentricBinaryParams,
    eccentric_harmonic_angular_frequency,
    eccentric_harmonic_projection_angular_frequency,
    eccentric_polarizations,
)
from egb_multi.external import default_lisaorbits, state_from_lisaorbits


PRODUCTION_CSV = Path(__file__).resolve().parent / "production_eccentric_circular_fdot_tdi2_signal_gr_template_grid_production.csv"
PAPER_FIGURE_DIR = Path("/Users/shubhanshutiwari/Downloads/Effect_of_eccentricity_for_GB_in_LISA-2/figures")
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def make_source(eccentricity: float, f0_hz: float = 1.0e-3) -> EccentricBinaryParams:
    return EccentricBinaryParams(
        mean_motion=np.pi * f0_hz,
        eccentricity=eccentricity,
        m1_solar=0.5,
        m2_solar=0.5,
        inclination=0.8,
        beta=0.2,
        lambda_=-1.0,
        psi=0.4,
        phi0=0.2,
    )


def harmonic_power(eccentricity: float, max_harmonic: int = 40, samples: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    source = make_source(eccentricity)
    projection_frequency = eccentric_harmonic_projection_angular_frequency(source)
    period = 2.0 * np.pi / projection_frequency
    times = np.linspace(0.0, period, samples, endpoint=False)
    h_plus, h_cross = eccentric_polarizations(source, times)
    harmonics = np.arange(1, max_harmonic + 1)
    power = []
    for harmonic in harmonics:
        angular_frequency = eccentric_harmonic_angular_frequency(source, int(harmonic))
        carrier_projection = np.exp(-1.0j * angular_frequency * times)
        coeff_plus = np.mean(h_plus * carrier_projection)
        coeff_cross = np.mean(h_cross * carrier_projection)
        power.append(np.abs(coeff_plus) ** 2 + np.abs(coeff_cross) ** 2)
    power = np.asarray(power)
    total = np.sum(power)
    if total > 0.0:
        power = power / total
    return harmonics, power


def plot_source_harmonics(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    widths = 0.18
    eccentricities = [0.0, 0.05, 0.1, 0.3]
    for offset, (eccentricity, color) in enumerate(zip(eccentricities, COLORS, strict=False)):
        harmonics, power = harmonic_power(eccentricity, max_harmonic=12)
        ax.bar(
            harmonics + (offset - 1.5) * widths,
            np.maximum(power, 1.0e-12),
            width=widths,
            color=color,
            label=f"$e={eccentricity:g}$",
            alpha=0.9,
        )
    ax.set_yscale("log")
    ax.set_ylim(1.0e-8, 1.2)
    ax.set_xticks(np.arange(1, 13))
    ax.set_xlabel("source harmonic index")
    ax.set_ylabel("fractional source power")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(frameon=True, fontsize=9)
    path = out_dir / "paper_source_harmonic_content.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def load_production_rows() -> list[dict[str, str]]:
    return list(csv.DictReader(PRODUCTION_CSV.open()))


def plot_fdot_absorption(out_dir: Path) -> Path:
    rows = load_production_rows()
    fdot_factors = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    displayed_eccentricities = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]

    x = np.arange(fdot_factors.size, dtype=np.float64)
    width = 0.12
    offsets = (np.arange(len(displayed_eccentricities)) - 0.5 * (len(displayed_eccentricities) - 1)) * width
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for offset, color, eccentricity in zip(offsets, COLORS, displayed_eccentricities, strict=True):
        subset = [row for row in rows if float(row["eccentricity"]) == eccentricity]
        best_match_factors = []
        for row in subset:
            gr_fdot = float(row["circular_gr_fdot_hz_s"])
            best_fdot = float(row["best_circular_fdot_hz_s"])
            best_match_factors.append(best_fdot / gr_fdot if gr_fdot != 0.0 else 0.0)
        best_match_factors = np.array(best_match_factors)
        counts = np.array([np.sum(np.isclose(best_match_factors, factor)) for factor in fdot_factors])
        ax.bar(
            x + offset,
            counts,
            width=width,
            color=color,
            alpha=0.88,
            label=f"$e_0={eccentricity:g}$",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{factor:g}" for factor in fdot_factors])
    ax.set_xlabel("match-maximizing $\\dot f_\\mathrm{circ}/\\dot f_\\mathrm{GR}$")
    ax.set_ylabel("number of production-grid cases")
    ax.grid(True, which="major", axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=3, frameon=True)

    path = out_dir / "paper_fdot_absorption_diagnostic.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_frequency_domain_complexity(out_dir: Path) -> Path:
    eccentricities = np.array([0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    thresholds = [0.9, 0.99, 0.999]
    required = {threshold: [] for threshold in thresholds}
    entropy = []
    for eccentricity in eccentricities:
        harmonics, power = harmonic_power(float(eccentricity), max_harmonic=40)
        cumulative = np.cumsum(power)
        for threshold in thresholds:
            index = int(np.searchsorted(cumulative, threshold, side="left"))
            required[threshold].append(int(harmonics[min(index, harmonics.size - 1)]))
        normalized = power[power > 0.0]
        entropy.append(float(np.exp(-np.sum(normalized * np.log(normalized)))))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for color, threshold in zip(COLORS, thresholds, strict=False):
        axes[0].plot(
            eccentricities,
            required[threshold],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{100.0 * threshold:.1f}% power",
        )
    axes[0].set_xlabel("eccentricity")
    axes[0].set_ylabel("highest harmonic required")
    axes[0].set_yticks(np.arange(1, 15, 1))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=True, fontsize=9)

    axes[1].plot(eccentricities, entropy, marker="o", linewidth=2.0, color="#0072B2")
    axes[1].set_xlabel("eccentricity")
    axes[1].set_ylabel("effective number of powered harmonics")
    axes[1].grid(True, alpha=0.3)

    path = out_dir / "paper_frequency_domain_harmonic_complexity.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_lisa_time_domain_response_need(out_dir: Path) -> Path:
    duration = 365.25 * 86400.0
    times = np.linspace(0.0, duration, 4096)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    link_pairs = [(0, 1), (1, 2), (2, 0), (0, 2), (2, 1), (1, 0)]
    labels = ["21", "32", "13", "31", "23", "12"]
    days = times / 86400.0

    arm_seconds = []
    for sender, receiver in link_pairs:
        arm_seconds.append(state.arm_length(sender, receiver))
    arm_seconds = np.asarray(arm_seconds)
    mean_arm = np.mean(arm_seconds, axis=1, keepdims=True)
    fractional = (arm_seconds - mean_arm) / mean_arm

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for label, series, color in zip(labels, fractional, COLORS, strict=True):
        axes[0].plot(days, 1.0e3 * series, linewidth=1.4, color=color, label=label)
    axes[0].set_xlabel("time [days]")
    axes[0].set_ylabel("arm-delay variation [parts per thousand]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="link", fontsize=8, ncol=2, frameon=True)

    spread = np.max(arm_seconds, axis=0) - np.min(arm_seconds, axis=0)
    axes[1].plot(days, spread, linewidth=2.0, color="#D55E00")
    axes[1].set_xlabel("time [days]")
    axes[1].set_ylabel("instantaneous max-min arm delay [s]")
    axes[1].grid(True, alpha=0.3)

    path = out_dir / "paper_lisa_time_domain_response_diagnostic.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    paths = [
        plot_source_harmonics(out_dir),
        plot_fdot_absorption(out_dir),
        plot_frequency_domain_complexity(out_dir),
        plot_lisa_time_domain_response_need(out_dir),
    ]
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = PAPER_FIGURE_DIR / path.name
        shutil.copy2(path, target)
        print(f"wrote {path}")
        print(f"copied {target}")


if __name__ == "__main__":
    main()
