"""Compare the old and corrected PN frequency conventions.

This script is intentionally separate from the paper-generation scripts. It
reconstructs the previous 1PN/periastron convention for comparison plots and
does not modify the manuscript.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from egb_multi.constants import C_M_PER_S, G_M3_KG_S2
from egb_multi.eccentric import (
    EccentricBinaryParams,
    eccentric_complex_strain,
    mikkola_kepler,
    peters_mathews_evolution,
)
from production_subdominant_corrections_summary import (
    SECONDS_PER_YEAR,
    load_a_channel_psd,
    make_source,
    mismatch,
)


PAPER_FIGURE_DIR = Path("/Users/shubhanshutiwari/Downloads/Effect_of_eccentricity_for_GB_in_LISA-2/figures")
OUT_DIR = Path(__file__).resolve().parent


def old_eccentric_complex_strain(
    source: EccentricBinaryParams,
    t: np.ndarray,
    *,
    physics_mode: str = "1pn",
    evolution_mode: str = "fixed",
) -> np.ndarray:
    """Previous convention: input was treated as radial n and lambda=(1+k)l."""

    tt = np.asarray(t, dtype=np.float64)
    include_1pn = physics_mode in {"1pn_no_periastron", "1pn"}
    include_periastron = physics_mode == "1pn"

    if evolution_mode in {"peters_mathews", "peters_mathews_eccentricity_only"}:
        evolution = peters_mathews_evolution(source, tt)
        if evolution_mode == "peters_mathews":
            e = evolution["eccentricity"]
            n = evolution["mean_motion"]
            mean_anomaly = evolution["mean_anomaly"]
        else:
            e = evolution["eccentricity"]
            n = source.mean_motion
            mean_anomaly = n * (tt - source.t0)
    else:
        e = source.eccentricity
        n = source.mean_motion
        mean_anomaly = n * (tt - source.t0)

    m = source.total_mass_kg
    eta = source.symmetric_mass_ratio
    gm = G_M3_KG_S2 * m
    xi = gm * n / C_M_PER_S**3
    x_full = xi ** (2.0 / 3.0)
    x = x_full if include_1pn else 0.0

    u = mikkola_kepler(mean_anomaly, e)
    one_minus_ecosu = 1.0 - e * np.cos(u)
    one_minus_e2 = 1.0 - e * e

    semimajor_scale = (gm / n**2) ** (1.0 / 3.0)
    r = semimajor_scale * one_minus_ecosu * (
        1.0 + x * (-18.0 + 2.0 * eta - (6.0 - 7.0 * eta) * e * np.cos(u)) / (6.0 * one_minus_ecosu)
    )
    rdot = e * (gm * n) ** (1.0 / 3.0) * (6.0 + x * (6.0 - 7.0 * eta)) * np.sin(u) / (6.0 * one_minus_ecosu)

    e_phi = e * (1.0 + x * (4.0 - eta))
    v_wrapped = 2.0 * np.arctan2(
        np.sqrt(1.0 + e_phi) * np.sin(u / 2.0),
        np.sqrt(1.0 - e_phi) * np.cos(u / 2.0),
    )
    v = np.unwrap(v_wrapped) if v_wrapped.ndim > 0 else v_wrapped
    u_unwrapped = np.unwrap(u) if np.ndim(u) > 0 else u

    k_1pn = 3.0 * x_full / one_minus_e2 if include_periastron else 0.0
    w = (v - u_unwrapped + e * np.sin(u)) * (1.0 + k_1pn)
    lam = (1.0 + k_1pn) * mean_anomaly + source.phi0
    phi = lam + w
    phidot = (
        n
        * np.sqrt(one_minus_e2)
        / one_minus_ecosu**2
        * (1.0 + x * (3.0 - (4.0 - eta) * e * e + (1.0 - eta) * e * np.cos(u)) / (one_minus_e2 * one_minus_ecosu))
    )

    cos_i = np.cos(source.inclination)
    sin_i = np.sin(source.inclination)
    sin_2phi = np.sin(2.0 * phi)
    cos_2phi = np.cos(2.0 * phi)
    radial = gm / r
    tangential = r * r * phidot * phidot
    radial_velocity = rdot * rdot
    a = radial + tangential - radial_velocity
    b = 2.0 * r * rdot * phidot

    strain_scale = G_M3_KG_S2 * m * eta / (C_M_PER_S**4 * source.distance_m)
    h_cross = -2.0 * strain_scale * cos_i * (a * sin_2phi - b * cos_2phi)
    h_plus = -strain_scale * ((1.0 + cos_i * cos_i) * (a * cos_2phi + b * sin_2phi) + sin_i * sin_i * (radial - tangential - radial_velocity))
    return h_plus.astype(np.complex128) - 1.0j * h_cross.astype(np.complex128)


def plot_phase_drift(path: Path) -> None:
    source = EccentricBinaryParams(
        mean_motion=np.pi * 3.0e-3,
        eccentricity=0.0,
        m1_solar=1.4,
        m2_solar=1.4,
        inclination=0.8,
        phi0=0.0,
    )
    days = 30.0
    t = np.linspace(0.0, days * 86400.0, 4096)
    old = old_eccentric_complex_strain(source, t, physics_mode="1pn")
    new = eccentric_complex_strain(source, t, physics_mode="1pn")
    phase_old = np.unwrap(np.angle(old))
    phase_new = np.unwrap(np.angle(new))
    phase_delta = phase_old - phase_new

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), constrained_layout=True, sharex=True)
    axes[0].plot(t / 86400.0, np.real(old) / np.max(np.abs(old)), label="old convention", color="#D55E00", lw=1.2)
    axes[0].plot(t / 86400.0, np.real(new) / np.max(np.abs(new)), label="new convention", color="#0072B2", lw=1.0, alpha=0.8)
    axes[0].set_ylabel("normalized Re[h]")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(t / 86400.0, phase_delta, color="#009E73", lw=1.4)
    axes[1].set_xlabel("time [days]")
    axes[1].set_ylabel(r"$\Delta \arg h$ [rad]")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(r"Circular limit: old $(1+k)n$ phase vs corrected $\omega$ phase", fontsize=11)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def load_new_source_correction_rows() -> list[dict[str, str]]:
    return list(csv.DictReader((OUT_DIR / "production_subdominant_corrections_summary.csv").open()))


def compute_old_source_correction_rows() -> list[dict[str, object]]:
    psd_freq, psd_a = load_a_channel_psd(OUT_DIR / "lisainstrument_welch_psd.csv")
    duration = 4.0 * SECONDS_PER_YEAR
    dt = 50.0
    times = np.arange(0.0, duration, dt, dtype=np.float64)
    freq = np.fft.rfftfreq(times.size, dt)
    psd = np.interp(freq, psd_freq, psd_a, left=np.inf, right=np.inf)
    frequencies = [2.0e-4, 1.0e-3, 3.0e-3]
    eccentricities = [0.1, 0.3]
    masses = [(0.5, 0.5), (0.8, 0.6), (1.4, 1.4)]

    rows: list[dict[str, object]] = []
    for f0_hz in frequencies:
        band_mask = (freq >= 0.20 * f0_hz) & (freq <= min(freq[-1], 10.0 * f0_hz))
        for eccentricity in eccentricities:
            for m1_solar, m2_solar in masses:
                source = make_source(f0_hz, eccentricity, m1_solar, m2_solar)
                x = (G_M3_KG_S2 * source.total_mass_kg * source.mean_motion / C_M_PER_S**3) ** (2.0 / 3.0)
                old_matched_source = EccentricBinaryParams(
                    mean_motion=source.mean_motion * (1.0 + 3.0 * x / (1.0 - eccentricity**2)),
                    eccentricity=source.eccentricity,
                    m1_solar=source.m1_solar,
                    m2_solar=source.m2_solar,
                    distance_m=source.distance_m,
                    inclination=source.inclination,
                    beta=source.beta,
                    lambda_=source.lambda_,
                    psi=source.psi,
                    phi0=source.phi0,
                    t0=source.t0,
                )
                strains = {
                    "fixed_1pn": old_eccentric_complex_strain(source, times, physics_mode="1pn", evolution_mode="fixed"),
                    "fixed_1pn_no_periastron_matched": old_eccentric_complex_strain(
                        old_matched_source,
                        times,
                        physics_mode="1pn_no_periastron",
                        evolution_mode="fixed",
                    ),
                    "full_pm": old_eccentric_complex_strain(source, times, physics_mode="1pn", evolution_mode="peters_mathews"),
                    "pm_eccentricity_only": old_eccentric_complex_strain(
                        source,
                        times,
                        physics_mode="1pn",
                        evolution_mode="peters_mathews_eccentricity_only",
                    ),
                }
                comparisons = {
                    "periastron": ("fixed_1pn", "fixed_1pn_no_periastron_matched"),
                    "full_pm_vs_fixed": ("full_pm", "fixed_1pn"),
                    "pm_eccentricity_only": ("pm_eccentricity_only", "fixed_1pn"),
                }
                for comparison, (reference, template) in comparisons.items():
                    rows.append(
                        {
                            "convention": "old n-based",
                            "comparison": comparison,
                            "f0_hz": f0_hz,
                            "eccentricity": eccentricity,
                            "m1_solar": m1_solar,
                            "m2_solar": m2_solar,
                            "mismatch": mismatch(strains[reference], strains[template], psd, dt, band_mask.copy()),
                        }
                    )
    return rows


def plot_source_correction_overlay(old_rows: list[dict[str, object]], new_rows: list[dict[str, str]], path: Path) -> None:
    normalized_new_rows: list[dict[str, object]] = []
    name_map = {
        "periastron_same_azimuthal_frequency": "periastron",
        "full_pm_vs_fixed": "full_pm_vs_fixed",
        "pm_eccentricity_only": "pm_eccentricity_only",
    }
    for row in new_rows:
        normalized_new_rows.append(
            {
                "convention": "new omega-based",
                "comparison": name_map[row["comparison"]],
                "f0_hz": float(row["f0_hz"]),
                "eccentricity": float(row["eccentricity"]),
                "m1_solar": float(row["m1_solar"]),
                "m2_solar": float(row["m2_solar"]),
                "mismatch": float(row["mismatch"]),
            }
        )
    rows = old_rows + normalized_new_rows
    comparisons = ["periastron", "full_pm_vs_fixed", "pm_eccentricity_only"]
    labels = {
        "periastron": "periastron / no-periastron",
        "full_pm_vs_fixed": "full PM / fixed eccentric",
        "pm_eccentricity_only": "PM e(t) only / fixed eccentric",
    }
    colors = {
        "periastron": "#D55E00",
        "full_pm_vs_fixed": "#009E73",
        "pm_eccentricity_only": "#CC79A7",
    }
    styles = {"old n-based": "--", "new omega-based": "-"}
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})
    fig, axes = plt.subplots(1, len(eccentricities), figsize=(12.2, 4.4), constrained_layout=True, sharey=True)
    for ax, eccentricity in zip(axes, eccentricities, strict=True):
        subset_e = [row for row in rows if float(row["eccentricity"]) == eccentricity]
        frequencies = np.array(sorted({float(row["f0_hz"]) for row in subset_e}))
        for comparison in comparisons:
            for convention in ["old n-based", "new omega-based"]:
                medians = []
                for f0_hz in frequencies:
                    values = [
                        max(float(row["mismatch"]), 1.0e-16)
                        for row in subset_e
                        if row["comparison"] == comparison and row["convention"] == convention and float(row["f0_hz"]) == f0_hz
                    ]
                    medians.append(np.median(values))
                ax.plot(
                    frequencies,
                    medians,
                    color=colors[comparison],
                    linestyle=styles[convention],
                    marker="o" if convention == "new omega-based" else "s",
                    linewidth=2.0 if convention == "new omega-based" else 1.5,
                    alpha=0.95 if convention == "new omega-based" else 0.65,
                    label=f"{labels[comparison]}, {convention}",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$f_0$ [Hz]")
        ax.set_title(rf"$e_0={eccentricity:g}$")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"source-level mismatch, $1-\mathrm{FF}$")
    axes[-1].legend(fontsize=7, frameon=True)
    fig.suptitle("Effect of PN frequency convention on source-correction tests", fontsize=11)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def image_side_by_side(left_path: Path, right_path: Path, title: str, out_path: Path) -> None:
    left = mpimg.imread(left_path)
    right = mpimg.imread(right_path)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), constrained_layout=True)
    for ax, img, subtitle in zip(axes, [left, right], ["old version", "new version"], strict=True):
        ax.imshow(img)
        ax.set_title(subtitle, fontsize=12)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    phase_path = OUT_DIR / "pn_convention_phase_old_vs_new.png"
    source_overlay_path = OUT_DIR / "pn_convention_source_corrections_old_vs_new_overlay.png"
    source_side_by_side_path = OUT_DIR / "pn_convention_source_corrections_old_vs_new_side_by_side.png"
    full_side_by_side_path = OUT_DIR / "pn_convention_full_model_old_vs_new_side_by_side.png"

    plot_phase_drift(phase_path)
    old_rows = compute_old_source_correction_rows()
    with (OUT_DIR / "pn_convention_old_source_corrections.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(old_rows[0]))
        writer.writeheader()
        writer.writerows(old_rows)
    plot_source_correction_overlay(old_rows, load_new_source_correction_rows(), source_overlay_path)
    image_side_by_side(
        PAPER_FIGURE_DIR / "source_corrections_summary.png",
        PAPER_FIGURE_DIR / "production_subdominant_corrections_summary.png",
        "Optional source-correction plot before and after the PN convention fix",
        source_side_by_side_path,
    )
    image_side_by_side(
        PAPER_FIGURE_DIR / "full_model_vs_fixed_circular.png",
        PAPER_FIGURE_DIR / "production_full_model_vs_fixed_circular.png",
        "Fixed eccentric baseline comparison before and after the PN convention fix",
        full_side_by_side_path,
    )
    print(f"wrote {phase_path}")
    print(f"wrote {source_overlay_path}")
    print(f"wrote {source_side_by_side_path}")
    print(f"wrote {full_side_by_side_path}")


if __name__ == "__main__":
    main()
