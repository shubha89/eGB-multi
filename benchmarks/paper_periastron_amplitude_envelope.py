"""Amplitude-envelope comparison for Newtonian and 1PN periastron waveforms."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import ANNOTATION_SIZE, LEGEND_SIZE, apply_tick_style, use_paper_plot_style

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain, eccentric_dynamics  # noqa: E402


PAPER_FIGURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "Effect_of_eccentricity_for_GB_in_LISA-new"
    / "figures"
)


def source() -> EccentricBinaryParams:
    return EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.6,
        m1_solar=1.4,
        m2_solar=1.4,
        inclination=0.8,
        beta=0.2,
        lambda_=-1.0,
        psi=0.4,
        phi0=0.2,
    )


def sliding_rms_envelope(series: np.ndarray, samples_per_window: int) -> np.ndarray:
    window = np.ones(samples_per_window, dtype=np.float64) / samples_per_window
    power = np.convolve(np.real(series) ** 2, window, mode="same")
    return np.sqrt(np.maximum(power, 0.0))


def binned_minmax(
    x: np.ndarray,
    y: np.ndarray,
    samples_per_bin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    usable = (x.size // samples_per_bin) * samples_per_bin
    xx = x[:usable].reshape(-1, samples_per_bin)
    yy = y[:usable].reshape(-1, samples_per_bin)
    return np.mean(xx, axis=1), np.min(yy, axis=1), np.max(yy, axis=1)


def periastron_timescale_days(source_params: EccentricBinaryParams) -> tuple[float, float]:
    dynamics = eccentric_dynamics(
        source_params,
        np.asarray([0.0], dtype=np.float64),
        physics_mode="1pn+periastron",
        evolution_mode="fixed",
    )
    azimuthal = float(np.asarray(dynamics["azimuthal_frequency"])[0])
    radial = float(np.asarray(dynamics["radial_mean_motion"])[0])
    full_period_days = 2.0 * np.pi / (azimuthal - radial) / 86400.0
    return full_period_days, 0.5 * full_period_days


def main() -> None:
    use_paper_plot_style()
    out_dir = Path(__file__).resolve().parent
    src = source()
    duration_days = 356.0
    dt = 20.0
    t = np.arange(0.0, duration_days * 86400.0, dt)
    days = t / 86400.0

    newtonian = eccentric_complex_strain(src, t, physics_mode="newtonian", evolution_mode="fixed")
    with_periastron = eccentric_complex_strain(src, t, physics_mode="1pn+periastron", evolution_mode="fixed")

    envelope_window_days = 0.5
    samples_per_window = int(round(envelope_window_days * 86400.0 / dt))
    env_newtonian = sliding_rms_envelope(newtonian, samples_per_window)
    env_with_periastron = sliding_rms_envelope(with_periastron, samples_per_window)

    edge = samples_per_window
    valid = slice(edge, -edge)
    normalization = np.median(env_with_periastron[valid])
    env_newtonian_norm = env_newtonian / normalization
    env_with_periastron_norm = env_with_periastron / normalization
    envelope_ratio = env_with_periastron / env_newtonian
    periastron_period_days, envelope_period_days = periastron_timescale_days(src)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.6), constrained_layout=True)

    waveform_scale = np.max(np.abs(np.real(with_periastron)))
    bin_days = 0.2
    samples_per_bin = int(round(bin_days * 86400.0 / dt))
    bin_days_mid, peri_min, peri_max = binned_minmax(
        days,
        np.real(with_periastron) / waveform_scale,
        samples_per_bin,
    )
    _, newt_min, newt_max = binned_minmax(
        days,
        np.real(newtonian) / waveform_scale,
        samples_per_bin,
    )
    axes[0].fill_between(
        bin_days_mid,
        peri_min,
        peri_max,
        color="#0072B2",
        alpha=0.34,
        linewidth=0.0,
        label="1PN+periastron",
    )
    axes[0].fill_between(
        bin_days_mid,
        newt_min,
        newt_max,
        color="#D55E00",
        alpha=0.34,
        linewidth=0.0,
        label="Newtonian",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.7, alpha=0.55)
    axes[0].set_xlim(0.0, duration_days)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_xlabel("time [days]")
    axes[0].set_ylabel(r"normalized ${\rm Re}[h(t)]$")
    axes[0].set_title(r"Full-baseline time-domain strain range ($0.2\,{\rm d}$ bins)")
    axes[0].legend(fontsize=LEGEND_SIZE, ncol=2, loc="upper right")

    stride = max(1, int(round(0.05 * 86400.0 / dt)))
    axes[1].plot(days[::stride], envelope_ratio[::stride], color="#009E73", linewidth=1.5)
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("time [days]")
    axes[1].set_ylabel(
        r"$A_{\rm env}^{\rm RMS}({\rm 1PN+peri})$"
        "\n"
        r"$/$ $A_{\rm env}^{\rm RMS}({\rm Newtonian})$"
    )
    axes[1].set_title(r"Sliding RMS amplitude-envelope ratio, $0.5\,{\rm d}$ window")
    axes[1].set_ylim(
        np.percentile(envelope_ratio[valid], 0.5) * 0.995,
        np.percentile(envelope_ratio[valid], 99.5) * 1.005,
    )
    arrow_start = 90.0
    arrow_end = arrow_start + envelope_period_days
    arrow_y = np.percentile(envelope_ratio[valid], 99.0) * 0.998
    axes[1].annotate(
        "",
        xy=(arrow_start, arrow_y),
        xytext=(arrow_end, arrow_y),
        arrowprops={"arrowstyle": "<->", "color": "0.2", "linewidth": 1.0},
    )
    axes[1].text(
        0.5 * (arrow_start + arrow_end),
        arrow_y * 1.0008,
        rf"$T_\varpi/2 \simeq {envelope_period_days:.0f}\,{{\rm d}}$",
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_SIZE,
        color="0.2",
    )
    axes[1].text(
        0.02,
        0.93,
        rf"$T_\varpi \simeq {periastron_period_days:.0f}\,{{\rm d}}$",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_SIZE,
        color="0.2",
    )

    for ax in axes:
        ax.grid(True, alpha=0.3)
    apply_tick_style(axes)

    png_path = out_dir / "paper_periastron_timeseries_envelope_ratio_1pn_peri_vs_newtonian_e06_m14_m14_356days.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    csv_path = out_dir / "paper_periastron_amplitude_envelope_ratio_1pn_peri_vs_newtonian_e06_m14_m14_356days.csv"
    sample_indices = np.linspace(0, t.size - 1, 1001, dtype=int)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time_days", "envelope_1pn_periastron", "envelope_newtonian", "envelope_ratio"],
        )
        writer.writeheader()
        for index in sample_indices:
            writer.writerow(
                {
                    "time_days": float(days[index]),
                    "envelope_1pn_periastron": float(env_with_periastron_norm[index]),
                    "envelope_newtonian": float(env_newtonian_norm[index]),
                    "envelope_ratio": float(envelope_ratio[index]),
                }
            )

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    target = PAPER_FIGURE_DIR / png_path.name
    shutil.copy2(png_path, target)
    print(f"ratio_min={np.min(envelope_ratio[valid]):.6f}")
    print(f"ratio_max={np.max(envelope_ratio[valid]):.6f}")
    print(f"ratio_std={np.std(envelope_ratio[valid]):.6f}")
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    print(f"copied {target}")


if __name__ == "__main__":
    main()
