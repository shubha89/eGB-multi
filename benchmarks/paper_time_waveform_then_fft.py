"""Show time-domain eccentric GB waveforms and harmonic power."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import ANNOTATION_SIZE, TITLE_SIZE, apply_tick_style, use_paper_plot_style

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain


PAPER_FIGURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "Effect_of_eccentricity_for_GB_in_LISA-new"
    / "figures"
)


def source(eccentricity: float, f0_hz: float) -> EccentricBinaryParams:
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


def harmonic_relative_power(
    h_plus: np.ndarray,
    dt: float,
    f0_hz: float,
    harmonics: np.ndarray,
) -> np.ndarray:
    """Return harmonic powers normalized to the strongest displayed harmonic."""

    spectrum = np.fft.rfft(h_plus - np.mean(h_plus)) * dt
    freq = np.fft.rfftfreq(h_plus.size, dt)
    powers = []
    for harmonic in harmonics:
        harmonic_frequency = 0.5 * float(harmonic) * f0_hz
        index = int(np.argmin(np.abs(freq - harmonic_frequency)))
        powers.append(float(np.abs(spectrum[index]) ** 2))
    powers_array = np.asarray(powers, dtype=np.float64)
    peak = float(np.max(powers_array))
    if peak == 0.0:
        return powers_array
    return powers_array / peak


def main() -> None:
    use_paper_plot_style()
    out_dir = Path(__file__).resolve().parent
    f0_hz = 1.0e-3
    eccentricities = [0.0, 0.3, 0.6]
    colors = ["#0072B2", "#D55E00", "#009E73"]
    dt = 5.0
    radial_period = 2.0 / f0_hz
    duration = 512.0 * radial_period
    times = np.arange(0.0, duration, dt)
    harmonics = np.arange(1, 13, dtype=np.int64)
    harmonic_positions = 0.5 * harmonics.astype(np.float64)
    fig, axes = plt.subplots(
        len(eccentricities),
        2,
        figsize=(7.0, 5.6),
        sharex="col",
        constrained_layout=True,
    )
    zoom_cycles = 8.0
    zoom_duration = zoom_cycles * (2.0 / f0_hz)
    zoom = times <= zoom_duration

    curves = []
    for eccentricity in eccentricities:
        strain = eccentric_complex_strain(
            source(eccentricity, f0_hz),
            times,
            physics_mode="newtonian",
            evolution_mode="fixed",
        )
        h_plus = np.real(strain)
        scale = np.max(np.abs(h_plus[zoom]))
        relative_power = harmonic_relative_power(h_plus, dt, f0_hz, harmonics)
        contributing = int(np.count_nonzero(relative_power >= 0.1))
        curves.append((eccentricity, h_plus / scale, relative_power, contributing))

    for index, (eccentricity, h_plus_scaled, relative_power, contributing) in enumerate(curves):
        time_ax = axes[index, 0]
        power_ax = axes[index, 1]
        time_ax.plot(
            times[zoom] / 3600.0,
            h_plus_scaled[zoom],
            color=colors[index],
            linewidth=1.8,
        )
        floor = 1.0e-8
        power_ax.bar(
            harmonic_positions,
            np.maximum(relative_power, floor),
            width=0.08,
            bottom=floor,
            color=colors[index],
            edgecolor=colors[index],
            linewidth=0.6,
            alpha=0.65,
        )
        power_ax.axhline(0.1, color="0.2", linestyle="--", linewidth=0.9, alpha=0.8)
        power_ax.text(
            0.98,
            0.97,
            rf"$N_{{\geq 10\%}}={contributing}$",
            transform=power_ax.transAxes,
            ha="right",
            va="top",
            fontsize=ANNOTATION_SIZE,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        for harmonic in harmonics:
            power_ax.axvline(0.5 * harmonic, color="0.75", linewidth=0.55, alpha=0.35)
        time_ax.set_ylim(-1.15, 1.15)
        time_ax.set_ylabel(rf"$h_+$, $e={eccentricity:g}$")
        time_ax.grid(True, alpha=0.3)
        power_ax.set_yscale("log")
        power_ax.set_xlim(0.25, 6.25)
        power_ax.set_ylim(floor, 2.0)
        power_ax.set_ylabel("relative power")
        power_ax.grid(True, which="both", alpha=0.3)

    axes[-1, 0].set_xlabel("time [hours]")
    axes[-1, 1].set_xlabel("$f/f_0$")
    axes[0, 0].set_title("time domain", fontsize=TITLE_SIZE)
    axes[0, 1].set_title("harmonic power", fontsize=TITLE_SIZE)

    fig.suptitle("fixed Newtonian eccentric source", fontsize=TITLE_SIZE)
    apply_tick_style(axes)

    path = out_dir / "paper_time_waveform_then_fft_e0_e01_e03.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    target = PAPER_FIGURE_DIR / path.name
    shutil.copy2(path, target)
    print(f"wrote {path}")
    print(f"copied {target}")


if __name__ == "__main__":
    main()
