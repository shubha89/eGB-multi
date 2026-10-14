"""Plot long eccentric waveform modulation from periastron advance.

Run with:

    python benchmarks/periastron_advance_modulation_plot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain


def normalized_rms_envelope(series: np.ndarray, samples_per_window: int) -> np.ndarray:
    window = np.ones(samples_per_window, dtype=np.float64) / samples_per_window
    power = np.convolve(np.real(series) ** 2, window, mode="same")
    envelope = np.sqrt(power)
    return envelope / np.median(envelope)


def analytic_phase(series: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(hilbert(np.real(series))))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    source = EccentricBinaryParams(
        mean_motion=np.pi * 2.0e-4,
        eccentricity=0.5,
        m1_solar=0.8,
        m2_solar=0.6,
        inclination=1.0,
        phi0=0.3,
    )
    duration_days = 4.0 * 365.25
    dt = 100.0
    t = np.arange(0.0, duration_days * 86400.0, dt)
    days = t / 86400.0

    no_precession = eccentric_complex_strain(source, t, physics_mode="1pn_no_periastron")
    full = eccentric_complex_strain(source, t, physics_mode="1pn")
    envelope_window = int(5.0 * 86400.0 / dt)
    env_no_precession = normalized_rms_envelope(no_precession, envelope_window)
    env_full = normalized_rms_envelope(full, envelope_window)
    env_ratio = env_full / env_no_precession
    phase_drift_cycles = (analytic_phase(full) - analytic_phase(no_precession)) / (2.0 * np.pi)
    phase_drift_cycles = phase_drift_cycles - phase_drift_cycles[0]

    edge = envelope_window
    valid = slice(edge, -edge)
    zoom_days = 20.0
    end_start = max(0, t.size - int(zoom_days * 86400.0 / dt))
    end_window = slice(end_start, t.size)
    stride = max(1, int(3600.0 / dt))
    fig, axes = plt.subplots(5, 1, figsize=(11.5, 11.0), sharex=False, constrained_layout=True)

    axes[0].plot(days[::stride], np.real(full[::stride]), linewidth=0.55, label="with periastron advance")
    axes[0].plot(days[::stride], np.real(no_precession[::stride]), linewidth=0.55, alpha=0.7, label="without periastron advance")
    axes[0].plot(days[::stride], env_full[::stride] * np.max(np.abs(np.real(full))), color="tab:blue", linewidth=0.9, alpha=0.85)
    axes[0].plot(days[::stride], -env_full[::stride] * np.max(np.abs(np.real(full))), color="tab:blue", linewidth=0.9, alpha=0.85)
    axes[0].set_ylabel("Re[h]")
    axes[0].set_title("4-year eccentric waveform, hourly samples with envelope guide")
    axes[0].legend(fontsize=8)

    axes[1].plot(days, env_full, linewidth=1.0, label="envelope with periastron advance")
    axes[1].plot(days, env_no_precession, linewidth=1.0, alpha=0.75, label="envelope without periastron advance")
    axes[1].set_ylabel("normalized envelope")
    axes[1].set_title("amplitude envelope")
    axes[1].legend(fontsize=8)

    axes[2].plot(days, env_ratio, color="tab:red", linewidth=1.0)
    axes[2].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_ylim(
        np.percentile(env_ratio[valid], 0.5) * 0.995,
        np.percentile(env_ratio[valid], 99.5) * 1.005,
    )
    axes[2].set_ylabel("envelope ratio")
    axes[2].set_title("periastron-advance amplitude modulation")

    axes[3].plot(days, phase_drift_cycles, color="tab:purple", linewidth=1.0)
    axes[3].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[3].set_ylabel("phase drift [cycles]")
    axes[3].set_title("accumulated carrier phase drift from periastron advance")

    shifted_end_days = days[end_window] - days[end_window][0]
    zoom_scale = np.max(np.abs(np.real(full[end_window])))
    axes[4].plot(shifted_end_days, np.real(full[end_window]) / zoom_scale, linewidth=0.8, label="with periastron advance")
    axes[4].plot(
        shifted_end_days,
        np.real(no_precession[end_window]) / zoom_scale,
        linewidth=0.8,
        alpha=0.7,
        label="without periastron advance",
    )
    axes[4].plot(
        shifted_end_days,
        (np.real(full[end_window]) - np.real(no_precession[end_window])) / zoom_scale,
        linewidth=0.8,
        color="tab:red",
        label="difference",
    )
    axes[4].set_ylabel("Re[h]")
    axes[4].set_xlabel("time in final 20-day window [days]")
    axes[4].set_title("final-window waveform and difference")
    axes[4].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].set_xlabel("time [days]")
    axes[1].set_xlabel("time [days]")
    axes[2].set_xlabel("time [days]")
    axes[3].set_xlabel("time [days]")

    png_path = out_dir / "periastron_advance_modulation.png"
    long_png_path = out_dir / "periastron_advance_long.png"
    fig.savefig(png_path, dpi=180)
    fig.savefig(long_png_path, dpi=180)
    plt.close(fig)

    print(f"samples={t.size}, duration_days={duration_days}, dt={dt}")
    print(f"envelope ratio min={np.min(env_ratio[valid]):.6f}, max={np.max(env_ratio[valid]):.6f}, std={np.std(env_ratio[valid]):.6f}")
    print(f"phase drift cycles min={np.min(phase_drift_cycles[valid]):.6f}, max={np.max(phase_drift_cycles[valid]):.6f}")
    print(f"wrote {png_path}")
    print(f"wrote {long_png_path}")


if __name__ == "__main__":
    main()
