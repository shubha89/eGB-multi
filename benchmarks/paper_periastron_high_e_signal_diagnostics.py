"""High-eccentricity signal diagnostics for 1PN periastron advance."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain, eccentric_dynamics  # noqa: E402


PAPER_FIGURE_DIR = Path(__file__).resolve().parents[1] / "reports" / "Effect_of_eccentricity_for_GB_in_LISA" / "figures"


def source() -> EccentricBinaryParams:
    return EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.8,
        m1_solar=2.0,
        m2_solar=2.0,
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


def make_waveform_window_plot(
    t: np.ndarray,
    no_periastron: np.ndarray,
    with_periastron: np.ndarray,
    *,
    dt: float,
    out_dir: Path,
) -> Path:
    windows = [(0.0, 1.0), (20.0, 21.0), (99.0, 100.0), (199.0, 200.0)]
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 8.4), sharey=True, constrained_layout=True)

    for ax, (start_day, stop_day) in zip(axes, windows, strict=True):
        start = int(round(start_day * 86400.0 / dt))
        stop = int(round(stop_day * 86400.0 / dt))
        hours = (t[start:stop] - start_day * 86400.0) / 3600.0
        scale = np.max(np.abs(np.real(with_periastron[start:stop])))

        ax.plot(hours, np.real(with_periastron[start:stop]) / scale, color="#0072B2", linewidth=1.05, label="1PN+periastron")
        ax.plot(hours, np.real(no_periastron[start:stop]) / scale, color="#D55E00", linewidth=0.95, alpha=0.82, label="1PN no periastron")
        ax.text(0.015, 0.82, f"{start_day:g}-{stop_day:g} d", transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.set_xlim(0.0, 24.0)
        ax.set_ylabel("normalized Re[h]")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time in window [hours]")
    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    fig.suptitle(r"$e=0.8$, $2M_\odot+2M_\odot$, $f_0=1\,\mathrm{mHz}$, fixed source", fontsize=10)

    path = out_dir / "paper_periastron_waveform_windows_1pn_vs_1pn_peri_e08_m2_m2_200days.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def make_envelope_ratio_plot(
    t: np.ndarray,
    no_periastron: np.ndarray,
    with_periastron: np.ndarray,
    *,
    dt: float,
    out_dir: Path,
) -> tuple[Path, Path]:
    days = t / 86400.0
    envelope_window_days = 0.5
    samples_per_window = int(round(envelope_window_days * 86400.0 / dt))
    env_no_periastron = sliding_rms_envelope(no_periastron, samples_per_window)
    env_with_periastron = sliding_rms_envelope(with_periastron, samples_per_window)
    edge = samples_per_window
    valid = slice(edge, -edge)
    envelope_ratio = env_with_periastron / env_no_periastron

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.7), constrained_layout=True)

    zoom_days = 3.0
    zoom_stop = int(round(zoom_days * 86400.0 / dt))
    zoom_hours = days[:zoom_stop] * 24.0
    waveform_scale = np.max(np.abs(np.real(with_periastron[:zoom_stop])))
    axes[0].plot(
        zoom_hours,
        np.real(with_periastron[:zoom_stop]) / waveform_scale,
        color="0.35",
        linewidth=0.55,
        alpha=0.72,
        label=r"Re[$h_{\rm 1PN+peri}$]",
    )
    axes[0].plot(zoom_hours, env_with_periastron[:zoom_stop] / waveform_scale, color="#0072B2", linewidth=1.6, label=r"$+A_{\rm env}^{\rm RMS}$")
    axes[0].plot(zoom_hours, -env_with_periastron[:zoom_stop] / waveform_scale, color="#0072B2", linewidth=1.6, label=r"$-A_{\rm env}^{\rm RMS}$")
    axes[0].axvspan(30.0, 42.0, color="#D55E00", alpha=0.14, label="0.5-day RMS window")
    axes[0].axhline(0.0, color="black", linewidth=0.7, alpha=0.55)
    axes[0].set_xlim(0.0, zoom_days * 24.0)
    axes[0].set_xlabel("time in example window [hours]")
    axes[0].set_ylabel("normalized waveform")
    axes[0].set_title(r"Envelope construction: $A_{\rm env}^{\rm RMS}(t)=\sqrt{\langle {\rm Re}[h]^2\rangle_{0.5\,{\rm d}}}$")
    axes[0].legend(fontsize=8, ncol=2, loc="lower right")

    stride = max(1, int(round(0.05 * 86400.0 / dt)))
    axes[1].plot(days[::stride], envelope_ratio[::stride], color="#009E73", linewidth=1.5)
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("time [days]")
    axes[1].set_ylabel(
        r"$A_{\rm env}^{\rm RMS}({\rm 1PN+peri})$"
        "\n"
        r"$/$ $A_{\rm env}^{\rm RMS}({\rm 1PN\,no\,peri})$"
    )
    axes[1].set_title("Sliding RMS amplitude-envelope ratio")
    axes[1].set_ylim(
        np.percentile(envelope_ratio[valid], 0.5) * 0.995,
        np.percentile(envelope_ratio[valid], 99.5) * 1.005,
    )

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.suptitle(r"$e=0.8$, $2M_\odot+2M_\odot$, $f_0=1\,\mathrm{mHz}$, fixed source, 200 days", fontsize=10)
    png_path = out_dir / "paper_periastron_amplitude_envelope_ratio_1pn_vs_1pn_peri_e08_m2_m2_200days.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    csv_path = out_dir / "paper_periastron_amplitude_envelope_ratio_1pn_vs_1pn_peri_e08_m2_m2_200days.csv"
    sample_indices = np.linspace(0, t.size - 1, 1001, dtype=int)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_days", "envelope_ratio"])
        writer.writeheader()
        for index in sample_indices:
            writer.writerow({"time_days": float(days[index]), "envelope_ratio": float(envelope_ratio[index])})

    print(f"ratio_min={np.min(envelope_ratio[valid]):.6f}")
    print(f"ratio_max={np.max(envelope_ratio[valid]):.6f}")
    print(f"ratio_std={np.std(envelope_ratio[valid]):.6f}")
    return png_path, csv_path


def make_phase_plot(src: EccentricBinaryParams, t: np.ndarray, *, out_dir: Path) -> Path:
    days = t / 86400.0
    dynamics = eccentric_dynamics(src, t, physics_mode="1pn+periastron", evolution_mode="fixed")
    mean_anomaly = np.asarray(dynamics["mean_anomaly"], dtype=np.float64)
    lambda_phase = np.asarray(dynamics["azimuthal_frequency"], dtype=np.float64) * (t - src.t0) + src.phi0
    advance_cycles = (lambda_phase - mean_anomaly - src.phi0) / (2.0 * np.pi)

    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    ax.plot(days, advance_cycles, color="#0072B2", linewidth=1.8)
    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"$\Delta\varpi/2\pi$ [cycles]")
    ax.set_title("Accumulated 1PN periastron advance")
    ax.grid(True, alpha=0.3)
    fig.suptitle(r"$e=0.8$, $2M_\odot+2M_\odot$, $f_0=1\,\mathrm{mHz}$, fixed source", fontsize=10)
    path = out_dir / "paper_periastron_phase_residual_e08_m2_m2_200days.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"advance_cycles_200d={advance_cycles[-1]:.6f}")
    return path


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    src = source()
    duration_days = 200.0
    dt = 20.0
    t = np.arange(0.0, duration_days * 86400.0, dt)
    no_periastron = eccentric_complex_strain(src, t, physics_mode="1pn_no_periastron", evolution_mode="fixed")
    with_periastron = eccentric_complex_strain(src, t, physics_mode="1pn+periastron", evolution_mode="fixed")

    paths = [
        make_waveform_window_plot(t, no_periastron, with_periastron, dt=dt, out_dir=out_dir),
        make_envelope_ratio_plot(t, no_periastron, with_periastron, dt=dt, out_dir=out_dir)[0],
        make_phase_plot(src, t, out_dir=out_dir),
    ]
    residual = no_periastron - with_periastron
    print(f"cumulative_residual_200d={np.linalg.norm(residual) / np.linalg.norm(with_periastron):.6f}")

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = PAPER_FIGURE_DIR / path.name
        shutil.copy2(path, target)
        print(f"copied {target}")


if __name__ == "__main__":
    main()
