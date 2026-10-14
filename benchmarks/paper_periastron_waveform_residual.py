"""Time-series residual between 1PN and 1PN+periastron eccentric waveforms."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain  # noqa: E402


PAPER_FIGURE_DIR = Path(__file__).resolve().parents[1] / "reports" / "Effect_of_eccentricity_for_GB_in_LISA" / "figures"


def source() -> EccentricBinaryParams:
    return EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.5,
        m1_solar=1.4,
        m2_solar=1.4,
        inclination=0.8,
        beta=0.2,
        lambda_=-1.0,
        psi=0.4,
        phi0=0.2,
    )


def rms_ratio_by_window(
    residual: np.ndarray,
    reference: np.ndarray,
    samples_per_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows = residual.size // samples_per_window
    residual_trimmed = residual[: windows * samples_per_window].reshape(windows, samples_per_window)
    reference_trimmed = reference[: windows * samples_per_window].reshape(windows, samples_per_window)
    ratios = np.linalg.norm(residual_trimmed, axis=1) / np.linalg.norm(reference_trimmed, axis=1)
    cumulative = np.linalg.norm(residual_trimmed.reshape(windows, samples_per_window), axis=1)
    cumulative = np.sqrt(np.cumsum(cumulative**2))
    reference_cumulative = np.linalg.norm(reference_trimmed.reshape(windows, samples_per_window), axis=1)
    reference_cumulative = np.sqrt(np.cumsum(reference_cumulative**2))
    return ratios, cumulative / reference_cumulative


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    src = source()
    duration_days = 100.0
    dt = 20.0
    t = np.arange(0.0, duration_days * 86400.0, dt)
    days = t / 86400.0

    no_periastron = eccentric_complex_strain(src, t, physics_mode="1pn_no_periastron", evolution_mode="fixed")
    with_periastron = eccentric_complex_strain(src, t, physics_mode="1pn+periastron", evolution_mode="fixed")
    residual = no_periastron - with_periastron

    samples_per_window = int(round(5.0 * 86400.0 / dt))
    window_mid_days = np.arange(0.5 * 5.0, duration_days, 5.0)
    window_ratio, cumulative_ratio = rms_ratio_by_window(residual, with_periastron, samples_per_window)

    fig, axes = plt.subplots(4, 1, figsize=(8.2, 8.6), constrained_layout=True)

    axes[0].plot(window_mid_days, window_ratio, marker="o", markersize=3.5, linewidth=1.4, label="5-day RMS residual")
    axes[0].plot(window_mid_days, cumulative_ratio, marker="s", markersize=3.2, linewidth=1.4, label="cumulative RMS residual")
    axes[0].set_xlabel("time [days]")
    axes[0].set_ylabel("relative RMS residual")
    axes[0].set_title(r"Residual growth for $h_{\rm 1PN\,no\,peri}-h_{\rm 1PN+peri}$")
    axes[0].legend(fontsize=8)

    stride = max(1, int(round(1800.0 / dt)))
    scale = np.max(np.abs(np.real(with_periastron)))
    axes[1].plot(days[::stride], np.real(residual[::stride]) / scale, color="#D55E00", linewidth=0.9)
    axes[1].axhline(0.0, color="black", linewidth=0.7, linestyle="--")
    axes[1].set_xlabel("time [days]")
    axes[1].set_ylabel(r"Re[$\Delta h$] / max Re[$h$]")
    axes[1].set_title("Residual time series, sampled every 30 minutes")

    zoom_hours = 8.0
    zoom_samples = int(round(zoom_hours * 3600.0 / dt))
    early = slice(0, zoom_samples)
    late_start = t.size - zoom_samples
    late = slice(late_start, t.size)

    early_hours = (t[early] - t[early][0]) / 3600.0
    late_hours = (t[late] - t[late][0]) / 3600.0

    for ax, span, hours, title in [
        (axes[2], early, early_hours, "First 8 hours"),
        (axes[3], late, late_hours, "Final 8 hours"),
    ]:
        local_scale = np.max(np.abs(np.real(with_periastron[span])))
        ax.plot(hours, np.real(with_periastron[span]) / local_scale, color="#0072B2", linewidth=1.2, label="1PN+periastron")
        ax.plot(hours, np.real(no_periastron[span]) / local_scale, color="#009E73", linewidth=1.0, alpha=0.8, label="1PN no periastron")
        ax.plot(hours, np.real(residual[span]) / local_scale, color="#D55E00", linewidth=1.0, label="difference")
        ax.set_xlabel("time in window [hours]")
        ax.set_ylabel("normalized Re[h]")
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=3)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.suptitle(r"$e=0.5$, $1.4M_\odot+1.4M_\odot$, $f_0=1\,\mathrm{mHz}$, fixed source", fontsize=10)
    png_path = out_dir / "paper_periastron_waveform_residual_1pn_minus_1pn_peri_e05_m14_m14_100days.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    csv_path = out_dir / "paper_periastron_waveform_residual_1pn_minus_1pn_peri_e05_m14_m14_100days.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["window_mid_day", "window_rms_residual", "cumulative_rms_residual"],
        )
        writer.writeheader()
        for day, ratio, cumulative in zip(window_mid_days, window_ratio, cumulative_ratio, strict=True):
            writer.writerow(
                {
                    "window_mid_day": float(day),
                    "window_rms_residual": float(ratio),
                    "cumulative_rms_residual": float(cumulative),
                }
            )

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    target = PAPER_FIGURE_DIR / png_path.name
    shutil.copy2(png_path, target)
    print(f"final_cumulative_residual={cumulative_ratio[-1]:.6f}")
    print(f"first_window_residual={window_ratio[0]:.6f}")
    print(f"last_window_residual={window_ratio[-1]:.6f}")
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    print(f"copied {target}")


if __name__ == "__main__":
    main()
