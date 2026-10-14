"""Periastron-advance phase residual diagnostic for the paper."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import EccentricBinaryParams, eccentric_dynamics  # noqa: E402


PAPER_FIGURE_DIR = Path(__file__).resolve().parents[1] / "reports" / "Effect_of_eccentricity_for_GB_in_LISA" / "figures"
PHYSICS_MODE = "1pn+periastron"


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


def periastron_crossings(
    days: np.ndarray,
    mean_anomaly: np.ndarray,
    lambda_phase: np.ndarray,
    *,
    phi0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_cycle = int(np.ceil(mean_anomaly[0] / (2.0 * np.pi)))
    last_cycle = int(np.floor(mean_anomaly[-1] / (2.0 * np.pi)))
    cycle_numbers = np.arange(first_cycle, last_cycle + 1, dtype=np.float64)
    targets = cycle_numbers * 2.0 * np.pi
    crossing_days = np.interp(targets, mean_anomaly, days)
    crossing_lambda = np.interp(targets, mean_anomaly, lambda_phase)
    crossing_advance = (crossing_lambda - targets - phi0) / (2.0 * np.pi)
    return crossing_days, cycle_numbers, crossing_advance


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    src = source()
    duration_days = 100.0
    dt = 20.0
    t = np.arange(0.0, duration_days * 86400.0, dt)
    days = t / 86400.0
    dynamics = eccentric_dynamics(src, t, physics_mode=PHYSICS_MODE, evolution_mode="fixed")

    mean_anomaly = np.asarray(dynamics["mean_anomaly"], dtype=np.float64)
    radial_frequency = np.asarray(dynamics["radial_mean_motion"], dtype=np.float64)
    azimuthal_frequency = np.asarray(dynamics["azimuthal_frequency"], dtype=np.float64)
    lambda_phase = azimuthal_frequency * (t - src.t0) + src.phi0
    advance_phase = lambda_phase - mean_anomaly - src.phi0
    advance_cycles = advance_phase / (2.0 * np.pi)
    radial_cycles = (mean_anomaly - mean_anomaly[0]) / (2.0 * np.pi)

    periastron_days, periastron_cycles, periastron_advance = periastron_crossings(days, mean_anomaly, lambda_phase, phi0=src.phi0)
    marker_stride = max(1, periastron_days.size // 45)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), constrained_layout=True)

    axes[0].plot(days, advance_cycles, color="#0072B2", linewidth=1.8, label=r"$\lambda-l-\phi_0$")
    axes[0].scatter(
        periastron_days[::marker_stride],
        periastron_advance[::marker_stride],
        color="#D55E00",
        s=16,
        zorder=3,
        label="interpolated periastron passages",
    )
    axes[0].set_xlabel("time [days]")
    axes[0].set_ylabel(r"$\Delta\varpi / 2\pi$ [cycles]")
    axes[0].set_title("Accumulated 1PN periastron advance")
    axes[0].legend(fontsize=8)

    axes[1].plot(radial_cycles, advance_cycles, color="#009E73", linewidth=1.6, label=r"$\lambda-l-\phi_0$")
    axes[1].scatter(
        periastron_cycles[::marker_stride],
        periastron_advance[::marker_stride],
        color="#D55E00",
        s=16,
        zorder=3,
        label="interpolated periastron passages",
    )
    axes[1].set_xlabel("radial cycles")
    axes[1].set_ylabel("advance [cycles]")
    axes[1].set_title("Same residual versus radial orbit count")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.suptitle(r"$e=0.5$, $1.4M_\odot+1.4M_\odot$, $f_0=1\,\mathrm{mHz}$, fixed 1PN+periastron", fontsize=10)
    png_path = out_dir / "paper_periastron_phase_residual_e05_m14_m14_100days.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    csv_path = out_dir / "paper_periastron_phase_residual_e05_m14_m14_100days.csv"
    sample_indices = np.linspace(0, t.size - 1, 401, dtype=int)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time_days",
                "radial_cycles",
                "advance_cycles",
                "advance_radians",
                "radial_frequency_hz",
                "azimuthal_frequency_hz",
            ],
        )
        writer.writeheader()
        for index in sample_indices:
            writer.writerow(
                {
                    "time_days": float(days[index]),
                    "radial_cycles": float(radial_cycles[index]),
                    "advance_cycles": float(advance_cycles[index]),
                    "advance_radians": float(advance_phase[index]),
                    "radial_frequency_hz": float(radial_frequency[index] / (2.0 * np.pi)),
                    "azimuthal_frequency_hz": float(azimuthal_frequency[index] / (2.0 * np.pi)),
                }
            )

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    target = PAPER_FIGURE_DIR / png_path.name
    shutil.copy2(png_path, target)
    print(f"advance_cycles_100d={advance_cycles[-1]:.6f}")
    print(f"radial_cycles_100d={radial_cycles[-1]:.3f}")
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    print(f"copied {target}")


if __name__ == "__main__":
    main()
