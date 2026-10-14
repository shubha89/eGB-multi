"""Waveform-window comparison for 1PN with and without periastron advance."""

from __future__ import annotations

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


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    src = source()
    dt = 20.0
    duration_days = 100.0
    t = np.arange(0.0, duration_days * 86400.0, dt)

    no_periastron = eccentric_complex_strain(src, t, physics_mode="1pn_no_periastron", evolution_mode="fixed")
    with_periastron = eccentric_complex_strain(src, t, physics_mode="1pn+periastron", evolution_mode="fixed")

    windows = [(0.0, 1.0), (20.0, 21.0), (99.0, 100.0)]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.8), sharey=True, constrained_layout=True)

    for ax, (start_day, stop_day) in zip(axes, windows, strict=True):
        start = int(round(start_day * 86400.0 / dt))
        stop = int(round(stop_day * 86400.0 / dt))
        hours = (t[start:stop] - start_day * 86400.0) / 3600.0
        scale = np.max(np.abs(np.real(with_periastron[start:stop])))

        ax.plot(hours, np.real(with_periastron[start:stop]) / scale, color="#0072B2", linewidth=1.1, label="1PN+periastron")
        ax.plot(hours, np.real(no_periastron[start:stop]) / scale, color="#D55E00", linewidth=1.0, alpha=0.82, label="1PN no periastron")
        ax.text(0.015, 0.82, f"{start_day:g}-{stop_day:g} d", transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.set_xlim(0.0, 24.0)
        ax.set_ylabel("normalized Re[h]")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time in window [hours]")
    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    fig.suptitle(r"$e=0.5$, $1.4M_\odot+1.4M_\odot$, $f_0=1\,\mathrm{mHz}$, fixed source", fontsize=10)

    png_path = out_dir / "paper_periastron_waveform_windows_1pn_vs_1pn_peri_e05_m14_m14.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    target = PAPER_FIGURE_DIR / png_path.name
    shutil.copy2(png_path, target)
    print(f"wrote {png_path}")
    print(f"copied {target}")


if __name__ == "__main__":
    main()
