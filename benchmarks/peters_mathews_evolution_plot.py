from __future__ import annotations

import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain, peters_mathews_evolution


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


def main() -> None:
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.3,
        m1_solar=0.6,
        m2_solar=0.4,
        inclination=0.8,
        phi0=0.2,
    )
    t = np.linspace(0.0, 4.0 * SECONDS_PER_YEAR, 8192)
    comparison_duration = 3.0 * 24.0 * 3600.0
    comparison_t = np.arange(t[-1] - comparison_duration, t[-1], 20.0)

    start = time.perf_counter()
    fixed = eccentric_complex_strain(source, comparison_t, evolution_mode="fixed")
    fixed_wall = time.perf_counter() - start

    start = time.perf_counter()
    evolved = eccentric_complex_strain(source, comparison_t, evolution_mode="peters_mathews")
    evolved_wall = time.perf_counter() - start

    evolution = peters_mathews_evolution(source, t)
    residual = np.linalg.norm(evolved - fixed) / np.linalg.norm(fixed)

    print(f"fixed_wall_s={fixed_wall:.6f}")
    print(f"peters_mathews_wall_s={evolved_wall:.6f}")
    print(f"relative_waveform_residual={residual:.6e}")
    print(f"initial_n={evolution['mean_motion'][0]:.12e}")
    print(f"final_n={evolution['mean_motion'][-1]:.12e}")
    print(f"initial_e={evolution['eccentricity'][0]:.12e}")
    print(f"final_e={evolution['eccentricity'][-1]:.12e}")

    years = t / SECONDS_PER_YEAR
    comparison_days = (comparison_t - comparison_t[0]) / (24.0 * 3600.0)
    scale = np.max(np.abs(np.real(fixed)))
    fixed_real = np.real(fixed) / scale
    evolved_real = np.real(evolved) / scale
    difference_real = evolved_real - fixed_real

    fig, axes = plt.subplots(5, 1, figsize=(9, 10.5))

    axes[0].plot(years, evolution["mean_motion"] / np.pi, color="tab:blue")
    axes[0].set_ylabel("GW-like f0 [Hz]")
    axes[0].grid(alpha=0.25)

    axes[1].plot(years, evolution["eccentricity"], color="tab:orange")
    axes[1].set_ylabel("eccentricity")
    axes[1].grid(alpha=0.25)

    axes[2].plot(comparison_days, fixed_real, label="fixed", color="0.25", linewidth=0.8)
    axes[2].plot(comparison_days, evolved_real, label="Peters-Mathews", color="tab:green", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("overlay")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right")

    axes[3].plot(comparison_days, fixed_real, label="fixed", color="0.25", linewidth=0.8)
    axes[3].plot(comparison_days, evolved_real + 2.4, label="Peters-Mathews + offset", color="tab:green", linewidth=0.8)
    axes[3].set_ylabel("separated")
    axes[3].grid(alpha=0.25)
    axes[3].legend(loc="upper right")

    axes[4].plot(comparison_days, difference_real, color="tab:red", linewidth=0.8)
    axes[4].axhline(0.0, color="0.25", linewidth=0.8, alpha=0.6)
    axes[4].set_xlabel("time in final 3 day window [day]")
    axes[4].set_ylabel("PM - fixed")
    axes[4].grid(alpha=0.25)

    fig.suptitle(f"Peters-Mathews evolution, residual vs fixed = {residual:.3e}")
    fig.tight_layout()

    out = Path(__file__).with_name("peters_mathews_evolution.png")
    fig.savefig(out, dpi=180)
    print(f"plot={out}")


if __name__ == "__main__":
    main()
