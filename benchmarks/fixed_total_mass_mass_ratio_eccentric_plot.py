"""Plot eccentric waveforms for different mass ratios at fixed total mass.

Run with:

    python benchmarks/fixed_total_mass_mass_ratio_eccentric_plot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain


def envelope(series: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(np.real(series)))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    total_mass = 1.0
    mass_pairs = [(0.50, 0.50), (0.70, 0.30), (0.85, 0.15), (0.95, 0.05)]
    eccentricity = 0.2
    f0 = 1.0e-3
    duration_days = 30.0
    dt = 50.0
    t = np.arange(0.0, duration_days * 86400.0, dt)
    days = t / 86400.0

    waveforms = {}
    envelopes = {}
    for m1, m2 in mass_pairs:
        assert np.isclose(m1 + m2, total_mass)
        source = EccentricBinaryParams(
            mean_motion=np.pi * f0,
            eccentricity=eccentricity,
            m1_solar=m1,
            m2_solar=m2,
            inclination=1.0,
            phi0=0.3,
        )
        key = f"{m1:.2f}+{m2:.2f} Msun, eta={source.symmetric_mass_ratio:.3f}"
        h = eccentric_complex_strain(source, t, physics_mode="1pn")
        waveforms[key] = h
        envelopes[key] = envelope(h)

    reference_key = next(iter(waveforms))
    reference = waveforms[reference_key]
    reference_norm = np.linalg.norm(reference)

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.0), sharex=True, constrained_layout=True)
    for key, h in waveforms.items():
        axes[0].plot(days, np.real(h), linewidth=0.8, label=key)
        axes[1].plot(days, envelopes[key], linewidth=1.0, label=key)
        axes[2].plot(days, envelopes[key] / np.median(envelopes[key]), linewidth=1.0, label=key)
        if key != reference_key:
            scaled = h * (np.vdot(h, reference) / np.vdot(h, h))
            residual = np.real(scaled - reference) / np.max(np.abs(np.real(reference)))
            axes[3].plot(days, residual, linewidth=0.8, label=key)

    axes[0].set_ylabel("Re[h]")
    axes[0].set_title(f"eccentric source waveform, e={eccentricity}, M={total_mass} Msun, f0={f0:g} Hz")
    axes[1].set_ylabel("envelope")
    axes[1].set_title("absolute amplitude envelope")
    axes[2].set_ylabel("normalized envelope")
    axes[2].set_title("shape after removing median amplitude")
    axes[3].set_ylabel("scaled residual")
    axes[3].set_xlabel("time [days]")
    axes[3].set_title(f"residual after best complex scale vs {reference_key}")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    png_path = out_dir / "fixed_total_mass_mass_ratio_eccentric.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    zoom_days = 2.0
    zoom = days <= zoom_days
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.0), sharex=True, constrained_layout=True)
    for key, h in waveforms.items():
        scaled_to_equal = h * (np.vdot(h, reference) / np.vdot(h, h))
        axes[0].plot(days[zoom], np.real(h[zoom]), linewidth=1.0, label=key)
        axes[1].plot(days[zoom], np.real(scaled_to_equal[zoom]), linewidth=1.0, label=key)
        axes[2].plot(days[zoom], envelopes[key][zoom] / np.median(envelopes[key]), linewidth=1.0, label=key)
        if key != reference_key:
            residual = np.real(scaled_to_equal - reference) / np.max(np.abs(np.real(reference[zoom])))
            axes[3].plot(days[zoom], residual[zoom], linewidth=1.0, label=key)

    axes[0].set_ylabel("Re[h]")
    axes[0].set_title(f"zoomed raw waveform, first {zoom_days:g} days")
    axes[1].set_ylabel("scaled Re[h]")
    axes[1].set_title("after best complex scale to equal-mass waveform")
    axes[2].set_ylabel("norm. envelope")
    axes[2].set_title("normalized envelope shape")
    axes[3].set_ylabel("scaled residual")
    axes[3].set_xlabel("time [days]")
    axes[3].set_title("residual vs equal-mass waveform")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    zoom_path = out_dir / "fixed_total_mass_mass_ratio_eccentric_zoom.png"
    fig.savefig(zoom_path, dpi=180)
    plt.close(fig)

    print(f"duration_days={duration_days}, dt={dt}, samples={t.size}")
    for key, h in waveforms.items():
        amp_ratio = np.median(envelopes[key]) / np.median(envelopes[reference_key])
        scaled = h * (np.vdot(h, reference) / np.vdot(h, h))
        rel = np.linalg.norm(scaled - reference) / reference_norm
        print(f"{key}: median envelope ratio={amp_ratio:.4f}, scaled residual vs equal mass={rel:.3e}")
    print(f"wrote {png_path}")
    print(f"wrote {zoom_path}")


if __name__ == "__main__":
    main()
