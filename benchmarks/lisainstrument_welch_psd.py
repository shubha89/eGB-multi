"""Generate lisainstrument noise and estimate TDI PSDs with Welch.

Run with:

    python benchmarks/lisainstrument_welch_psd.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.noise import estimate_channel_psds, simulate_lisainstrument_tdi_noise


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    size = 8192
    dt = 10.0
    nperseg = 2048

    start = time.perf_counter()
    noise = simulate_lisainstrument_tdi_noise(size=size, dt=dt, seed=1234, order=5, delay_order=5)
    simulation_s = time.perf_counter() - start

    freq_xyz, psd_xyz = estimate_channel_psds(noise.xyz, noise.dt, nperseg=nperseg)
    freq_aet, psd_aet = estimate_channel_psds(noise.aet, noise.dt, nperseg=nperseg)
    if not np.array_equal(freq_xyz, freq_aet):
        raise RuntimeError("XYZ and AET Welch grids differ")

    rows = []
    for index, frequency in enumerate(freq_xyz):
        row = {"frequency_hz": frequency}
        for channel in ("X", "Y", "Z"):
            row[f"PSD_{channel}"] = psd_xyz[channel][index]
        for channel in ("A", "E", "T"):
            row[f"PSD_{channel}"] = psd_aet[channel][index]
        rows.append(row)

    csv_path = out_dir / "lisainstrument_welch_psd.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    positive = freq_xyz > 0.0
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    for channel in ("X", "Y", "Z"):
        axes[0].loglog(freq_xyz[positive], psd_xyz[channel][positive], label=channel)
    for channel in ("A", "E", "T"):
        axes[1].loglog(freq_aet[positive], psd_aet[channel][positive], label=channel)
    for ax in axes:
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("Welch PSD")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_title("pyTDI X/Y/Z")
    axes[1].set_title("A/E/T from X/Y/Z")
    fig.suptitle(f"lisainstrument noise PSD, size={size}, dt={dt:g}s, nperseg={nperseg}")
    png_path = out_dir / "lisainstrument_welch_psd.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"generated {size} samples at dt={dt:g}s in {simulation_s:.3f}s")
    print(f"Welch df={freq_xyz[1] - freq_xyz[0]:.6e} Hz, bins={freq_xyz.size}")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
