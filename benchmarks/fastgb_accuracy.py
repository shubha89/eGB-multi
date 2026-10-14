"""Compare FastGB FD X/Y/Z to FFTs of exact-link pyTDI X/Y/Z.

This is a convention-checking benchmark, not yet a final validation test. It
reports both raw residuals and residuals after fitting one complex scale factor
per TDI channel.

Run with:

    python benchmarks/fastgb_accuracy.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.external import FastGBBackend, default_lisaorbits, state_from_lisaorbits
from egb_multi.sources import GalacticBinaryParams
from egb_multi.tdi import compute_xyz


def relative_norm(model: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(model - reference) / np.linalg.norm(reference))


def best_fit_scale(model: np.ndarray, reference: np.ndarray) -> complex:
    denom = np.vdot(model, model)
    if denom == 0:
        return 0.0j
    return complex(np.vdot(model, reference) / denom)


def select_fft_bins(series: np.ndarray, dt: float, kmin: int, n_bins: int) -> np.ndarray:
    spectrum = np.fft.fft(series) * dt
    if kmin + n_bins > spectrum.size:
        raise ValueError(
            f"requested bins [{kmin}, {kmin + n_bins}) exceed FFT size {spectrum.size}; "
            "decrease dt or the maximum comparison frequency"
        )
    return spectrum[kmin : kmin + n_bins]


def compare_one_frequency(f0: float, *, out_dir: Path) -> dict[str, float]:
    duration = 90.0 * 24.0 * 3600.0
    dt = 100.0
    n_time = int(duration / dt)
    duration = n_time * dt
    fastgb_bins = 128

    source = GalacticBinaryParams(
        f0=f0,
        fdot=0.0,
        amplitude=1.0e-22,
        beta=0.3,
        lambda_=-0.7,
        psi=0.4,
        inclination=1.0,
        phi0=0.2,
    )

    orbits = default_lisaorbits("equal")
    fastgb = FastGBBackend(delta_t=dt, duration=duration, n_samples=fastgb_bins, orbits=orbits)
    fast = fastgb.frequency_domain_xyz(source)
    kmin = int(fast.kmin[0])
    n_bins = fast.x.shape[-1]
    fast_xyz = {"X": fast.x[0], "Y": fast.y[0], "Z": fast.z[0]}

    times = np.arange(n_time, dtype=np.float64) * dt
    state = state_from_lisaorbits(orbits, times)
    exact_link_xyz = compute_xyz(state, source, generation=1, measurement_order=5, delay_order=5)

    rows = {"f0": f0, "kmin": kmin}
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 7.0), sharex="col", constrained_layout=True)
    freq = (np.arange(kmin, kmin + n_bins) / duration)

    for row, channel in enumerate(("X", "Y", "Z")):
        model = select_fft_bins(exact_link_xyz[channel], dt, kmin, n_bins)
        reference = fast_xyz[channel]
        alpha = best_fit_scale(model, reference)
        scaled_model = alpha * model
        raw = relative_norm(model, reference)
        scaled = relative_norm(scaled_model, reference)
        peak = np.max(np.abs(reference))
        residual = np.abs(reference - scaled_model) / peak

        rows[f"{channel}_raw_relerr"] = raw
        rows[f"{channel}_scaled_relerr"] = scaled
        rows[f"{channel}_scale_abs"] = abs(alpha)
        rows[f"{channel}_scale_phase"] = float(np.angle(alpha))
        rows[f"{channel}_max_peak_residual"] = float(np.max(residual))

        overlay_ax = axes[row, 0]
        residual_ax = axes[row, 1]
        overlay_ax.plot(freq, np.abs(reference), label="FastGB", linewidth=1.5)
        overlay_ax.plot(freq, np.abs(scaled_model), label="scaled exact-link pyTDI FFT", linewidth=1.0, linestyle="--")
        overlay_ax.set_ylabel(f"|{channel}(f)|")
        overlay_ax.grid(True, alpha=0.3)

        residual_ax.plot(freq, residual, color="tab:red", linewidth=1.0)
        residual_ax.set_ylabel(f"{channel} residual")
        residual_ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("frequency [Hz]")
    axes[-1, 1].set_xlabel("frequency [Hz]")
    axes[0, 0].legend()
    axes[0, 0].set_title("amplitude overlay")
    axes[0, 1].set_title("|FastGB - scaled exact-link| / peak(|FastGB|)")
    fig.suptitle(f"FastGB vs exact-link pyTDI FFT, f0={f0:.1e} Hz")
    plot_path = out_dir / f"fastgb_exact_link_accuracy_f0_{f0:.0e}.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    rows["plot"] = str(plot_path)
    return rows


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    frequencies = [1.0e-4, 3.0e-4, 1.0e-3]
    rows = [compare_one_frequency(f0, out_dir=out_dir) for f0 in frequencies]

    csv_path = out_dir / "fastgb_accuracy.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"f0={row['f0']:.1e} Hz, kmin={row['kmin']}")
        for channel in ("X", "Y", "Z"):
            print(
                f"  {channel}: raw={row[f'{channel}_raw_relerr']:.3e}, "
                f"scaled={row[f'{channel}_scaled_relerr']:.3e}, "
                f"max_peak_resid={row[f'{channel}_max_peak_residual']:.3e}, "
                f"|scale|={row[f'{channel}_scale_abs']:.3e}, "
                f"phase={row[f'{channel}_scale_phase']:.3f}"
            )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
