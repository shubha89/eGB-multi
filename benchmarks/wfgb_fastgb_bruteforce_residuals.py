"""Accuracy comparison: brute-force exact links vs FastGB and wFGB.

This is a deliberately small diagnostic, not a production validation suite.
For each single circular Galactic binary:

1. compute exact time-domain one-way links and pyTDI Michelson X/Y/Z;
2. FFT the brute-force X/Y/Z reference;
3. compute installed FastGB frequency-domain X/Y/Z;
4. compute the standalone wFGB reference X/Y/Z and FFT it;
5. compare all methods on the same narrow Fourier-bin support.

The plotted residuals use one best-fit complex normalization per channel. This
removes global amplitude/phase convention differences and emphasizes waveform
shape differences on the common Fourier-bin support.

Run with:

    python benchmarks/wfgb_fastgb_bruteforce_residuals.py

Outputs:

    benchmarks/wfgb_fastgb_bruteforce_residuals.csv
    benchmarks/wfgb_fastgb_bruteforce_residuals.png
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.external import FastGBBackend, default_lisaorbits, state_from_lisaorbits
from egb_multi.sources import GalacticBinaryParams
from egb_multi.tdi import compute_xyz
from egb_multi.wfgb import WFGBConfig, synthesize_wfgb


def make_source(f0: float) -> GalacticBinaryParams:
    return GalacticBinaryParams(
        f0=f0,
        fdot=0.0,
        amplitude=1.0e-22,
        beta=0.3,
        lambda_=-0.7,
        psi=0.4,
        inclination=1.0,
        phi0=0.2,
    )


def elapsed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def fft_slice(series: np.ndarray, *, kmin: int, bins: int) -> np.ndarray:
    spectrum = np.fft.fft(np.asarray(series, dtype=np.complex128))
    indices = np.arange(kmin, kmin + bins, dtype=np.int64) % spectrum.size
    return spectrum[indices]


def best_fit_scale(model: np.ndarray, reference: np.ndarray) -> complex:
    denom = np.vdot(model, model)
    return 0.0j if denom == 0.0 else complex(np.vdot(model, reference) / denom)


def combined_xyz_residual(
    model: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    scaled: bool,
) -> tuple[float, dict[str, complex]]:
    numerator = 0.0
    denominator = 0.0
    scales: dict[str, complex] = {}
    for channel in ("X", "Y", "Z"):
        model_channel = np.asarray(model[channel], dtype=np.complex128)
        reference_channel = np.asarray(reference[channel], dtype=np.complex128)
        scale = best_fit_scale(model_channel, reference_channel) if scaled else 1.0 + 0.0j
        scales[channel] = scale
        residual = scale * model_channel - reference_channel
        numerator += np.linalg.norm(residual) ** 2
        denominator += np.linalg.norm(reference_channel) ** 2
    return float(np.sqrt(numerator / denominator)), scales


def fastgb_slices(result, *, bins: int | None = None) -> tuple[dict[str, np.ndarray], int, int]:
    kmin = int(result.kmin[0])
    count = result.x.shape[-1] if bins is None else min(bins, result.x.shape[-1])
    return (
        {
            "X": result.x[0, :count],
            "Y": result.y[0, :count],
            "Z": result.z[0, :count],
        },
        kmin,
        count,
    )


def fft_slices_from_xyz(xyz: dict[str, np.ndarray], *, kmin: int, bins: int) -> dict[str, np.ndarray]:
    return {channel: fft_slice(xyz[channel], kmin=kmin, bins=bins) for channel in ("X", "Y", "Z")}


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    frequencies = [1.0e-4, 3.0e-4, 1.0e-3]
    duration = 7.0 * 24.0 * 3600.0
    samples = 4096
    dt = duration / samples
    fastgb_bins = 64
    orbits = default_lisaorbits("equal")
    times = np.linspace(0.0, duration, samples, endpoint=False)
    state = state_from_lisaorbits(orbits, times)
    fastgb = FastGBBackend(delta_t=dt, duration=duration, n_samples=fastgb_bins, orbits=orbits)
    wfgb_config = WFGBConfig(
        duration=duration,
        dt=dt,
        buffer_duration=2.0 * duration,
        window_a=0.36,
        window_power=4,
        oversampling_power=3,
        min_slow_samples=1024,
        db=1,
        use_time_varying_delays=True,
    )

    warm_source = make_source(frequencies[0])
    compute_xyz(state, warm_source, generation=1, measurement_order=5, delay_order=5)
    fastgb.frequency_domain_xyz(warm_source)
    synthesize_wfgb(warm_source, wfgb_config, orbits=orbits)

    rows = []
    for f0 in frequencies:
        source = make_source(f0)
        brute_xyz, brute_s = elapsed(lambda: compute_xyz(state, source, generation=1, measurement_order=5, delay_order=5))
        fastgb_result, fastgb_s = elapsed(lambda: fastgb.frequency_domain_xyz(source))
        wfgb_result, wfgb_s = elapsed(lambda: synthesize_wfgb(source, wfgb_config, orbits=orbits))

        fastgb_xyz, kmin, bins = fastgb_slices(fastgb_result)
        brute_bins = fft_slices_from_xyz(brute_xyz, kmin=kmin, bins=bins)
        wfgb_bins = fft_slices_from_xyz(wfgb_result.xyz, kmin=kmin, bins=bins)

        fastgb_raw, _ = combined_xyz_residual(fastgb_xyz, brute_bins, scaled=False)
        fastgb_scaled, fastgb_scales = combined_xyz_residual(fastgb_xyz, brute_bins, scaled=True)
        wfgb_raw, _ = combined_xyz_residual(wfgb_bins, brute_bins, scaled=False)
        wfgb_scaled, wfgb_scales = combined_xyz_residual(wfgb_bins, brute_bins, scaled=True)

        row = {
            "f0_hz": f0,
            "duration_days": duration / 86400.0,
            "samples": samples,
            "dt_s": dt,
            "comparison_kmin": kmin,
            "comparison_bins": bins,
            "wfgb_groups": len(wfgb_result.groups),
            "bruteforce_xyz_s": brute_s,
            "fastgb_xyz_s": fastgb_s,
            "wfgb_xyz_s": wfgb_s,
            "fastgb_raw_residual_xyz": fastgb_raw,
            "fastgb_scaled_residual_xyz": fastgb_scaled,
            "wfgb_raw_residual_xyz": wfgb_raw,
            "wfgb_scaled_residual_xyz": wfgb_scaled,
            "fastgb_scale_abs_X": abs(fastgb_scales["X"]),
            "fastgb_scale_phase_X": float(np.angle(fastgb_scales["X"])),
            "wfgb_scale_abs_X": abs(wfgb_scales["X"]),
            "wfgb_scale_phase_X": float(np.angle(wfgb_scales["X"])),
        }
        rows.append(row)
        print(
            f"f0={f0:.1e} Hz | FastGB scaled residual {fastgb_scaled:.3e} | "
            f"wFGB scaled residual {wfgb_scaled:.3e} | "
            f"times brute/FastGB/wFGB {brute_s:.3f}/{fastgb_s:.3f}/{wfgb_s:.3f}s"
        )

    csv_path = out_dir / "wfgb_fastgb_bruteforce_residuals.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["f0_hz"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), constrained_layout=True)
    axes[0].plot(x, [row["fastgb_scaled_residual_xyz"] for row in rows], marker="o", label="FastGB")
    axes[0].plot(x, [row["wfgb_scaled_residual_xyz"] for row in rows], marker="s", label="wFGB")
    axes[1].plot(x, [row["bruteforce_xyz_s"] for row in rows], marker="D", label="brute-force links + pyTDI")
    axes[1].plot(x, [row["fastgb_xyz_s"] for row in rows], marker="o", label="FastGB")
    axes[1].plot(x, [row["wfgb_xyz_s"] for row in rows], marker="s", label="wFGB reference")
    axes[0].set_ylabel("relative XYZ residual vs brute force")
    axes[1].set_ylabel("wall time [s]")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("source frequency [Hz]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("accuracy on common frequency support")
    axes[1].set_title("limited-test wall times")

    png_path = out_dir / "wfgb_fastgb_bruteforce_residuals.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
