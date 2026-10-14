"""Population residual scan: brute-force exact links vs FastGB and wFGB.

This extends ``wfgb_fastgb_bruteforce_residuals.py`` from three fixed sources
to a small population with different frequencies, sky locations, orientations,
polarizations and phases.

Run with:

    python benchmarks/wfgb_fastgb_population_residuals.py

Outputs:

    benchmarks/wfgb_fastgb_population_residuals.csv
    benchmarks/wfgb_fastgb_population_residuals.png
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


def make_population(n_sources: int, seed: int = 20260715) -> list[GalacticBinaryParams]:
    rng = np.random.default_rng(seed)
    frequencies = np.geomspace(1.0e-4, 3.0e-3, n_sources)
    rng.shuffle(frequencies)
    return [
        GalacticBinaryParams(
            f0=float(f0),
            fdot=0.0,
            amplitude=1.0e-22,
            beta=float(rng.uniform(-0.9, 0.9)),
            lambda_=float(rng.uniform(-np.pi, np.pi)),
            psi=float(rng.uniform(0.0, np.pi)),
            inclination=float(rng.uniform(0.0, np.pi)),
            phi0=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        for f0 in frequencies
    ]


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


def combined_xyz_residual(model: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> float:
    numerator = 0.0
    denominator = 0.0
    for channel in ("X", "Y", "Z"):
        model_channel = np.asarray(model[channel], dtype=np.complex128)
        reference_channel = np.asarray(reference[channel], dtype=np.complex128)
        scale = best_fit_scale(model_channel, reference_channel)
        residual = scale * model_channel - reference_channel
        numerator += np.linalg.norm(residual) ** 2
        denominator += np.linalg.norm(reference_channel) ** 2
    return float(np.sqrt(numerator / denominator))


def fastgb_slices(result) -> tuple[dict[str, np.ndarray], int, int]:
    kmin = int(result.kmin[0])
    return (
        {
            "X": result.x[0],
            "Y": result.y[0],
            "Z": result.z[0],
        },
        kmin,
        result.x.shape[-1],
    )


def fft_slices_from_xyz(xyz: dict[str, np.ndarray], *, kmin: int, bins: int) -> dict[str, np.ndarray]:
    return {channel: fft_slice(xyz[channel], kmin=kmin, bins=bins) for channel in ("X", "Y", "Z")}


def binned_medians(rows: list[dict[str, float]], key: str, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([row["f0_hz"] for row in rows], dtype=np.float64)
    y = np.array([row[key] for row in rows], dtype=np.float64)
    centers = np.sqrt(bins[:-1] * bins[1:])
    medians = np.full(centers.shape, np.nan)
    for index, (lo, hi) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
        mask = (x >= lo) & (x < hi if index < centers.size - 1 else x <= hi)
        if np.any(mask):
            medians[index] = float(np.median(y[mask]))
    return centers, medians


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    n_sources = 36
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
    sources = make_population(n_sources)

    warm = sources[0]
    compute_xyz(state, warm, generation=1, measurement_order=5, delay_order=5)
    fastgb.frequency_domain_xyz(warm)
    synthesize_wfgb(warm, wfgb_config, orbits=orbits)

    rows = []
    for index, source in enumerate(sources, start=1):
        brute_xyz, brute_s = elapsed(lambda src=source: compute_xyz(state, src, generation=1, measurement_order=5, delay_order=5))
        fastgb_result, fastgb_s = elapsed(lambda src=source: fastgb.frequency_domain_xyz(src))
        wfgb_result, wfgb_s = elapsed(lambda src=source: synthesize_wfgb(src, wfgb_config, orbits=orbits))

        fastgb_xyz, kmin, bins = fastgb_slices(fastgb_result)
        reference = fft_slices_from_xyz(brute_xyz, kmin=kmin, bins=bins)
        wfgb_xyz = fft_slices_from_xyz(wfgb_result.xyz, kmin=kmin, bins=bins)

        fastgb_residual = combined_xyz_residual(fastgb_xyz, reference)
        wfgb_residual = combined_xyz_residual(wfgb_xyz, reference)
        row = {
            "source_index": index,
            "f0_hz": source.f0,
            "beta": source.beta,
            "lambda": source.lambda_,
            "psi": source.psi,
            "inclination": source.inclination,
            "phi0": source.phi0,
            "comparison_kmin": kmin,
            "comparison_bins": bins,
            "fastgb_residual_xyz": fastgb_residual,
            "wfgb_residual_xyz": wfgb_residual,
            "residual_ratio_fastgb_over_wfgb": fastgb_residual / wfgb_residual,
            "bruteforce_xyz_s": brute_s,
            "fastgb_xyz_s": fastgb_s,
            "wfgb_xyz_s": wfgb_s,
        }
        rows.append(row)
        print(
            f"{index:02d}/{n_sources} f0={source.f0:.3e} | "
            f"FastGB {fastgb_residual:.3e} | wFGB {wfgb_residual:.3e} | "
            f"ratio {fastgb_residual / wfgb_residual:.1f}"
        )

    csv_path = out_dir / "wfgb_fastgb_population_residuals.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    frequencies = np.array([row["f0_hz"] for row in rows])
    fastgb_residuals = np.array([row["fastgb_residual_xyz"] for row in rows])
    wfgb_residuals = np.array([row["wfgb_residual_xyz"] for row in rows])
    bins = np.geomspace(np.min(frequencies), np.max(frequencies), 7)
    med_x, fastgb_med = binned_medians(rows, "fastgb_residual_xyz", bins)
    _, wfgb_med = binned_medians(rows, "wfgb_residual_xyz", bins)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    axes[0].scatter(frequencies, fastgb_residuals, s=32, alpha=0.65, label="FastGB")
    axes[0].scatter(frequencies, wfgb_residuals, s=32, alpha=0.65, label="wFGB")
    axes[0].plot(med_x, fastgb_med, marker="o", linewidth=2.2, label="FastGB median")
    axes[0].plot(med_x, wfgb_med, marker="s", linewidth=2.2, label="wFGB median")
    axes[1].scatter(
        frequencies,
        [row["residual_ratio_fastgb_over_wfgb"] for row in rows],
        s=34,
        color="#009E73",
        alpha=0.75,
    )
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("relative XYZ residual vs brute force")
    axes[1].set_ylabel("FastGB residual / wFGB residual")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("source frequency [Hz]")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title("population residual scan")
    axes[1].set_title("relative improvement")

    png_path = out_dir / "wfgb_fastgb_population_residuals.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"median FastGB residual: {np.median(fastgb_residuals):.3e}")
    print(f"median wFGB residual:   {np.median(wfgb_residuals):.3e}")
    print(f"median ratio:           {np.median(fastgb_residuals / wfgb_residuals):.2f}")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
