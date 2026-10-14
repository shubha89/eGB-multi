"""Eccentric correction to the Karnesis et al. 2021 CGB confusion model.

The absolute circular foreground is the analytic model from Karnesis et al.
2021. The eccentric correction is the Monte Carlo ratio:

    S_ecc(f) ~= R_ecc/circ(f) * S_Karnesis_circ(f)

Run with:

    python benchmarks/eccentric_corrected_karnesis_confusion.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from monte_carlo_circular_eccentric_confusion import (
    binned_mean,
    circular_to_eccentric_source,
    draw_population,
    fd_psd,
)

from egb_multi.external import FastGBBackend, default_lisaorbits
from egb_multi.fastgb_convention import (
    circular_component_from_fastgb_source,
    eccentric_components_for_fastgb_response,
    frequency_domain_xyz_from_components,
)
from egb_multi.noise import karnesis_2021_cgb_confusion_psd


def ae_mean_psd(result, duration: float) -> tuple[np.ndarray, np.ndarray]:
    df = 1.0 / duration
    freq = (int(result.kmin[0]) + np.arange(result.x.shape[-1], dtype=np.float64)) * df
    a = (result.z[0] - result.x[0]) / np.sqrt(2.0)
    e = (result.x[0] - 2.0 * result.y[0] + result.z[0]) / np.sqrt(6.0)
    return freq, 0.5 * (fd_psd(a, duration) + fd_psd(e, duration))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    n_sources = 3000
    eccentricities = [0.0, 0.03, 0.05, 0.1, 0.2, 0.3]
    eccentric_order = 2
    harmonic_samples = 2048
    duration = 365.25 * 24.0 * 3600.0
    t_obs_years = 4.0
    snr_threshold = 7
    smoothing = "running_median"
    response_bins = 256
    bins = np.logspace(np.log10(2.0e-4), np.log10(1.2e-2), 140)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    backend = FastGBBackend(
        delta_t=10.0,
        duration=duration,
        n_samples=response_bins,
        orbits=default_lisaorbits("equal"),
    )
    sources = draw_population(n_sources, seed=4321)

    start = time.perf_counter()
    circular_components = [circular_component_from_fastgb_source(source) for source in sources]
    circular = frequency_domain_xyz_from_components(backend, circular_components)
    circular_freq, circular_psd = ae_mean_psd(circular, duration)
    circular_binned = binned_mean(circular_freq, circular_psd, bins)
    circular_s = time.perf_counter() - start

    paper_psd = karnesis_2021_cgb_confusion_psd(
        bin_centers,
        t_obs_years=t_obs_years,
        snr_threshold=snr_threshold,
        smoothing=smoothing,
    )

    rows = []
    ratios_by_e = {}
    corrected_by_e = {}
    runtimes = {}
    for eccentricity in eccentricities:
        if eccentricity == 0.0:
            ratio = np.ones_like(bin_centers)
            elapsed = 0.0
        else:
            start = time.perf_counter()
            eccentric_components = []
            for source in sources:
                eccentric_components.extend(
                    eccentric_components_for_fastgb_response(
                        circular_to_eccentric_source(source, eccentricity),
                        order=eccentric_order,
                        samples=harmonic_samples,
                        circular_reference=source,
                    )
                )
            eccentric = frequency_domain_xyz_from_components(backend, eccentric_components)
            eccentric_freq, eccentric_psd = ae_mean_psd(eccentric, duration)
            eccentric_binned = binned_mean(eccentric_freq, eccentric_psd, bins)
            ratio = eccentric_binned / circular_binned
            elapsed = time.perf_counter() - start
        ratio = np.where(np.isfinite(ratio) & (ratio > 0.0), ratio, np.nan)
        corrected = paper_psd * ratio
        ratios_by_e[eccentricity] = ratio
        corrected_by_e[eccentricity] = corrected
        runtimes[eccentricity] = elapsed
        print(
            f"e={eccentricity:.2f} | response {elapsed:.3f}s | "
            f"median ratio={np.nanmedian(ratio):.3f} | "
            f"max ratio={np.nanmax(ratio):.3f}"
        )

        for i, frequency in enumerate(bin_centers):
            rows.append(
                {
                    "frequency_hz": frequency,
                    "eccentricity": eccentricity,
                    "ratio_eccentric_over_circular": ratio[i],
                    "karnesis_circular_psd": paper_psd[i],
                    "eccentric_corrected_psd": corrected[i],
                }
            )

    csv_path = out_dir / "eccentric_corrected_karnesis_confusion.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.2), sharex=True, constrained_layout=True)
    axes[0].loglog(bin_centers, paper_psd, color="black", linewidth=2.0, label="Karnesis circular CGB model")
    for eccentricity in eccentricities:
        if eccentricity == 0.0:
            continue
        axes[0].loglog(bin_centers, corrected_by_e[eccentricity], linewidth=1.2, label=f"corrected e={eccentricity:g}")
        axes[1].semilogx(bin_centers, ratios_by_e[eccentricity], linewidth=1.2, label=f"e={eccentricity:g}")

    axes[0].set_ylabel("PSD [1/Hz]")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Eccentric correction to Karnesis CGB confusion, Tobs={t_obs_years:g} yr")

    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("Monte Carlo ratio")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    finite_ratios = np.concatenate([r[np.isfinite(r)] for e, r in ratios_by_e.items() if e != 0.0])
    axes[1].set_ylim(0.0, min(np.nanpercentile(finite_ratios, 98) * 1.2, 5.0))

    png_path = out_dir / "eccentric_corrected_karnesis_confusion.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"circular Monte Carlo response time: {circular_s:.3f}s")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
