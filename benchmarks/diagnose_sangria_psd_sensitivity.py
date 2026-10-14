"""PSD-weighting sensitivity for Sangria small-e diagnostics."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_sangria_small_e_outliers import load_samples
from production_sangria_subset_eccentric_circular import compare_aet_with_absolute_residual, make_source
from production_eccentric_circular_fdot_tdi2_mismatch_grid import response_aet
from production_eccentric_circular_mismatch_grid import load_aet_psd

from egb_multi.eccentric_jax import precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits


SAMPLE_CATALOG = Path(__file__).resolve().parent / (
    "production_sangria_subset_eccentric_circular_tdi2_production_seed20260717_n8perbin_sample_catalog.csv"
)
OUT_CSV = Path(__file__).resolve().parent / "diagnose_sangria_psd_sensitivity.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=int, action="append", default=None)
    parser.add_argument("--eccentricity", type=float, default=1.0e-3)
    parser.add_argument("--duration-days", type=float, default=365.25)
    parser.add_argument("--dt-s", type=float, default=50.0)
    return parser.parse_args()


def log_smooth_positive(values: np.ndarray, width: int = 9) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    positive = np.isfinite(out) & (out > 0.0)
    if not np.any(positive):
        return out
    logv = np.log(out[positive])
    pad = width // 2
    padded = np.pad(logv, pad, mode="edge")
    smoothed = np.array([np.median(padded[i : i + width]) for i in range(logv.size)])
    out[positive] = np.exp(smoothed)
    return out


def main() -> None:
    args = parse_args()
    source_indices = args.source_index or [0, 1, 3, 4]
    samples = load_samples(SAMPLE_CATALOG)
    times = np.arange(0.0, args.duration_days * 86400.0, args.dt_s, dtype=np.float64)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)
    freq = np.fft.rfftfreq(times.size, args.dt_s)
    psd_freq, psd_base = load_aet_psd(Path(__file__).resolve().parent / "lisainstrument_welch_psd.csv")
    welch = {channel: np.interp(freq, psd_freq, psd_base[channel], left=np.inf, right=np.inf) for channel in ("A", "E", "T")}
    smoothed = {channel: log_smooth_positive(welch[channel], width=11) for channel in ("A", "E", "T")}
    flat = {channel: np.ones_like(freq) for channel in ("A", "E", "T")}
    psd_cases = {"welch": welch, "smoothed_welch": smoothed, "flat": flat}

    rows = []
    for source_index in source_indices:
        sample = samples[source_index]
        f0_hz = float(sample["f0_hz"])
        band_mask = (freq >= 0.20 * f0_hz) & (freq <= min(freq[-1], 10.0 * f0_hz))
        fdot = float(sample["fdot_hz_s"])
        circular = response_aet(make_source(sample, eccentricity=0.0, fdot=fdot), geometry, state, batch_size=1, generation=2, measurement_order=3, delay_order=3)
        eccentric = response_aet(
            make_source(sample, eccentricity=args.eccentricity, fdot=fdot),
            geometry,
            state,
            batch_size=1,
            generation=2,
            measurement_order=3,
            delay_order=3,
        )
        for psd_name, psd in psd_cases.items():
            metrics = compare_aet_with_absolute_residual(eccentric, circular, psd, args.dt_s, band_mask.copy())
            row = {
                "source_index": source_index,
                "psd": psd_name,
                "f0_hz": f0_hz,
                "eccentricity": args.eccentricity,
                **metrics,
            }
            rows.append(row)
            print(
                f"src={source_index:03d} psd={psd_name}: FF={metrics['fitting_factor_AE']:.8f} "
                f"mismatch={metrics['mismatch_AE']:.3e} relres={metrics['relative_noise_weighted_residual_AE']:.3e}",
                flush=True,
            )

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
