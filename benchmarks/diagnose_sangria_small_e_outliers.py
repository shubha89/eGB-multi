"""Diagnostics for large small-e mismatches in the Sangria subset run."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from production_sangria_subset_eccentric_circular import (
    compare_aet_with_absolute_residual,
    make_source,
)
from production_eccentric_circular_fdot_tdi2_mismatch_grid import response_aet
from production_eccentric_circular_mismatch_grid import aet_inner, fd_aet, load_aet_psd

from egb_multi.eccentric_jax import precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits


SAMPLE_CATALOG = Path(__file__).resolve().parent / (
    "production_sangria_subset_eccentric_circular_tdi2_production_seed20260717_n8perbin_sample_catalog.csv"
)
OUT_CSV = Path(__file__).resolve().parent / "diagnose_sangria_small_e_outliers.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=int, action="append", default=None)
    parser.add_argument("--duration-days", type=float, default=365.25)
    parser.add_argument("--dt-s", type=float, default=50.0)
    parser.add_argument("--eccentricity", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--measurement-order", type=int, default=3)
    parser.add_argument("--delay-order", type=int, default=3)
    parser.add_argument("--tdi-generation", type=int, default=2)
    return parser.parse_args()


def load_samples(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in csv.DictReader(path.open()):
        converted: dict[str, object] = {}
        for key, value in row.items():
            if key in {"catalog", "catalog_index"}:
                converted[key] = value if key == "catalog" else int(value)
            else:
                converted[key] = float(value)
        rows.append(converted)
    return rows


def channel_norms(aet: dict[str, np.ndarray], psd: dict[str, np.ndarray], dt: float, mask: np.ndarray) -> dict[str, float]:
    fd = fd_aet(aet, dt)
    df = 1.0 / (np.asarray(aet["A"]).size * dt)
    return {channel: float(np.sqrt(max(aet_inner(fd, fd, psd, df, mask.copy(), channels=(channel,)).real, 0.0))) for channel in ("A", "E")}


def harmonic_window_norms(
    aet: dict[str, np.ndarray],
    psd: dict[str, np.ndarray],
    dt: float,
    freq: np.ndarray,
    f0_hz: float,
) -> dict[str, float]:
    out = {}
    for label, lo, hi in (
        ("k1", 0.45, 0.55),
        ("k2", 0.90, 1.10),
        ("k3", 1.40, 1.60),
    ):
        mask = (freq >= lo * f0_hz) & (freq <= hi * f0_hz)
        norms = channel_norms(aet, psd, dt, mask)
        out[f"{label}_snr_A"] = norms["A"]
        out[f"{label}_snr_E"] = norms["E"]
        out[f"{label}_snr_AE"] = float(np.hypot(norms["A"], norms["E"]))
    return out


def main() -> None:
    args = parse_args()
    samples = load_samples(SAMPLE_CATALOG)
    source_indices = args.source_index or [0, 1, 3, 4]
    selected = [samples[index] for index in source_indices]

    out_dir = Path(__file__).resolve().parent
    psd_freq, psd_base = load_aet_psd(out_dir / "lisainstrument_welch_psd.csv")
    times = np.arange(0.0, args.duration_days * 86400.0, args.dt_s, dtype=np.float64)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)
    freq = np.fft.rfftfreq(times.size, args.dt_s)
    psd = {channel: np.interp(freq, psd_freq, psd_base[channel], left=np.inf, right=np.inf) for channel in ("A", "E", "T")}

    rows: list[dict[str, object]] = []
    for sample_index, sample in zip(source_indices, selected, strict=True):
        f0_hz = float(sample["f0_hz"])
        catalog_fdot = float(sample["fdot_hz_s"])
        full_mask = (freq >= 0.20 * f0_hz) & (freq <= min(freq[-1], 10.0 * f0_hz))

        circular_same_fdot = response_aet(
            make_source(sample, eccentricity=0.0, fdot=catalog_fdot),
            geometry,
            state,
            batch_size=args.batch_size,
            generation=args.tdi_generation,
            measurement_order=args.measurement_order,
            delay_order=args.delay_order,
        )
        circular_zero_fdot = response_aet(
            make_source(sample, eccentricity=0.0, fdot=0.0),
            geometry,
            state,
            batch_size=args.batch_size,
            generation=args.tdi_generation,
            measurement_order=args.measurement_order,
            delay_order=args.delay_order,
        )
        eccentric_same_fdot = response_aet(
            make_source(sample, eccentricity=args.eccentricity, fdot=catalog_fdot),
            geometry,
            state,
            batch_size=args.batch_size,
            generation=args.tdi_generation,
            measurement_order=args.measurement_order,
            delay_order=args.delay_order,
        )
        eccentric_zero_fdot = response_aet(
            make_source(sample, eccentricity=args.eccentricity, fdot=0.0),
            geometry,
            state,
            batch_size=args.batch_size,
            generation=args.tdi_generation,
            measurement_order=args.measurement_order,
            delay_order=args.delay_order,
        )

        cases = [
            ("e0_self_same_fdot", circular_same_fdot, circular_same_fdot),
            ("e0_fdot_vs_zero", circular_same_fdot, circular_zero_fdot),
            ("small_e_same_fdot", eccentric_same_fdot, circular_same_fdot),
            ("small_e_zero_fdot", eccentric_zero_fdot, circular_zero_fdot),
        ]
        circular_norms = harmonic_window_norms(circular_same_fdot, psd, args.dt_s, freq, f0_hz)
        eccentric_norms = harmonic_window_norms(eccentric_same_fdot, psd, args.dt_s, freq, f0_hz)
        for label, signal, template in cases:
            metrics = compare_aet_with_absolute_residual(signal, template, psd, args.dt_s, full_mask.copy())
            row = {
                "source_index": sample_index,
                "case": label,
                "catalog_index": sample["catalog_index"],
                "f0_hz": f0_hz,
                "amplitude": sample["amplitude"],
                "fdot_hz_s": catalog_fdot,
                "inclination": sample["inclination"],
                "beta": sample["beta"],
                "lambda": sample["lambda"],
                "psi": sample["psi"],
                "phi0": sample["phi0"],
                **metrics,
            }
            if label == "small_e_same_fdot":
                row.update({f"circ_{key}": value for key, value in circular_norms.items()})
                row.update({f"ecc_{key}": value for key, value in eccentric_norms.items()})
            rows.append(row)
            print(
                f"src={sample_index:03d} {label}: FF={metrics['fitting_factor_AE']:.8f} "
                f"mismatch={metrics['mismatch_AE']:.3e} relres={metrics['relative_noise_weighted_residual_AE']:.3e} "
                f"SNR={metrics['signal_snr_AE']:.3e}",
                flush=True,
            )

    with OUT_CSV.open("w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
