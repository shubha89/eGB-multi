"""Small circular-template maximization checks for Sangria subset sources."""

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

from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits


SAMPLE_CATALOG = Path(__file__).resolve().parent / (
    "production_sangria_subset_eccentric_circular_tdi2_production_seed20260717_n8perbin_sample_catalog.csv"
)
OUT_CSV = Path(__file__).resolve().parent / "diagnose_sangria_template_maximization.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=int, action="append", default=None)
    parser.add_argument("--eccentricity", type=float, default=1.0e-3)
    parser.add_argument("--duration-days", type=float, default=365.25)
    parser.add_argument("--dt-s", type=float, default=50.0)
    parser.add_argument("--phase-points", type=int, default=9)
    parser.add_argument("--f0-bin-offsets", type=int, nargs="*", default=[-2, -1, 0, 1, 2])
    return parser.parse_args()


def shifted_source(sample: dict[str, object], *, eccentricity: float, fdot: float, f0_shift_hz: float, phi_shift: float) -> EccentricBinaryParams:
    source = make_source(sample, eccentricity=eccentricity, fdot=fdot)
    return EccentricBinaryParams(
        mean_motion=np.pi * (float(sample["f0_hz"]) + f0_shift_hz),
        eccentricity=source.eccentricity,
        m1_solar=source.m1_solar,
        m2_solar=source.m2_solar,
        distance_m=source.distance_m,
        beta=source.beta,
        lambda_=source.lambda_,
        psi=source.psi,
        inclination=source.inclination,
        phi0=source.phi0 + phi_shift,
        fdot=source.fdot,
    )


def main() -> None:
    args = parse_args()
    samples = load_samples(SAMPLE_CATALOG)
    source_indices = args.source_index or [0, 1]
    times = np.arange(0.0, args.duration_days * 86400.0, args.dt_s, dtype=np.float64)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)
    freq = np.fft.rfftfreq(times.size, args.dt_s)
    psd_freq, psd_base = load_aet_psd(Path(__file__).resolve().parent / "lisainstrument_welch_psd.csv")
    psd = {channel: np.interp(freq, psd_freq, psd_base[channel], left=np.inf, right=np.inf) for channel in ("A", "E", "T")}
    df = 1.0 / (times.size * args.dt_s)
    phase_grid = np.linspace(-np.pi, np.pi, args.phase_points, endpoint=False)

    rows: list[dict[str, object]] = []
    for source_index in source_indices:
        sample = samples[source_index]
        f0_hz = float(sample["f0_hz"])
        fdot = float(sample["fdot_hz_s"])
        band_mask = (freq >= 0.20 * f0_hz) & (freq <= min(freq[-1], 10.0 * f0_hz))
        eccentric = response_aet(
            make_source(sample, eccentricity=args.eccentricity, fdot=fdot),
            geometry,
            state,
            batch_size=1,
            generation=2,
            measurement_order=3,
            delay_order=3,
        )

        candidates = []
        for mode in ("baseline", "phase", "f0", "phase_f0"):
            phase_values = [0.0] if mode in {"baseline", "f0"} else phase_grid
            f0_values = [0.0] if mode in {"baseline", "phase"} else [offset * df for offset in args.f0_bin_offsets]
            best = None
            best_phi = None
            best_f0_shift = None
            count = 0
            for phi_shift in phase_values:
                for f0_shift in f0_values:
                    count += 1
                    circular = response_aet(
                        shifted_source(
                            sample,
                            eccentricity=0.0,
                            fdot=fdot,
                            f0_shift_hz=float(f0_shift),
                            phi_shift=float(phi_shift),
                        ),
                        geometry,
                        state,
                        batch_size=1,
                        generation=2,
                        measurement_order=3,
                        delay_order=3,
                    )
                    metrics = compare_aet_with_absolute_residual(eccentric, circular, psd, args.dt_s, band_mask.copy())
                    if best is None or metrics["fitting_factor_AE"] > best["fitting_factor_AE"]:
                        best = metrics
                        best_phi = float(phi_shift)
                        best_f0_shift = float(f0_shift)
            assert best is not None
            row = {
                "source_index": source_index,
                "mode": mode,
                "candidate_count": count,
                "f0_hz": f0_hz,
                "eccentricity": args.eccentricity,
                "best_phi_shift_rad": best_phi,
                "best_f0_shift_hz": best_f0_shift,
                "best_f0_shift_bins": best_f0_shift / df,
                **best,
            }
            rows.append(row)
            print(
                f"src={source_index:03d} {mode}: candidates={count} FF={best['fitting_factor_AE']:.8f} "
                f"mismatch={best['mismatch_AE']:.3e} relres={best['relative_noise_weighted_residual_AE']:.3e} "
                f"phi={best_phi:.3f} df_bins={best_f0_shift / df:.1f}",
                flush=True,
            )

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
