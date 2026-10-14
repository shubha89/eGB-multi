"""Check the Sangria amplitude-to-distance mapping at source-polarization level."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_sangria_small_e_outliers import load_samples
from production_sangria_subset_eccentric_circular import make_source

from egb_multi.eccentric import eccentric_polarizations


SAMPLE_CATALOG = Path(__file__).resolve().parent / (
    "production_sangria_subset_eccentric_circular_tdi2_production_seed20260717_n8perbin_sample_catalog.csv"
)
OUT_CSV = Path(__file__).resolve().parent / "diagnose_sangria_amplitude_convention.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=int, action="append", default=None)
    return parser.parse_args()


def sinusoid_amplitude(y: np.ndarray, phase: np.ndarray) -> float:
    design = np.column_stack([np.cos(phase), np.sin(phase)])
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(np.hypot(coeff[0], coeff[1]))


def main() -> None:
    args = parse_args()
    source_indices = args.source_index or [0, 1, 3, 4]
    samples = load_samples(SAMPLE_CATALOG)
    rows = []
    for source_index in source_indices:
        sample = samples[source_index]
        source = make_source(sample, eccentricity=0.0, fdot=0.0)
        f0_hz = float(sample["f0_hz"])
        period = 1.0 / f0_hz
        times = np.linspace(0.0, 8.0 * period, 4096, endpoint=False)
        hplus, hcross = eccentric_polarizations(source, times, physics_mode="newtonian")
        phase = 2.0 * np.pi * f0_hz * times + float(sample["phi0"])
        measured_plus = sinusoid_amplitude(hplus, phase)
        measured_cross = sinusoid_amplitude(hcross, phase)
        amp = float(sample["amplitude"])
        cosi = np.cos(float(sample["inclination"]))
        expected_plus = amp * (1.0 + cosi * cosi)
        expected_cross = 2.0 * amp * abs(cosi)
        row = {
            "source_index": source_index,
            "f0_hz": f0_hz,
            "catalog_amplitude": amp,
            "inclination": sample["inclination"],
            "measured_plus_amplitude": measured_plus,
            "expected_plus_amplitude": expected_plus,
            "plus_ratio": measured_plus / expected_plus,
            "measured_cross_amplitude": measured_cross,
            "expected_cross_amplitude": expected_cross,
            "cross_ratio": measured_cross / expected_cross if expected_cross > 0 else np.nan,
        }
        rows.append(row)
        print(
            f"src={source_index:03d}: plus ratio={row['plus_ratio']:.6f}, "
            f"cross ratio={row['cross_ratio']:.6f}",
            flush=True,
        )

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
