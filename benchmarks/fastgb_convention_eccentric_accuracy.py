"""Apples-to-apples FastGB-convention eccentric accuracy check.

This benchmark uses FastGB's response algebra for both sides:

1. FastGB package circular FD X/Y/Z,
2. local circular reproduction of FastGB's response algebra,
3. local eccentric harmonic source inserted into the same FastGB response
   algebra.

This isolates source eccentricity/convention effects from the separate
time-domain pyTDI response pipeline.

Run with:

    python benchmarks/fastgb_convention_eccentric_accuracy.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.constants import C_M_PER_S, G_M3_KG_S2, KILOPARSEC_M, SOLAR_MASS_KG
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import FastGBBackend, FastGBResult, default_lisaorbits
from egb_multi.fastgb_convention import fastgb_convention_circular_xyz, fastgb_convention_eccentric_xyz
from egb_multi.sources import GalacticBinaryParams


def circular_strain_amplitude(source: EccentricBinaryParams) -> float:
    total_mass = (source.m1_solar + source.m2_solar) * SOLAR_MASS_KG
    eta = source.symmetric_mass_ratio
    return float(
        2.0
        * eta
        * (G_M3_KG_S2 * total_mass) ** (5.0 / 3.0)
        * source.mean_motion ** (2.0 / 3.0)
        / (C_M_PER_S**4 * source.distance_m)
    )


def as_fastgb_source(source: EccentricBinaryParams) -> GalacticBinaryParams:
    return GalacticBinaryParams(
        f0=source.mean_motion / np.pi,
        amplitude=circular_strain_amplitude(source),
        beta=source.beta,
        lambda_=source.lambda_,
        psi=source.psi,
        inclination=source.inclination,
        phi0=source.phi0,
    )


def best_fit_scale(model: np.ndarray, reference: np.ndarray) -> complex:
    denom = np.vdot(model, model)
    return 0.0j if denom == 0 else complex(np.vdot(model, reference) / denom)


def compare_xyz(model: FastGBResult, reference: FastGBResult) -> dict[str, float]:
    model_kmin = int(model.kmin[0])
    reference_kmin = int(reference.kmin[0])
    first = max(model_kmin, reference_kmin)
    last = min(model_kmin + model.x.shape[-1], reference_kmin + reference.x.shape[-1])
    if first >= last:
        raise ValueError("model and reference have no overlapping frequency bins")

    row: dict[str, float] = {
        "overlap_bins": float(last - first),
        "overlap_kmin": float(first),
    }
    raw_num = 0.0
    scaled_num = 0.0
    den = 0.0
    for channel in ("x", "y", "z"):
        model_bins = getattr(model, channel)[0, first - model_kmin : last - model_kmin]
        reference_bins = getattr(reference, channel)[0, first - reference_kmin : last - reference_kmin]
        alpha = best_fit_scale(model_bins, reference_bins)
        scaled = alpha * model_bins
        raw_num += np.linalg.norm(model_bins - reference_bins) ** 2
        scaled_num += np.linalg.norm(scaled - reference_bins) ** 2
        den += np.linalg.norm(reference_bins) ** 2
        row[f"{channel.upper()}_raw_relerr"] = float(np.linalg.norm(model_bins - reference_bins) / np.linalg.norm(reference_bins))
        row[f"{channel.upper()}_scaled_relerr"] = float(np.linalg.norm(scaled - reference_bins) / np.linalg.norm(reference_bins))
        row[f"{channel.upper()}_scale_abs"] = abs(alpha)
        row[f"{channel.upper()}_scale_phase"] = float(np.angle(alpha))
    row["XYZ_raw_relerr"] = float(np.sqrt(raw_num / den))
    row["XYZ_scaled_relerr"] = float(np.sqrt(scaled_num / den))
    return row


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    eccentricity = 1.0e-4
    frequencies = [1.0e-4, 3.0e-4, 1.0e-3]
    backend = FastGBBackend(
        delta_t=100.0,
        duration=90.0 * 24.0 * 3600.0,
        n_samples=128,
        orbits=default_lisaorbits("equal"),
    )

    rows = []
    for f0 in frequencies:
        eccentric_source = EccentricBinaryParams(
            mean_motion=np.pi * f0,
            eccentricity=eccentricity,
            m1_solar=0.5,
            m2_solar=0.5,
            distance_m=KILOPARSEC_M,
            beta=0.3,
            lambda_=-0.7,
            psi=0.4,
            inclination=1.0,
            phi0=0.2,
        )
        fastgb_source = as_fastgb_source(eccentric_source)
        fastgb_circular = backend.frequency_domain_xyz(fastgb_source)
        local_circular = fastgb_convention_circular_xyz(backend, fastgb_source)
        local_eccentric = fastgb_convention_eccentric_xyz(
            backend,
            eccentric_source,
            order=1,
            samples=2048,
            circular_reference=fastgb_source,
        )

        circular_row = compare_xyz(local_circular, fastgb_circular)
        eccentric_row = compare_xyz(local_eccentric, fastgb_circular)
        row = {
            "f0": f0,
            "eccentricity": eccentricity,
            "fastgb_kmin": int(fastgb_circular.kmin[0]),
            "eccentric_kmin": int(local_eccentric.kmin[0]),
            "eccentric_bins": local_eccentric.x.shape[-1],
            "circular_reproduction_XYZ_raw_relerr": circular_row["XYZ_raw_relerr"],
            "circular_reproduction_XYZ_scaled_relerr": circular_row["XYZ_scaled_relerr"],
            **{f"eccentric_{key}": value for key, value in eccentric_row.items()},
        }
        rows.append(row)
        print(
            f"f0={f0:.1e} Hz | circular reproduction raw={row['circular_reproduction_XYZ_raw_relerr']:.3e} | "
            f"eccentric raw={row['eccentric_XYZ_raw_relerr']:.3e} | "
            f"eccentric scaled={row['eccentric_XYZ_scaled_relerr']:.3e}"
        )

    csv_path = out_dir / "fastgb_convention_eccentric_accuracy.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    x = np.array([row["f0"] for row in rows])
    ax.plot(x, [row["circular_reproduction_XYZ_raw_relerr"] for row in rows], marker="o", label="local circular vs FastGB")
    ax.plot(x, [row["eccentric_XYZ_scaled_relerr"] for row in rows], marker="s", label="e=1e-4 eccentric vs FastGB, scaled")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("relative XYZ residual")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("FastGB-convention apples-to-apples residual")
    png_path = out_dir / "fastgb_convention_eccentric_accuracy.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
