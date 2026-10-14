"""FastGB-convention residual as a function of eccentricity.

The reference is FastGB's circular FD X/Y/Z. The model is the local eccentric
harmonic source inserted into the same FastGB response algebra and calibrated
to the circular limit.

Run with:

    python benchmarks/fastgb_convention_eccentricity_sweep.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastgb_convention_eccentric_accuracy import as_fastgb_source, compare_xyz

from egb_multi.constants import KILOPARSEC_M
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import FastGBBackend, default_lisaorbits
from egb_multi.fastgb_convention import fastgb_convention_eccentric_xyz


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    eccentricities = [0.0, 1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 5.0e-2, 1.0e-1, 2.0e-1, 3.0e-1]
    frequencies = [1.0e-4, 3.0e-4, 1.0e-3]
    eccentric_order = 1
    harmonic_samples = 2048
    backend = FastGBBackend(
        delta_t=100.0,
        duration=90.0 * 24.0 * 3600.0,
        n_samples=128,
        orbits=default_lisaorbits("equal"),
    )

    rows = []
    for f0 in frequencies:
        circular_source = EccentricBinaryParams(
            mean_motion=np.pi * f0,
            eccentricity=0.0,
            m1_solar=0.5,
            m2_solar=0.5,
            distance_m=KILOPARSEC_M,
            beta=0.3,
            lambda_=-0.7,
            psi=0.4,
            inclination=1.0,
            phi0=0.2,
        )
        fastgb_source = as_fastgb_source(circular_source)
        reference = backend.frequency_domain_xyz(fastgb_source)

        for eccentricity in eccentricities:
            source = EccentricBinaryParams(**{**circular_source.__dict__, "eccentricity": eccentricity})
            model = fastgb_convention_eccentric_xyz(
                backend,
                source,
                order=eccentric_order,
                samples=harmonic_samples,
                circular_reference=fastgb_source,
            )
            residuals = compare_xyz(model, reference)
            row = {
                "f0": f0,
                "eccentricity": eccentricity,
                "eccentric_order": eccentric_order,
                "harmonic_samples": harmonic_samples,
                "model_kmin": int(model.kmin[0]),
                "model_bins": model.x.shape[-1],
                **residuals,
            }
            rows.append(row)
            print(
                f"f0={f0:.1e} Hz e={eccentricity:.4f} | "
                f"XYZ raw={row['XYZ_raw_relerr']:.3e} | "
                f"XYZ scaled={row['XYZ_scaled_relerr']:.3e}"
            )

    csv_path = out_dir / "fastgb_convention_eccentricity_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    for f0 in frequencies:
        subset = [row for row in rows if row["f0"] == f0]
        x = np.array([row["eccentricity"] for row in subset])
        axes[0].plot(x, [row["XYZ_raw_relerr"] for row in subset], marker="o", label=f"f0={f0:.0e} Hz")
        axes[1].plot(x, [row["XYZ_scaled_relerr"] for row in subset], marker="s", label=f"f0={f0:.0e} Hz")

    for ax in axes:
        ax.set_xscale("symlog", linthresh=1.0e-4)
        ax.set_yscale("log")
        ax.set_xlabel("eccentricity")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("relative XYZ residual")
    axes[0].set_title("raw residual")
    axes[1].set_title("best-fit complex-scaled residual")
    fig.suptitle("FastGB-convention eccentric source vs circular FastGB")
    png_path = out_dir / "fastgb_convention_eccentricity_sweep.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
