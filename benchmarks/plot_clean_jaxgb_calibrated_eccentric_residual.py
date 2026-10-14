"""Clean JAXGB-calibrated residual-vs-eccentricity plot.

This reads the detailed JAXGB/brute-force comparison CSV and extracts the
convention-free circular-limit residual: local eccentric brute force compared
against the local e=0 brute-force waveform. This is the version that removes
JAXGB/local amplitude, phase, Fourier-normalization, and finite-support
convention differences. The plotted curve is the best, i.e. minimum, residual
over the tested frequencies at each eccentricity.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    in_csv = out_dir / "jaxgb_bruteforce_circular_eccentric_comparison.csv"
    rows = list(csv.DictReader(in_csv.open()))
    method = "brute eccentric vs brute e=0 full spectrum"

    by_f0: dict[float, dict[float, float]] = defaultdict(dict)
    for row in rows:
        if row["method"] != method:
            continue
        by_f0[float(row["f0_hz"])][float(row["eccentricity"])] = float(row["scaled_residual_xyz"])

    eccentricities = sorted({e for values in by_f0.values() for e in values})
    clean_rows = []
    for eccentricity in eccentricities:
        residual_values = []
        for f0, values in by_f0.items():
            if eccentricity not in values:
                continue
            residual_values.append((f0, values[eccentricity]))
        if not residual_values:
            continue
        best = min(residual_values, key=lambda item: item[1])
        clean_rows.append(
            {
                "eccentricity": eccentricity,
                "best_f0_hz": best[0],
                "best_convention_free_residual": best[1],
                "median_convention_free_residual": float(np.median([item[1] for item in residual_values])),
                "min_convention_free_residual": float(np.min([item[1] for item in residual_values])),
                "max_convention_free_residual": float(np.max([item[1] for item in residual_values])),
            }
        )

    out_csv = out_dir / "jaxgb_calibrated_best_residual_vs_eccentricity.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean_rows[0]))
        writer.writeheader()
        writer.writerows(clean_rows)

    x = np.array([row["eccentricity"] for row in clean_rows])
    y = np.array([row["best_convention_free_residual"] for row in clean_rows])

    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    ax.plot(x, y, marker="o", linewidth=2.0, color="#0072B2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("convention-free XYZ residual")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(8.0e-5, 0.4)
    ax.set_ylim(max(np.min(y[y > 0]) * 0.5, 1.0e-5), min(np.max(y) * 1.8, 2.0))

    out_png = out_dir / "jaxgb_calibrated_best_residual_vs_eccentricity.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(f"wrote {out_csv}")
    print(f"wrote {out_png}")
    for row in clean_rows:
        print(
            f"e={row['eccentricity']:.1e}: best residual="
            f"{row['best_convention_free_residual']:.3e} at f0={row['best_f0_hz']:.1e} Hz"
        )


if __name__ == "__main__":
    main()
