"""Contour plots of eccentric-vs-circular mismatch over mass and eccentricity."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def chirp_mass(m1: float, m2: float) -> float:
    total = m1 + m2
    eta = m1 * m2 / total**2
    return total * eta ** (3.0 / 5.0)


def load_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) if key != "preset" else value for key, value in row.items()})
    return rows


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "production_eccentric_circular_mismatch_grid_production.csv"
    rows = load_rows(csv_path)

    frequencies = sorted({float(row["f0_hz"]) for row in rows})
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})
    mass_pairs = sorted({(float(row["m1_solar"]), float(row["m2_solar"])) for row in rows}, key=lambda pair: chirp_mass(*pair))
    chirp_masses = np.array([chirp_mass(*pair) for pair in mass_pairs])

    positive = [float(row["mismatch_AE"]) for row in rows if float(row["mismatch_AE"]) > 0.0]
    vmin = max(min(positive), 1.0e-8)
    vmax = max(positive)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.4), constrained_layout=True, sharex=True, sharey=True)
    last_mesh = None
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 28)
    for ax, f0 in zip(axes.flat, frequencies, strict=True):
        z = np.zeros((len(eccentricities), len(chirp_masses)), dtype=np.float64)
        for i, eccentricity in enumerate(eccentricities):
            for j, pair in enumerate(mass_pairs):
                values = [
                    max(float(row["mismatch_AE"]), vmin)
                    for row in rows
                    if float(row["f0_hz"]) == f0
                    and float(row["eccentricity"]) == eccentricity
                    and float(row["m1_solar"]) == pair[0]
                    and float(row["m2_solar"]) == pair[1]
                ]
                z[i, j] = np.median(values)

        last_mesh = ax.contourf(
            chirp_masses,
            eccentricities,
            np.maximum(z, vmin),
            levels=levels,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="viridis",
            extend="both",
        )
        ax.contour(
            chirp_masses,
            eccentricities,
            np.maximum(z, vmin),
            levels=levels[::4],
            colors="white",
            linewidths=0.45,
            alpha=0.55,
        )
        ax.set_yscale("symlog", linthresh=1.0e-3)
        ax.set_title(f"$f_0={f0:.0e}$ Hz")
        ax.grid(True, alpha=0.18)
        ax.set_xticks(chirp_masses)
        ax.set_xticklabels([f"{cm:.2f}" for cm in chirp_masses])

    for ax in axes[:, 0]:
        ax.set_ylabel("eccentricity")
    for ax in axes[-1, :]:
        ax.set_xlabel("chirp mass [$M_\\odot$]")

    assert last_mesh is not None
    cbar = fig.colorbar(last_mesh, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("median A/E mismatch, $1-\\mathrm{FF}$")
    mass_label = ", ".join(f"{m1:.1f}+{m2:.1f}$M_\\odot$" for m1, m2 in mass_pairs)
    fig.suptitle(f"Full-response eccentric vs circular mismatch over mass and eccentricity; mass pairs: {mass_label}")

    png_path = out_dir / "production_mass_eccentricity_mismatch_contours.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    print(f"wrote {png_path}")


def edges_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mids = 0.5 * (centers[1:] + centers[:-1])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate(([first], mids, [last]))


if __name__ == "__main__":
    main()
