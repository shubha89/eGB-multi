"""Study explicit sideband bases for high-eccentricity fixed sources.

Run with:

    python benchmarks/eccentric_sideband_basis_study.py

Outputs are written to:

    reports/eccentric_sideband_basis_study/
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

from eccentric_grouped_trilinear_ansatz_study import combined_xyz_residual, xyz_from_links

from egb_multi.constants import KILOPARSEC_M
from egb_multi.eccentric import (
    EccentricBinaryParams,
    EccentricHarmonicComponent,
    eccentric_harmonic_projection_angular_frequency,
    eccentric_polarizations,
    eccentric_sideband_components,
)
from egb_multi.eccentric_jax import (
    eccentric_links_jax,
    eccentric_sideband_links_grouped_jax,
    eccentric_sideband_links_jax,
    precompute_jax_link_geometry,
)
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.tensors import tracefree_matrix


def component_polarization_coefficients(source: EccentricBinaryParams, component: EccentricHarmonicComponent) -> tuple[complex, complex]:
    matrix = tracefree_matrix(component.tracefree_coefficients)
    eplus, ecross = source.polarization_basis()
    plus_weight = 0.5 * np.einsum("ij,ij", matrix, eplus)
    cross_weight = 0.5 * np.einsum("ij,ij", matrix, ecross)
    cosps = np.cos(2.0 * source.psi)
    sinps = np.sin(2.0 * source.psi)
    hplus_coeff = plus_weight * cosps - cross_weight * sinps
    hcross_coeff = plus_weight * sinps + cross_weight * cosps
    return hplus_coeff, hcross_coeff


def source_residual(source: EccentricBinaryParams, components: list[EccentricHarmonicComponent], *, samples: int = 4096) -> float:
    projection_frequency = eccentric_harmonic_projection_angular_frequency(source)
    duration = 2.0 * np.pi / projection_frequency
    times = source.t0 + np.linspace(0.0, duration, samples, endpoint=False)
    hplus, hcross = eccentric_polarizations(source, times)
    hplus_model = np.zeros(samples, dtype=np.complex128)
    hcross_model = np.zeros(samples, dtype=np.complex128)
    for component in components:
        hplus_coeff, hcross_coeff = component_polarization_coefficients(source, component)
        carrier = np.exp(2.0j * np.pi * component.frequency * times)
        hplus_model += hplus_coeff * carrier
        hcross_model += hcross_coeff * carrier
    numerator = np.linalg.norm(hplus - np.real(hplus_model)) ** 2 + np.linalg.norm(hcross - np.real(hcross_model)) ** 2
    denominator = np.linalg.norm(hplus) ** 2 + np.linalg.norm(hcross) ** 2
    return float(np.sqrt(numerator / denominator))


def make_source(eccentricity: float) -> EccentricBinaryParams:
    return EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=eccentricity,
        m1_solar=0.6,
        m2_solar=0.4,
        distance_m=KILOPARSEC_M,
        beta=0.3,
        lambda_=-0.7,
        psi=0.4,
        inclination=1.0,
        phi0=0.2,
        fdot=0.0,
    )


def median_elapsed(func, repeats: int = 3):
    timings = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = func()
        timings.append(time.perf_counter() - start)
    return value, float(np.median(timings))


def plot(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})
    for eccentricity in eccentricities:
        subset = sorted([row for row in rows if float(row["eccentricity"]) == eccentricity], key=lambda row: int(row["q_max"]))
        q = [int(row["q_max"]) for row in subset]
        axes[0].plot(q, [float(row["source_residual"]) for row in subset], marker="o", label=f"e={eccentricity:g}")
        axes[1].plot(q, [float(row["xyz_residual"]) for row in subset], marker="s", label=f"e={eccentricity:g}")
        axes[2].plot(q, [float(row["grouped_sideband_wall_s"]) for row in subset], marker="^", label=f"e={eccentricity:g}")
    axes[0].set_title("source reconstruction")
    axes[1].set_title("full Re(XYZ) residual")
    axes[2].set_title("grouped sideband link wall time")
    for ax in axes:
        ax.set_xlabel("q_max")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("scaled residual")
    axes[1].set_ylabel("scaled residual")
    axes[2].set_ylabel("seconds")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict[str, object]], path: Path, plot_path: Path) -> None:
    best = {}
    for row in rows:
        eccentricity = float(row["eccentricity"])
        current = best.get(eccentricity)
        if current is None or float(row["xyz_residual"]) < float(current["xyz_residual"]):
            best[eccentricity] = row
    lines = [
        "# Eccentric Sideband Basis Study",
        "",
        "This study uses explicit radial and periastron-advance quadrupole sidebands fitted by real least squares.",
        "",
        "| e | best q_max | components | source residual | Re(XYZ) residual | grouped sideband [s] | component sideband [s] | brute links [s] |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for eccentricity, row in sorted(best.items()):
        lines.append(
            f"| {eccentricity:.2f} | "
            f"{int(row['q_max'])} | "
            f"{int(row['component_count'])} | "
            f"{float(row['source_residual']):.3e} | "
            f"{float(row['xyz_residual']):.3e} | "
            f"{float(row['grouped_sideband_wall_s']):.4f} | "
            f"{float(row['component_sideband_wall_s']):.4f} | "
            f"{float(row['brute_links_s']):.4f} |"
        )
    lines.extend(["", f"![Sideband study]({plot_path.name})", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "reports" / "eccentric_sideband_basis_study"
    report_dir.mkdir(parents=True, exist_ok=True)

    duration = 86400.0
    samples = 2048
    times = np.linspace(0.0, duration, samples, endpoint=False)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)

    eccentricities = [0.1, 0.2, 0.3]
    q_values = [4, 8, 12, 16, 24]
    rows: list[dict[str, object]] = []
    for eccentricity in eccentricities:
        source = make_source(eccentricity)
        brute_links, brute_links_s = median_elapsed(
            lambda s=source: eccentric_links_jax(s, geometry, batch_size=1, physics_mode="1pn"),
            repeats=3,
        )
        brute_xyz = xyz_from_links(state, brute_links)
        for q_max in q_values:
            components = eccentric_sideband_components(
                source,
                q_max=q_max,
                samples=4096,
                fit_cycles=4.0,
                include_negative_frequencies=False,
                coefficient_tolerance=None,
            )
            src_residual = source_residual(source, components, samples=4096)
            component_links, component_links_s = median_elapsed(
                lambda s=source, q=q_max: eccentric_sideband_links_jax(
                    s,
                    geometry,
                    q_max=q,
                    harmonic_samples=4096,
                    fit_cycles=4.0,
                    batch_size=128,
                    response_mode="fastgb_transfer",
                    include_negative_frequencies=True,
                    coefficient_tolerance=None,
                ),
                repeats=3,
            )
            sideband_links, grouped_links_s = median_elapsed(
                lambda s=source, q=q_max: eccentric_sideband_links_grouped_jax(
                    s,
                    geometry,
                    q_max=q,
                    harmonic_samples=4096,
                    fit_cycles=4.0,
                    batch_size=1,
                    response_mode="fastgb_transfer",
                    include_negative_frequencies=True,
                    coefficient_tolerance=None,
                ),
                repeats=3,
            )
            sideband_xyz = xyz_from_links(state, sideband_links)
            row = {
                "eccentricity": eccentricity,
                "q_max": q_max,
                "component_count": len(components),
                "source_residual": src_residual,
                "brute_links_s": brute_links_s,
                "component_sideband_wall_s": component_links_s,
                "grouped_sideband_wall_s": grouped_links_s,
                "component_grouped_residual": combined_xyz_residual(
                    sideband_xyz,
                    xyz_from_links(state, component_links),
                    scaled=True,
                ),
                "xyz_residual": combined_xyz_residual(sideband_xyz, brute_xyz, scaled=True),
            }
            rows.append(row)
            print(
                f"e={eccentricity:.2f} q_max={q_max:2d} components={len(components):3d} "
                f"source={src_residual:.3e} xyz={row['xyz_residual']:.3e} "
                f"grouped={grouped_links_s:.4f}s component={component_links_s:.4f}s brute={brute_links_s:.4f}s",
                flush=True,
            )

    csv_path = report_dir / "eccentric_sideband_basis_study.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = report_dir / "eccentric_sideband_basis_study.png"
    plot(rows, plot_path)
    write_summary(rows, report_dir / "eccentric_sideband_basis_study.md", plot_path)
    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {report_dir / 'eccentric_sideband_basis_study.md'}")


if __name__ == "__main__":
    main()
