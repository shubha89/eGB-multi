"""Compare fixed-eccentric trilinear XYZ against JAX brute-force eccentric links.

Run with:

    python benchmarks/fixed_eccentric_trilinear_comparison.py

Outputs:

    benchmarks/fixed_eccentric_trilinear_comparison.csv
    benchmarks/fixed_eccentric_trilinear_comparison.png
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.constants import KILOPARSEC_M
from egb_multi.eccentric import EccentricBinaryParams, recommended_eccentric_order
from egb_multi.eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.tdi import compute_fixed_eccentric_triliear_xyz, pytdi_data_from_links


def elapsed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def xyz_from_links(state, links: dict[str, np.ndarray], *, order: int = 5, delay_order: int = 5) -> dict[str, np.ndarray]:
    from pytdi.michelson import compute_factorized_michelson

    data = pytdi_data_from_links(state, links)
    return {
        "X": compute_factorized_michelson(data, rot=0, order=order, delay_order=delay_order, generation=1, unit="frequency"),
        "Y": compute_factorized_michelson(data, rot=1, order=order, delay_order=delay_order, generation=1, unit="frequency"),
        "Z": compute_factorized_michelson(data, rot=2, order=order, delay_order=delay_order, generation=1, unit="frequency"),
    }


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


def best_fit_scale(model: np.ndarray, reference: np.ndarray) -> complex:
    denom = np.vdot(model, model)
    return 0.0j if denom == 0.0 else complex(np.vdot(model, reference) / denom)


def relative_l2(model: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    return float(np.linalg.norm(model - reference) / denominator) if denominator > 0.0 else np.nan


def combined_xyz_residual(model: dict[str, np.ndarray], reference: dict[str, np.ndarray], *, scaled: bool) -> float:
    numerator = 0.0
    denominator = 0.0
    for channel in ("X", "Y", "Z"):
        model_channel = np.asarray(model[channel])
        reference_channel = np.asarray(reference[channel])
        if scaled:
            model_channel = best_fit_scale(model_channel, reference_channel) * model_channel
        numerator += np.linalg.norm(model_channel - reference_channel) ** 2
        denominator += np.linalg.norm(reference_channel) ** 2
    return float(np.sqrt(numerator / denominator)) if denominator > 0.0 else np.nan


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    duration = 1.0 * 86400.0
    samples = 4096
    harmonic_samples = 1024
    eccentricities = [0.0, 0.01, 0.03, 0.05]
    physics_mode = "1pn"
    orbit_kinds = ["equal", "esa-trailing"]

    times = np.linspace(0.0, duration, samples, endpoint=False)
    rows = []

    for orbit_kind in orbit_kinds:
        state = state_from_lisaorbits(default_lisaorbits(orbit_kind), times)
        geometry = precompute_jax_link_geometry(state)

        warm = make_source(eccentricities[0])
        warm_links = eccentric_links_jax(warm, geometry, batch_size=1, physics_mode=physics_mode)
        xyz_from_links(state, warm_links)
        compute_fixed_eccentric_triliear_xyz(state, warm, harmonic_samples=harmonic_samples, physics_mode=physics_mode)

        for eccentricity in eccentricities:
            source = make_source(eccentricity)
            eccentric_order = recommended_eccentric_order(eccentricity)
            if eccentric_order is None:
                raise RuntimeError("benchmark grid should stay inside the recommended small-e harmonic range")

            brute_links, brute_links_s = elapsed(lambda s=source: eccentric_links_jax(s, geometry, batch_size=1, physics_mode=physics_mode))
            brute_xyz, brute_tdi_s = elapsed(lambda: xyz_from_links(state, brute_links))
            default_xyz, default_s = elapsed(
                lambda s=source: compute_fixed_eccentric_triliear_xyz(
                    state,
                    s,
                    harmonic_samples=harmonic_samples,
                    physics_mode=physics_mode,
                    measurement_order=3,
                    delay_order=3,
                )
            )
            exact_transfer_xyz, exact_transfer_s = elapsed(
                lambda s=source: compute_fixed_eccentric_triliear_xyz(
                    state,
                    s,
                    harmonic_samples=harmonic_samples,
                    physics_mode=physics_mode,
                    order="exact-transfer",
                    measurement_order=5,
                    delay_order=5,
                )
            )

            row = {
                "orbit_kind": orbit_kind,
                "eccentricity": eccentricity,
                "eccentric_order": eccentric_order,
                "duration_days": duration / 86400.0,
                "samples": samples,
                "dt_s": times[1] - times[0],
                "harmonic_samples": harmonic_samples,
                "physics_mode": physics_mode,
                "brute_force_links_s": brute_links_s,
                "brute_force_xyz_s": brute_links_s + brute_tdi_s,
                "fixed_triliear_default_s": default_s,
                "fixed_triliear_exact_transfer_s": exact_transfer_s,
                "default_raw_xyz_residual": combined_xyz_residual(default_xyz, brute_xyz, scaled=False),
                "default_scaled_xyz_residual": combined_xyz_residual(default_xyz, brute_xyz, scaled=True),
                "exact_transfer_raw_xyz_residual": combined_xyz_residual(exact_transfer_xyz, brute_xyz, scaled=False),
                "exact_transfer_scaled_xyz_residual": combined_xyz_residual(exact_transfer_xyz, brute_xyz, scaled=True),
            }
            rows.append(row)
            print(
                f"{orbit_kind} e={eccentricity:.2f} order={eccentric_order} | "
                f"default scaled residual {row['default_scaled_xyz_residual']:.3e} | "
                f"exact-transfer scaled residual {row['exact_transfer_scaled_xyz_residual']:.3e} | "
                f"brute XYZ {row['brute_force_xyz_s']:.4f}s | "
                f"fixed trilinear {default_s:.4f}/{exact_transfer_s:.4f}s",
                flush=True,
            )

    csv_path = out_dir / "fixed_eccentric_trilinear_comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(len(orbit_kinds), 2, figsize=(10.8, 7.8), constrained_layout=True)
    orbit_labels = {"equal": "equal arms", "esa-trailing": "unequal arms"}
    for row_index, orbit_kind in enumerate(orbit_kinds):
        subset = [row for row in rows if row["orbit_kind"] == orbit_kind]
        x = np.array([row["eccentricity"] for row in subset])
        axes[row_index, 0].plot(x, [row["default_scaled_xyz_residual"] for row in subset], marker="o", label="fixed trilinear default")
        axes[row_index, 0].plot(
            x,
            [row["exact_transfer_scaled_xyz_residual"] for row in subset],
            marker="s",
            label="fixed trilinear exact-transfer",
        )
        axes[row_index, 1].plot(x, [row["brute_force_xyz_s"] for row in subset], marker="o", label="JAX brute force + pyTDI")
        axes[row_index, 1].plot(x, [row["fixed_triliear_default_s"] for row in subset], marker="s", label="fixed trilinear default")
        axes[row_index, 1].plot(
            x,
            [row["fixed_triliear_exact_transfer_s"] for row in subset],
            marker="^",
            label="fixed trilinear exact-transfer",
        )

        axes[row_index, 0].set_xlabel("eccentricity")
        axes[row_index, 0].set_ylabel("scaled XYZ relative L2")
        axes[row_index, 0].set_yscale("log")
        axes[row_index, 0].set_title(f"{orbit_labels[orbit_kind]} residual")
        axes[row_index, 1].set_xlabel("eccentricity")
        axes[row_index, 1].set_ylabel("wall time [s]")
        axes[row_index, 1].set_yscale("log")
        axes[row_index, 1].set_title(f"{orbit_labels[orbit_kind]} timing")
        for ax in axes[row_index]:
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)

    png_path = out_dir / "fixed_eccentric_trilinear_comparison.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
