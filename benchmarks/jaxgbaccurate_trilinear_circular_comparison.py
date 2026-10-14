"""Compare JAXGBaccurate and trilinear circular pyTDI responses.

This is a convention-aware diagnostic rather than a final validation.  The
JAXGBaccurate response is generated in its native heterodyned frequency-domain
form and transformed back to the observation interval.  The local trilinear
responses are generated through `compute_triliear_xyz`.  Both are compared to
the local exact-link pyTDI response after fitting one complex scale per channel.

Run with:

    python benchmarks/jaxgbaccurate_trilinear_circular_comparison.py

Outputs:

    benchmarks/jaxgbaccurate_trilinear_circular_comparison.csv
    benchmarks/jaxgbaccurate_trilinear_circular_comparison_summary.csv
    benchmarks/jaxgbaccurate_trilinear_circular_comparison.png
    benchmarks/jaxgbaccurate_trilinear_circular_comparison_overlay_f0_1e-03.png
    benchmarks/jaxgbaccurate_trilinear_circular_comparison_overlay_equal_f0_1e-03.png
    benchmarks/jaxgbaccurate_trilinear_circular_comparison_overlay_esa-trailing_f0_1e-03.png
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jaxgb.jaxgb import JaxGBaccurate

from egb_multi.constants import C_M_PER_S, G_M3_KG_S2, KILOPARSEC_M, SOLAR_MASS_KG
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.noise import aet_from_xyz
from egb_multi.sources import GalacticBinaryParams
from egb_multi.tdi import compute_triliear_xyz, compute_xyz


def elapsed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def make_eccentric_source(f0: float) -> EccentricBinaryParams:
    """Return the e=0 parameter container used for JAXGBaccurate inputs."""

    return EccentricBinaryParams(
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
        fdot=0.0,
    )


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


def as_jaxgb_params(source: EccentricBinaryParams) -> jnp.ndarray:
    return jnp.asarray(
        [
            [
                source.mean_motion / np.pi,
                source.fdot,
                circular_strain_amplitude(source),
                source.lambda_,
                source.beta,
                source.psi,
                source.inclination,
                source.phi0,
            ]
        ],
        dtype=jnp.float64,
    )


def as_galactic_binary(source: EccentricBinaryParams) -> GalacticBinaryParams:
    return GalacticBinaryParams(
        f0=source.mean_motion / np.pi,
        fdot=source.fdot,
        amplitude=circular_strain_amplitude(source),
        beta=source.beta,
        lambda_=source.lambda_,
        psi=source.psi,
        inclination=source.inclination,
        phi0=source.phi0,
    )


def jaxgbaccurate_xyz_time(
    model: JaxGBaccurate,
    params: jnp.ndarray,
    *,
    dt: float,
    obs_samples: int,
) -> dict[str, np.ndarray]:
    tdi = model.get_tdi(params, tdi_generation=1.5, tdi_combination="XYZ")
    kmin = int(np.asarray(model.get_kmin(params[:, 0]))[0])
    target_length = int(np.rint(model.t_ext / dt))
    crop_start = int(np.rint(model.extension * model.t_obs / dt))
    out = {}
    for channel, bins in zip(("X", "Y", "Z"), tdi, strict=True):
        spectrum = np.zeros(target_length, dtype=np.complex128)
        values = np.asarray(bins[0], dtype=np.complex128)
        indices = np.arange(kmin, kmin + values.size, dtype=np.int64) % target_length
        spectrum[indices] = values
        time_series = np.fft.ifft(spectrum / dt, n=target_length)
        out[channel] = time_series[crop_start : crop_start + obs_samples]
    return out


def best_fit_scale(model: np.ndarray, reference: np.ndarray) -> complex:
    denom = np.vdot(model, model)
    return 0.0j if denom == 0.0 else complex(np.vdot(model, reference) / denom)


def relative_l2(model: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    return float(np.linalg.norm(model - reference) / denominator) if denominator > 0.0 else np.nan


def compare_basis(
    *,
    orbit_kind: str,
    f0: float,
    basis_name: str,
    models: dict[str, dict[str, np.ndarray]],
    reference: dict[str, np.ndarray],
    timings: dict[str, float],
) -> list[dict[str, object]]:
    rows = []
    for model_name, model_channels in models.items():
        for channel in reference:
            model = np.asarray(model_channels[channel])
            ref = np.asarray(reference[channel])
            alpha = best_fit_scale(model, ref)
            scaled = alpha * model
            rows.append(
                {
                    "orbit_kind": orbit_kind,
                    "f0_hz": f0,
                    "basis": basis_name,
                    "channel": channel,
                    "model": model_name,
                    "raw_relative_l2": relative_l2(model, ref),
                    "scaled_relative_l2": relative_l2(scaled, ref),
                    "scale_abs": abs(alpha),
                    "scale_phase": float(np.angle(alpha)),
                    "wall_s": timings[model_name],
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    keys = sorted({(row["orbit_kind"], row["f0_hz"], row["basis"], row["model"]) for row in rows})
    for orbit_kind, f0, basis, model_name in keys:
        subset = [
            row
            for row in rows
            if row["orbit_kind"] == orbit_kind
            and row["f0_hz"] == f0
            and row["basis"] == basis
            and row["model"] == model_name
        ]
        summary.append(
            {
                "orbit_kind": orbit_kind,
                "f0_hz": f0,
                "basis": basis,
                "model": model_name,
                "median_scaled_relative_l2": float(np.median([float(row["scaled_relative_l2"]) for row in subset])),
                "max_scaled_relative_l2": float(np.max([float(row["scaled_relative_l2"]) for row in subset])),
                "median_raw_relative_l2": float(np.median([float(row["raw_relative_l2"]) for row in subset])),
                "wall_s": float(np.median([float(row["wall_s"]) for row in subset])),
            }
        )
    return summary


def scaled_channel(model: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return best_fit_scale(model, reference) * model


def plot_summary(summary_rows: list[dict[str, object]], path: Path) -> None:
    xyz_rows = [row for row in summary_rows if row["basis"] == "XYZ"]
    orbit_order = ["equal", "esa-trailing"]
    orbit_labels = {
        "equal": "equal arms",
        "esa-trailing": "unequal arms",
    }
    model_order = ["jaxgbaccurate", "trilinear_default", "trilinear_exact_transfer"]
    labels = {
        "jaxgbaccurate": "JAXGBaccurate",
        "trilinear_default": "trilinear default",
        "trilinear_exact_transfer": "trilinear exact transfer",
    }
    fig, axes = plt.subplots(len(orbit_order), 2, figsize=(11.0, 7.8), constrained_layout=True)
    for row_index, orbit_kind in enumerate(orbit_order):
        orbit_rows = [row for row in xyz_rows if row["orbit_kind"] == orbit_kind]
        if not orbit_rows:
            continue
        for model_name in model_order:
            subset = sorted([row for row in orbit_rows if row["model"] == model_name], key=lambda row: float(row["f0_hz"]))
            axes[row_index, 0].plot(
                [float(row["f0_hz"]) for row in subset],
                [float(row["median_scaled_relative_l2"]) for row in subset],
                marker="o",
                label=labels[model_name],
            )
            axes[row_index, 1].plot(
                [float(row["f0_hz"]) for row in subset],
                [float(row["wall_s"]) for row in subset],
                marker="o",
                label=labels[model_name],
            )
        axes[row_index, 0].set_ylabel("median scaled relative L2")
        axes[row_index, 1].set_ylabel("wall time [s]")
        axes[row_index, 0].set_title(f"{orbit_labels[orbit_kind]} residual")
        axes[row_index, 1].set_title(f"{orbit_labels[orbit_kind]} timing")
        for ax in axes[row_index]:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("f0 [Hz]")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_overlay(
    *,
    orbit_kind: str,
    f0: float,
    times: np.ndarray,
    exact_xyz: dict[str, np.ndarray],
    jaxgb_xyz: dict[str, np.ndarray],
    trilinear_xyz: dict[str, np.ndarray],
    path: Path,
) -> None:
    channel = "X"
    window = slice(0, min(times.size, 1600))
    exact = np.real(exact_xyz[channel])
    jaxgb = np.real(jaxgb_xyz[channel])
    trilinear = np.real(trilinear_xyz[channel])
    scaled_jaxgb = np.real(scaled_channel(jaxgb, exact))
    scaled_trilinear = np.real(scaled_channel(trilinear, exact))

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.8), sharex=True, constrained_layout=True)
    tt_days = times[window] / 86400.0
    axes[0].plot(tt_days, exact[window], label="exact-link pyTDI", linewidth=1.2)
    axes[0].plot(tt_days, scaled_jaxgb[window], label="scaled JAXGBaccurate", linestyle="--", linewidth=1.0)
    axes[0].plot(tt_days, scaled_trilinear[window], label="scaled trilinear default", linestyle=":", linewidth=1.2)
    axes[0].set_ylabel("Re X")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tt_days, scaled_jaxgb[window] - exact[window], label="JAXGBaccurate - exact", linewidth=1.0)
    axes[1].plot(tt_days, scaled_trilinear[window] - exact[window], label="trilinear - exact", linewidth=1.0)
    axes[1].set_xlabel("time [days]")
    axes[1].set_ylabel("scaled residual")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"X-channel overlay, {orbit_kind}, f0={f0:.1e} Hz")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    duration = 7.0 * 86400.0
    samples = 4096
    dt = duration / samples
    response_bins = 128
    frequencies = [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3]
    overlay_f0 = 1.0e-3
    orbit_kinds = ["equal", "esa-trailing"]

    times = np.linspace(0.0, duration, samples, endpoint=False)
    rows: list[dict[str, object]] = []
    overlay_data_by_orbit = {}

    for orbit_kind in orbit_kinds:
        orbits = default_lisaorbits(orbit_kind)
        state = state_from_lisaorbits(orbits, times)
        jaxgb = JaxGBaccurate(orbits=orbits, t_obs=duration, t0=0.0, n=response_bins, extension=0.5)

        warm = make_eccentric_source(frequencies[0])
        jaxgbaccurate_xyz_time(jaxgb, as_jaxgb_params(warm), dt=dt, obs_samples=samples)
        compute_triliear_xyz(state, as_galactic_binary(warm))

        for f0 in frequencies:
            eccentric_source = make_eccentric_source(f0)
            source = as_galactic_binary(eccentric_source)
            params = as_jaxgb_params(eccentric_source)

            jaxgb_xyz, jaxgb_s = elapsed(lambda p=params: jaxgbaccurate_xyz_time(jaxgb, p, dt=dt, obs_samples=samples))
            trilinear_default_xyz, trilinear_default_s = elapsed(lambda s=source: compute_triliear_xyz(state, s))
            trilinear_exact_transfer_xyz, trilinear_exact_transfer_s = elapsed(
                lambda s=source: compute_triliear_xyz(
                    state,
                    s,
                    order="exact-transfer",
                    generation=1,
                    measurement_order=5,
                    delay_order=5,
                )
            )
            exact_xyz, exact_s = elapsed(lambda s=source: compute_xyz(state, s, generation=1, measurement_order=5, delay_order=5))

            xyz_models = {
                "jaxgbaccurate": jaxgb_xyz,
                "trilinear_default": trilinear_default_xyz,
                "trilinear_exact_transfer": trilinear_exact_transfer_xyz,
            }
            xyz_reference = exact_xyz
            timings = {
                "jaxgbaccurate": jaxgb_s,
                "trilinear_default": trilinear_default_s,
                "trilinear_exact_transfer": trilinear_exact_transfer_s,
            }
            rows.extend(
                compare_basis(
                    orbit_kind=orbit_kind,
                    f0=f0,
                    basis_name="XYZ",
                    models=xyz_models,
                    reference=xyz_reference,
                    timings=timings,
                )
            )

            aet_models = {name: aet_from_xyz(model) for name, model in xyz_models.items()}
            aet_reference = aet_from_xyz(xyz_reference)
            rows.extend(
                compare_basis(
                    orbit_kind=orbit_kind,
                    f0=f0,
                    basis_name="AET",
                    models=aet_models,
                    reference=aet_reference,
                    timings=timings,
                )
            )

            print(
                f"{orbit_kind} f0={f0:.1e} | JAXGBaccurate {jaxgb_s:.4f}s | "
                f"trilinear default {trilinear_default_s:.4f}s | "
                f"trilinear exact-transfer {trilinear_exact_transfer_s:.4f}s | "
                f"exact-link {exact_s:.4f}s"
            )
            if np.isclose(f0, overlay_f0):
                overlay_data_by_orbit[orbit_kind] = (exact_xyz, jaxgb_xyz, trilinear_default_xyz)

    summary_rows = summarize(rows)
    csv_path = out_dir / "jaxgbaccurate_trilinear_circular_comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "jaxgbaccurate_trilinear_circular_comparison_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_path = out_dir / "jaxgbaccurate_trilinear_circular_comparison.png"
    plot_summary(summary_rows, plot_path)

    overlay_path = out_dir / "jaxgbaccurate_trilinear_circular_comparison_overlay_f0_1e-03.png"
    overlay_equal_path = out_dir / "jaxgbaccurate_trilinear_circular_comparison_overlay_equal_f0_1e-03.png"
    overlay_unequal_path = out_dir / "jaxgbaccurate_trilinear_circular_comparison_overlay_esa-trailing_f0_1e-03.png"
    if "equal" in overlay_data_by_orbit:
        exact_xyz, jaxgb_xyz, trilinear_default_xyz = overlay_data_by_orbit["equal"]
        plot_overlay(
            orbit_kind="equal",
            f0=overlay_f0,
            times=times,
            exact_xyz=exact_xyz,
            jaxgb_xyz=jaxgb_xyz,
            trilinear_xyz=trilinear_default_xyz,
            path=overlay_path,
        )
        plot_overlay(
            orbit_kind="equal",
            f0=overlay_f0,
            times=times,
            exact_xyz=exact_xyz,
            jaxgb_xyz=jaxgb_xyz,
            trilinear_xyz=trilinear_default_xyz,
            path=overlay_equal_path,
        )
    if "esa-trailing" in overlay_data_by_orbit:
        exact_xyz, jaxgb_xyz, trilinear_default_xyz = overlay_data_by_orbit["esa-trailing"]
        plot_overlay(
            orbit_kind="esa-trailing",
            f0=overlay_f0,
            times=times,
            exact_xyz=exact_xyz,
            jaxgb_xyz=jaxgb_xyz,
            trilinear_xyz=trilinear_default_xyz,
            path=overlay_unequal_path,
        )

    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {overlay_path}")
    print(f"wrote {overlay_equal_path}")
    print(f"wrote {overlay_unequal_path}")
    print("XYZ median scaled relative L2:")
    for row in summary_rows:
        if row["basis"] == "XYZ":
            print(
                f"  {row['orbit_kind']} f0={float(row['f0_hz']):.1e} {row['model']}: "
                f"{float(row['median_scaled_relative_l2']):.3e}"
            )


if __name__ == "__main__":
    main()
