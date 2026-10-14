"""JAXGBaccurate calibrated circular template vs local eccentric response.

The JAXGBaccurate response is generated in its native heterodyned,
windowed frequency-domain representation, transformed back to the observation
interval, and calibrated channel-by-channel to the local e=0 brute-force
response. The calibrated circular template is then compared with local
eccentric time-domain responses.

Outputs:

    benchmarks/jaxgbaccurate_calibrated_eccentric_study.csv
    benchmarks/jaxgbaccurate_calibrated_eccentric_study.png
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jaxgb.jaxgb import JaxGBaccurate

from production_eccentric_circular_mismatch_grid import compare_aet, load_aet_psd

from egb_multi.constants import C_M_PER_S, G_M3_KG_S2, KILOPARSEC_M, SOLAR_MASS_KG
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.noise import aet_from_xyz
from egb_multi.tdi import pytdi_data_from_links


def elapsed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--physics-mode",
        choices=("newtonian", "1pn_no_periastron", "1pn"),
        default="newtonian",
        help="Eccentric source physics mode used for the local brute-force reference.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix inserted before .csv/.png so reruns do not overwrite existing artifacts.",
    )
    parser.add_argument(
        "--frequencies",
        nargs="+",
        type=float,
        default=None,
        help="Carrier frequencies in Hz. Defaults to the paper comparison grid.",
    )
    parser.add_argument(
        "--eccentricities",
        nargs="+",
        type=float,
        default=None,
        help="Eccentricities to compare. Defaults to the paper comparison grid.",
    )
    return parser.parse_args()


def make_source(f0: float, eccentricity: float) -> EccentricBinaryParams:
    return EccentricBinaryParams(
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


def xyz_from_links(state, links: dict[str, np.ndarray], *, generation: int = 1) -> dict[str, np.ndarray]:
    from pytdi.michelson import compute_factorized_michelson

    data = pytdi_data_from_links(state, links)
    return {
        "X": compute_factorized_michelson(data, rot=0, order=5, delay_order=5, generation=generation, unit="frequency"),
        "Y": compute_factorized_michelson(data, rot=1, order=5, delay_order=5, generation=generation, unit="frequency"),
        "Z": compute_factorized_michelson(data, rot=2, order=5, delay_order=5, generation=generation, unit="frequency"),
    }


def jaxgbaccurate_xyz_time(model: JaxGBaccurate, params: jnp.ndarray, *, dt: float, obs_samples: int) -> dict[str, np.ndarray]:
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


def calibrate_template(template: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, complex]]:
    calibrated = {}
    scales = {}
    for channel in ("A", "E", "T"):
        scale = best_fit_scale(np.asarray(template[channel]), np.asarray(reference[channel]))
        scales[channel] = scale
        calibrated[channel] = np.real(scale * np.asarray(template[channel]))
    return calibrated, scales


def main() -> None:
    args = parse_args()
    physics_mode = args.physics_mode
    out_dir = Path(__file__).resolve().parent
    duration = 7.0 * 24.0 * 3600.0
    samples = 4096
    dt = duration / samples
    response_bins = 128
    frequencies = args.frequencies or [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3]
    eccentricities = args.eccentricities or [0.0, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 0.1, 0.3]

    orbits = default_lisaorbits("equal")
    times = np.linspace(0.0, duration, samples, endpoint=False)
    state = state_from_lisaorbits(orbits, times)
    geometry = precompute_jax_link_geometry(state)
    jaxgb = JaxGBaccurate(orbits=orbits, t_obs=duration, t0=0.0, n=response_bins, extension=0.5)

    psd_freq, psd_base = load_aet_psd(out_dir / "lisainstrument_welch_psd.csv")
    freq = np.fft.rfftfreq(samples, dt)
    psd = {channel: np.interp(freq, psd_freq, psd_base[channel], left=np.inf, right=np.inf) for channel in ("A", "E", "T")}
    frequency_mask = (freq >= 1.0e-5) & (freq <= 1.0e-2)

    warm = make_source(frequencies[0], 0.0)
    eccentric_links_jax(warm, geometry, batch_size=1, physics_mode=physics_mode)
    jaxgbaccurate_xyz_time(jaxgb, as_jaxgb_params(warm), dt=dt, obs_samples=samples)

    rows = []
    for f0 in frequencies:
        e0_source = make_source(f0, 0.0)
        circular_template_xyz, jaxgb_s = elapsed(
            lambda: jaxgbaccurate_xyz_time(jaxgb, as_jaxgb_params(e0_source), dt=dt, obs_samples=samples)
        )
        circular_template_aet = aet_from_xyz(circular_template_xyz)

        e0_links, e0_link_s = elapsed(lambda: eccentric_links_jax(e0_source, geometry, batch_size=1, physics_mode=physics_mode))
        e0_xyz, e0_tdi_s = elapsed(lambda: xyz_from_links(state, e0_links))
        e0_aet = aet_from_xyz(e0_xyz)
        calibrated_template_aet, scales = calibrate_template(circular_template_aet, e0_aet)

        for eccentricity in eccentricities:
            source = make_source(f0, eccentricity)
            links, link_s = elapsed(lambda: eccentric_links_jax(source, geometry, batch_size=1, physics_mode=physics_mode))
            xyz, tdi_s = elapsed(lambda: xyz_from_links(state, links))
            eccentric_aet = aet_from_xyz(xyz)
            metrics = compare_aet(eccentric_aet, calibrated_template_aet, psd, dt, frequency_mask)
            local_circular_metrics = compare_aet(eccentric_aet, e0_aet, psd, dt, frequency_mask)
            row = {
                "f0_hz": f0,
                "eccentricity": eccentricity,
                "duration_days": duration / 86400.0,
                "samples": samples,
                "dt_s": dt,
                "physics_mode": physics_mode,
                "jaxgbaccurate_xyz_s": jaxgb_s,
                "bruteforce_links_s": link_s,
                "bruteforce_tdi_s": tdi_s,
                "bruteforce_xyz_s": link_s + tdi_s,
                "calibration_e0_bruteforce_xyz_s": e0_link_s + e0_tdi_s,
                "scale_abs_A": abs(scales["A"]),
                "scale_phase_A": float(np.angle(scales["A"])),
                "local_e0_mismatch_AE": local_circular_metrics["mismatch_AE"],
                "local_e0_noise_weighted_residual_AE": local_circular_metrics["noise_weighted_residual_AE"],
                "local_e0_relative_l2_residual_AE": local_circular_metrics["relative_l2_residual_AE"],
                **metrics,
            }
            rows.append(row)
            print(
                f"f0={f0:.1e} e={eccentricity:.1e} | "
                f"mismatch={row['mismatch_AE']:.3e} | "
                f"noise residual={row['noise_weighted_residual_AE']:.3e} | "
                f"times JAXGBacc/brute={jaxgb_s:.4f}/{link_s + tdi_s:.4f}s",
                flush=True,
            )

    add_convention_removed_metrics(rows)

    csv_path = out_dir / f"jaxgbaccurate_calibrated_eccentric_study{args.output_suffix}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    png_path = out_dir / f"jaxgbaccurate_calibrated_eccentric_study{args.output_suffix}.png"
    plot(rows, png_path)
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


def median_by_eccentricity(rows: list[dict[str, object]], key: str) -> tuple[np.ndarray, np.ndarray]:
    eccentricities = sorted({float(row["eccentricity"]) for row in rows if float(row["eccentricity"]) > 0.0})
    x = []
    y = []
    for eccentricity in eccentricities:
        subset = [row for row in rows if np.isclose(float(row["eccentricity"]), eccentricity)]
        best = float(np.median([float(row[key]) for row in subset]))
        x.append(eccentricity)
        y.append(best)
    return np.asarray(x), np.asarray(y)


def add_convention_removed_metrics(rows: list[dict[str, object]]) -> None:
    floors = {}
    for row in rows:
        if np.isclose(float(row["eccentricity"]), 0.0):
            floors[float(row["f0_hz"])] = (
                float(row["mismatch_AE"]),
                float(row["noise_weighted_residual_AE"]),
                float(row["relative_l2_residual_AE"]),
            )
    for row in rows:
        f0 = float(row["f0_hz"])
        mismatch_floor, noise_floor, l2_floor = floors[f0]
        row["mismatch_AE_convention_removed"] = max(float(row["mismatch_AE"]) - mismatch_floor, 0.0)
        row["noise_weighted_residual_AE_convention_removed"] = float(
            np.sqrt(max(float(row["noise_weighted_residual_AE"]) ** 2 - noise_floor**2, 0.0))
        )
        row["relative_l2_residual_AE_convention_removed"] = float(
            np.sqrt(max(float(row["relative_l2_residual_AE"]) ** 2 - l2_floor**2, 0.0))
        )


def plot(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    colors = {
        1.0e-4: "#0072B2",
        3.0e-4: "#009E73",
        1.0e-3: "#D55E00",
        3.0e-3: "#CC79A7",
    }
    for f0, color in colors.items():
        subset = [
            row
            for row in rows
            if np.isclose(float(row["f0_hz"]), f0) and float(row["eccentricity"]) > 0.0
        ]
        subset = sorted(subset, key=lambda row: float(row["eccentricity"]))
        x = np.array([float(row["eccentricity"]) for row in subset])
        mismatch = np.array([float(row["local_e0_mismatch_AE"]) for row in subset])
        residual = np.array([float(row["local_e0_noise_weighted_residual_AE"]) for row in subset])
        axes[0].plot(x, np.maximum(mismatch, 1.0e-12), marker="o", color=color, label=f"{f0:.0e} Hz")
        axes[1].plot(x, np.maximum(residual, 1.0e-12), marker="o", color=color, label=f"{f0:.0e} Hz")

    e0_rows = [row for row in rows if np.isclose(float(row["eccentricity"]), 0.0)]
    f0 = np.array([float(row["f0_hz"]) for row in e0_rows])
    axes[2].plot(f0, [float(row["jaxgbaccurate_xyz_s"]) for row in e0_rows], marker="o", label="JAXGBaccurate circular")
    axes[2].plot(f0, [float(row["bruteforce_xyz_s"]) for row in e0_rows], marker="s", label="brute e=0 + pyTDI")
    e01_rows = [row for row in rows if np.isclose(float(row["eccentricity"]), 0.1)]
    axes[2].plot(
        [float(row["f0_hz"]) for row in e01_rows],
        [float(row["bruteforce_xyz_s"]) for row in e01_rows],
        marker="^",
        label="brute e=0.1 + pyTDI",
    )

    axes[0].set_ylabel("mismatch")
    axes[1].set_ylabel("relative noise-weighted residual")
    axes[2].set_ylabel("wall time [s]")
    axes[0].set_xlabel("eccentricity")
    axes[1].set_xlabel("eccentricity")
    axes[2].set_xlabel("frequency [Hz]")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
