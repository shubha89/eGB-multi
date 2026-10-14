"""Compare JAXGB circular responses with local brute-force links.

This diagnostic has two roles:

1. Check the circular limit of the local eccentric time-domain response against
   JAXGB.
2. Quantify how the circular JAXGB response fails as the local signal is made
   eccentric.

The comparison is performed on the same narrow Fourier support returned by
JAXGB. We report both raw residuals and residuals after one best-fit complex
normalization per channel, because the local circular amplitude convention is
not identical to the public JAXGB/FastGB convention.

Run with:

    python benchmarks/jaxgb_bruteforce_circular_eccentric_comparison.py

Outputs:

    benchmarks/jaxgb_bruteforce_circular_eccentric_comparison.csv
    benchmarks/jaxgb_bruteforce_circular_eccentric_comparison.png
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

from jaxgb.jaxgb import JaxGB

from egb_multi.constants import C_M_PER_S, G_M3_KG_S2, KILOPARSEC_M, SOLAR_MASS_KG
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.tdi import pytdi_data_from_links


def elapsed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def circular_strain_amplitude(source: EccentricBinaryParams) -> float:
    """Circular quadrupole amplitude compatible with the local mass model."""

    total_mass = (source.m1_solar + source.m2_solar) * SOLAR_MASS_KG
    eta = source.symmetric_mass_ratio
    return float(
        2.0
        * eta
        * (G_M3_KG_S2 * total_mass) ** (5.0 / 3.0)
        * source.mean_motion ** (2.0 / 3.0)
        / (C_M_PER_S**4 * source.distance_m)
    )


def make_eccentric_source(f0: float, eccentricity: float) -> EccentricBinaryParams:
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


def as_jaxgb_params(source: EccentricBinaryParams) -> jnp.ndarray:
    # For this apples-to-apples diagnostic we pass the same angular coordinates
    # used by the local response. A separate convention test showed that this
    # gives the smallest e=0 residual for the current lisaorbits/local setup.
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


def xyz_from_eccentric_links(state, links: dict[str, np.ndarray], *, generation: int) -> dict[str, np.ndarray]:
    from pytdi.michelson import compute_factorized_michelson

    data = pytdi_data_from_links(state, links)
    return {
        "X": compute_factorized_michelson(data, rot=0, order=5, delay_order=5, generation=generation, unit="frequency"),
        "Y": compute_factorized_michelson(data, rot=1, order=5, delay_order=5, generation=generation, unit="frequency"),
        "Z": compute_factorized_michelson(data, rot=2, order=5, delay_order=5, generation=generation, unit="frequency"),
    }


def fft_slice(series: np.ndarray, *, kmin: int, bins: int) -> np.ndarray:
    spectrum = np.fft.fft(np.asarray(series, dtype=np.complex128))
    indices = np.arange(kmin, kmin + bins, dtype=np.int64) % spectrum.size
    return spectrum[indices]


def fft_slices_from_xyz(xyz: dict[str, np.ndarray], *, kmin: int, bins: int) -> dict[str, np.ndarray]:
    return {channel: fft_slice(xyz[channel], kmin=kmin, bins=bins) for channel in ("X", "Y", "Z")}


def fft_full_from_xyz(xyz: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {channel: np.fft.fft(np.asarray(xyz[channel], dtype=np.complex128)) for channel in ("X", "Y", "Z")}


def sparse_full_spectrum(model: dict[str, np.ndarray], *, kmin: int, full_bins: int) -> dict[str, np.ndarray]:
    out = {}
    for channel in ("X", "Y", "Z"):
        spectrum = np.zeros(full_bins, dtype=np.complex128)
        bins = np.asarray(model[channel], dtype=np.complex128)
        indices = np.arange(kmin, kmin + bins.size, dtype=np.int64) % full_bins
        spectrum[indices] = bins
        out[channel] = spectrum
    return out


def best_fit_scale(model: np.ndarray, reference: np.ndarray) -> complex:
    denom = np.vdot(model, model)
    return 0.0j if denom == 0.0 else complex(np.vdot(model, reference) / denom)


def combined_xyz_residual(
    model: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    scaled: bool,
) -> tuple[float, dict[str, complex]]:
    numerator = 0.0
    denominator = 0.0
    scales: dict[str, complex] = {}
    for channel in ("X", "Y", "Z"):
        model_channel = np.asarray(model[channel], dtype=np.complex128)
        reference_channel = np.asarray(reference[channel], dtype=np.complex128)
        scale = best_fit_scale(model_channel, reference_channel) if scaled else 1.0 + 0.0j
        scales[channel] = scale
        residual = scale * model_channel - reference_channel
        numerator += np.linalg.norm(residual) ** 2
        denominator += np.linalg.norm(reference_channel) ** 2
    return float(np.sqrt(numerator / denominator)), scales


def jaxgb_dict_from_tuple(tdi_tuple: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> dict[str, np.ndarray]:
    return {
        "X": np.asarray(tdi_tuple[0][0], dtype=np.complex128),
        "Y": np.asarray(tdi_tuple[1][0], dtype=np.complex128),
        "Z": np.asarray(tdi_tuple[2][0], dtype=np.complex128),
    }


def add_comparison_rows(
    rows: list[dict[str, object]],
    *,
    f0: float,
    eccentricity: float,
    method: str,
    model: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    wall_s: float,
    brute_wall_s: float,
    kmin: int,
    bins: int,
    duration: float,
    samples: int,
    dt: float,
) -> None:
    raw, _ = combined_xyz_residual(model, reference, scaled=False)
    scaled, scales = combined_xyz_residual(model, reference, scaled=True)
    rows.append(
        {
            "f0_hz": f0,
            "eccentricity": eccentricity,
            "method": method,
            "duration_days": duration / 86400.0,
            "samples": samples,
            "dt_s": dt,
            "comparison_kmin": kmin,
            "comparison_bins": bins,
            "bruteforce_xyz_s": brute_wall_s,
            "model_xyz_s": wall_s,
            "raw_residual_xyz": raw,
            "scaled_residual_xyz": scaled,
            "scale_abs_X": abs(scales["X"]),
            "scale_phase_X": float(np.angle(scales["X"])),
        }
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    duration = 7.0 * 24.0 * 3600.0
    samples = 4096
    dt = duration / samples
    response_bins = 128
    frequencies = [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3]
    eccentricities = [0.0, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 0.1, 0.3]

    orbits = default_lisaorbits("equal")
    times = np.linspace(0.0, duration, samples, endpoint=False)
    state = state_from_lisaorbits(orbits, times)
    geometry = precompute_jax_link_geometry(state)

    jaxgb = JaxGB(orbits=orbits, t_obs=duration, t0=0.0, n=response_bins, window="boxcar")

    warm = make_eccentric_source(frequencies[0], 0.0)
    eccentric_links_jax(warm, geometry, batch_size=1, physics_mode="newtonian")
    jaxgb.get_tdi(as_jaxgb_params(warm), tdi_generation=1.5, tdi_combination="XYZ")

    rows: list[dict[str, object]] = []
    for f0 in frequencies:
        circular = make_eccentric_source(f0, 0.0)
        params = as_jaxgb_params(circular)

        jaxgb_tdi, jaxgb_s = elapsed(lambda: jaxgb.get_tdi(params, tdi_generation=1.5, tdi_combination="XYZ"))
        kmin = int(np.asarray(jaxgb.get_kmin(params[:, 0]))[0])
        bins = response_bins
        jaxgb_xyz = jaxgb_dict_from_tuple(jaxgb_tdi)
        jaxgb_full = sparse_full_spectrum(jaxgb_xyz, kmin=kmin, full_bins=samples)

        eccentric_zero_full = None
        for eccentricity in eccentricities:
            ecc_source = make_eccentric_source(f0, eccentricity)
            links, link_s = elapsed(lambda: eccentric_links_jax(ecc_source, geometry, batch_size=1, physics_mode="newtonian"))
            ecc_xyz, tdi_s = elapsed(lambda: xyz_from_eccentric_links(state, links, generation=1))
            ecc_bins = fft_slices_from_xyz(ecc_xyz, kmin=kmin, bins=bins)
            ecc_full = fft_full_from_xyz(ecc_xyz)
            brute_s = link_s + tdi_s
            if eccentricity == 0.0:
                eccentric_zero_full = ecc_full
            add_comparison_rows(
                rows,
                f0=f0,
                eccentricity=eccentricity,
                method="JAXGB circular vs brute eccentric common support",
                model=jaxgb_xyz,
                reference=ecc_bins,
                wall_s=jaxgb_s,
                brute_wall_s=brute_s,
                kmin=kmin,
                bins=bins,
                duration=duration,
                samples=samples,
                dt=dt,
            )
            add_comparison_rows(
                rows,
                f0=f0,
                eccentricity=eccentricity,
                method="JAXGB circular vs brute eccentric full sparse",
                model=jaxgb_full,
                reference=ecc_full,
                wall_s=jaxgb_s,
                brute_wall_s=brute_s,
                kmin=kmin,
                bins=bins,
                duration=duration,
                samples=samples,
                dt=dt,
            )
            if eccentricity > 0.0:
                if eccentric_zero_full is None:
                    raise RuntimeError("e=0 reference was not computed before eccentric cases")
                add_comparison_rows(
                    rows,
                    f0=f0,
                    eccentricity=eccentricity,
                    method="brute eccentric vs brute e=0 full spectrum",
                    model=eccentric_zero_full,
                    reference=ecc_full,
                    wall_s=0.0,
                    brute_wall_s=brute_s,
                    kmin=kmin,
                    bins=bins,
                    duration=duration,
                    samples=samples,
                    dt=dt,
                )
            common_residual = rows[-2]["scaled_residual_xyz"] if eccentricity == 0.0 else rows[-3]["scaled_residual_xyz"]
            full_residual = rows[-1]["scaled_residual_xyz"] if eccentricity == 0.0 else rows[-2]["scaled_residual_xyz"]
            print(
                f"f0={f0:.1e} e={eccentricity:.1e} | "
                f"common residual {common_residual:.3e} | "
                f"full sparse residual {full_residual:.3e}",
                flush=True,
            )

    csv_path = out_dir / "jaxgb_bruteforce_circular_eccentric_comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = out_dir / "jaxgb_bruteforce_circular_eccentric_comparison.png"
    plot(rows, plot_path)
    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")


def plot(rows: list[dict[str, object]], path: Path) -> None:
    methods = [
        "JAXGB circular vs brute eccentric common support",
        "JAXGB circular vs brute eccentric full sparse",
        "brute eccentric vs brute e=0 full spectrum",
    ]
    colors = {
        1.0e-4: "#0072B2",
        3.0e-4: "#009E73",
        1.0e-3: "#D55E00",
        3.0e-3: "#CC79A7",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)

    for method in methods[:2]:
        subset = [row for row in rows if row["method"] == method and np.isclose(float(row["eccentricity"]), 0.0)]
        marker = "o" if "common" in method else "s"
        label = "common support" if "common" in method else "full sparse spectrum"
        axes[0].plot(
            [row["f0_hz"] for row in subset],
            [max(float(row["scaled_residual_xyz"]), 1.0e-16) for row in subset],
            marker=marker,
            label=label,
        )

    for f0, color in colors.items():
        for method, marker, ls in [
            ("JAXGB circular vs brute eccentric common support", "o", "-"),
            ("JAXGB circular vs brute eccentric full sparse", "s", "--"),
            ("brute eccentric vs brute e=0 full spectrum", "^", ":"),
        ]:
            subset = [row for row in rows if row["method"] == method and np.isclose(float(row["f0_hz"]), f0)]
            subset = [row for row in subset if float(row["eccentricity"]) > 0.0]
            subset = sorted(subset, key=lambda row: float(row["eccentricity"]))
            axes[1].plot(
                [row["eccentricity"] for row in subset],
                [max(float(row["scaled_residual_xyz"]), 1.0e-16) for row in subset],
                marker=marker,
                linestyle=ls,
                color=color,
                label=f"{f0:.0e} Hz {'common' if 'common' in method else ('local e0' if 'brute e=0' in method else 'full')}",
            )

    timing_rows = [row for row in rows if row["method"] == "JAXGB circular vs brute eccentric common support" and np.isclose(float(row["eccentricity"]), 0.0)]
    axes[2].plot([row["f0_hz"] for row in timing_rows], [row["model_xyz_s"] for row in timing_rows], marker="o", label="JAXGB")
    axes[2].plot([row["f0_hz"] for row in timing_rows], [row["bruteforce_xyz_s"] for row in timing_rows], marker="D", label="brute e=0 + pyTDI")

    ecc_timing = [row for row in rows if row["method"] == "JAXGB circular vs brute eccentric common support" and np.isclose(float(row["eccentricity"]), 0.1)]
    axes[2].plot([row["f0_hz"] for row in ecc_timing], [row["bruteforce_xyz_s"] for row in ecc_timing], marker="^", label="local e=0.1 brute + pyTDI")

    axes[0].set_title("circular-limit cross-check")
    axes[0].set_ylabel("best-fit scaled XYZ residual")
    axes[1].set_title("circular JAXGB vs eccentric brute force")
    axes[1].set_ylabel("best-fit scaled XYZ residual")
    axes[2].set_title("wall times")
    axes[2].set_ylabel("seconds")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_xlabel("frequency [Hz]")
    axes[1].set_xlabel("eccentricity")
    axes[2].set_xlabel("frequency [Hz]")
    axes[1].set_xlim(8.0e-5, 0.4)

    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
