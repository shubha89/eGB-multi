"""Diagnose the circular JAXGB/JAXGBaccurate convention floor.

This script compares circular JAXGB and JAXGBaccurate outputs against the local
exact-link pyTDI response on the native sparse frequency support returned by the
JAXGB models. It sweeps simple convention variants and channel permutations so
we can separate a real response approximation error from sign, phase, TDI, and
reconstruction mistakes.

Run with:

    python benchmarks/diagnose_jaxgb_convention_floor.py

Outputs:

    benchmarks/diagnose_jaxgb_convention_floor.csv
"""

from __future__ import annotations

import csv
import itertools
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jaxgb.jaxgb import JaxGB, JaxGBaccurate

from egb_multi.constants import C_M_PER_S, G_M3_KG_S2, KILOPARSEC_M, SOLAR_MASS_KG
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.sources import GalacticBinaryParams
from egb_multi.tdi import compute_xyz


CHANNELS = ("X", "Y", "Z")


def elapsed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


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


def make_source(f0: float) -> EccentricBinaryParams:
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


def as_jaxgb_params(
    source: EccentricBinaryParams,
    *,
    phi_sign: int,
    psi_sign: int,
    beta_sign: int,
    lambda_shift: float,
) -> jnp.ndarray:
    return jnp.asarray(
        [
            [
                source.mean_motion / np.pi,
                source.fdot,
                circular_strain_amplitude(source),
                source.lambda_ + lambda_shift,
                beta_sign * source.beta,
                psi_sign * source.psi,
                source.inclination,
                phi_sign * source.phi0,
            ]
        ],
        dtype=jnp.float64,
    )


def dict_from_tdi_tuple(tdi_tuple: tuple[jax.Array, jax.Array, jax.Array]) -> dict[str, np.ndarray]:
    return {channel: np.asarray(value[0], dtype=np.complex128) for channel, value in zip(CHANNELS, tdi_tuple, strict=True)}


def fft_bins(series: np.ndarray, *, kmin: int, bins: int) -> np.ndarray:
    spectrum = np.fft.fft(np.asarray(series, dtype=np.complex128))
    idx = np.arange(kmin, kmin + bins, dtype=np.int64) % spectrum.size
    return spectrum[idx]


def reference_bins(xyz: dict[str, np.ndarray], *, kmin: int, bins: int) -> dict[str, np.ndarray]:
    return {channel: fft_bins(xyz[channel], kmin=kmin, bins=bins) for channel in CHANNELS}


def best_fit_scale(model: np.ndarray, reference: np.ndarray) -> complex:
    denom = np.vdot(model, model)
    return 0.0j if denom == 0.0 else complex(np.vdot(model, reference) / denom)


def residual_with_scales(model: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> tuple[float, dict[str, complex]]:
    numerator = 0.0
    denominator = 0.0
    scales = {}
    for channel in CHANNELS:
        m = np.asarray(model[channel], dtype=np.complex128)
        r = np.asarray(reference[channel], dtype=np.complex128)
        scale = best_fit_scale(m, r)
        scales[channel] = scale
        numerator += np.linalg.norm(scale * m - r) ** 2
        denominator += np.linalg.norm(r) ** 2
    return float(np.sqrt(numerator / denominator)), scales


def best_channel_permutation(model: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> tuple[float, tuple[str, str, str], dict[str, complex]]:
    best = (np.inf, CHANNELS, {})
    for perm in itertools.permutations(CHANNELS):
        permuted = {channel: model[source_channel] for channel, source_channel in zip(CHANNELS, perm, strict=True)}
        residual, scales = residual_with_scales(permuted, reference)
        if residual < best[0]:
            best = (residual, perm, scales)
    return best


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_csv = out_dir / "diagnose_jaxgb_convention_floor.csv"

    duration = 7.0 * 86400.0
    samples = 4096
    response_bins = 128
    extension = 0.5
    f0 = 1.0e-3

    source = make_source(f0)
    gb_source = as_galactic_binary(source)
    orbits = default_lisaorbits("equal")

    times = np.linspace(0.0, duration, samples, endpoint=False)
    state = state_from_lisaorbits(orbits, times)
    refs = {}
    for generation in (1, 2):
        refs[("JaxGB", generation)] = compute_xyz(state, gb_source, generation=generation, measurement_order=5, delay_order=5)

    ext_duration = duration * (1.0 + 2.0 * extension)
    ext_samples = int(round(samples * (1.0 + 2.0 * extension)))
    ext_times = np.linspace(-extension * duration, duration + extension * duration, ext_samples, endpoint=False)
    ext_state = state_from_lisaorbits(orbits, ext_times)
    for generation in (1, 2):
        refs[("JaxGBaccurate", generation)] = compute_xyz(ext_state, gb_source, generation=generation, measurement_order=5, delay_order=5)

    models = {
        "JaxGB": JaxGB(orbits=orbits, t_obs=duration, t0=0.0, n=response_bins, window="boxcar"),
        "JaxGBaccurate": JaxGBaccurate(
            orbits=orbits,
            t_obs=duration,
            t0=0.0,
            n=response_bins,
            extension=extension,
            window="boxcar",
        ),
    }

    variants = []
    for phi_sign, psi_sign, beta_sign, lambda_shift in itertools.product(
        (1, -1),
        (1, -1),
        (1, -1),
        (0.0, np.pi),
    ):
        variants.append(
            {
                "phi_sign": phi_sign,
                "psi_sign": psi_sign,
                "beta_sign": beta_sign,
                "lambda_shift": lambda_shift,
            }
        )

    rows = []
    for model_name, model in models.items():
        for variant in variants:
            params = as_jaxgb_params(source, **variant)
            kmin = int(np.asarray(model.get_kmin(params[:, 0]))[0])
            for tdi_generation, local_generation in ((1.5, 1), (2, 2)):
                tdi, wall_s = elapsed(lambda m=model, p=params, g=tdi_generation: m.get_tdi(p, tdi_generation=g, tdi_combination="XYZ"))
                model_bins = dict_from_tdi_tuple(tdi)
                ref_bins = reference_bins(refs[(model_name, local_generation)], kmin=kmin, bins=response_bins)
                direct_residual, direct_scales = residual_with_scales(model_bins, ref_bins)
                best_residual, permutation, best_scales = best_channel_permutation(model_bins, ref_bins)
                rows.append(
                    {
                        "model": model_name,
                        "f0_hz": f0,
                        "duration_days": duration / 86400.0,
                        "samples": samples,
                        "response_bins": response_bins,
                        "extension": extension if model_name == "JaxGBaccurate" else 0.0,
                        "tdi_generation": tdi_generation,
                        "local_generation": local_generation,
                        "kmin": kmin,
                        **variant,
                        "wall_s": wall_s,
                        "direct_scaled_residual": direct_residual,
                        "best_permuted_scaled_residual": best_residual,
                        "best_permutation": "".join(permutation),
                        "direct_scale_abs_X": abs(direct_scales["X"]),
                        "direct_scale_phase_X": float(np.angle(direct_scales["X"])),
                        "best_scale_abs_X": abs(best_scales["X"]),
                        "best_scale_phase_X": float(np.angle(best_scales["X"])),
                    }
                )

    rows = sorted(rows, key=lambda row: (row["model"], row["best_permuted_scaled_residual"]))
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out_csv}")
    for model_name in ("JaxGB", "JaxGBaccurate"):
        print(f"\n{model_name} best rows:")
        for row in [r for r in rows if r["model"] == model_name][:8]:
            print(
                f"  residual={row['best_permuted_scaled_residual']:.3e} "
                f"direct={row['direct_scaled_residual']:.3e} "
                f"tdi={row['tdi_generation']} local={row['local_generation']} "
                f"perm={row['best_permutation']} "
                f"phi={row['phi_sign']} psi={row['psi_sign']} beta={row['beta_sign']} "
                f"lambda_shift={row['lambda_shift']:.3f}"
            )


if __name__ == "__main__":
    main()
