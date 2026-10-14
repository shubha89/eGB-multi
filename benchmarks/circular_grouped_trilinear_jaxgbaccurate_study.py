"""Circular wall-time and accuracy study for grouped trilinear response.

The study compares, at e=0:

1. JAX brute-force eccentric links + pyTDI, used as the reference.
2. Grouped JAX harmonic trilinear beta links + pyTDI.
3. Grouped JAX harmonic trilinear exact-transfer links + pyTDI.
4. JAXGBaccurate circular response, reconstructed with a complex sparse IFFT.

JAXGBaccurate returns an analytic/windowed frequency-domain response.  For an
apples-to-apples comparison to the real eccentric-JAX circular limit, the
JAXGBaccurate source parameters are mapped to the eccentric circular convention
and its reconstructed real part is compared to the real brute-force time series.

Run with:

    python benchmarks/circular_grouped_trilinear_jaxgbaccurate_study.py

Outputs:

    benchmarks/circular_grouped_trilinear_jaxgbaccurate_study.csv
    benchmarks/circular_grouped_trilinear_jaxgbaccurate_study.png
    reports/circular_grouped_trilinear_jaxgbaccurate_study/
"""

from __future__ import annotations

import csv
import shutil
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
from egb_multi.eccentric_jax import eccentric_harmonic_links_grouped_jax, eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.tdi import pytdi_data_from_links


CHANNELS = ("X", "Y", "Z")
MODEL_LABELS = {
    "brute_force_jax": "JAX brute force",
    "grouped_trilinear_beta": "grouped trilinear beta",
    "grouped_trilinear_exact_transfer": "grouped trilinear exact-transfer",
    "jaxgbaccurate": "JAXGBaccurate",
}


def make_sources(n_sources: int, *, seed: int = 1234) -> list[EccentricBinaryParams]:
    rng = np.random.default_rng(seed)
    return [
        EccentricBinaryParams(
            mean_motion=float(np.pi * rng.uniform(1.0e-4, 3.0e-3)),
            eccentricity=0.0,
            m1_solar=float(rng.uniform(0.35, 0.8)),
            m2_solar=float(rng.uniform(0.25, 0.8)),
            distance_m=KILOPARSEC_M,
            beta=float(rng.uniform(-0.9, 0.9)),
            lambda_=float(rng.uniform(-np.pi, np.pi)),
            psi=float(rng.uniform(0.0, np.pi)),
            inclination=float(rng.uniform(0.0, np.pi)),
            phi0=float(rng.uniform(0.0, 2.0 * np.pi)),
            fdot=0.0,
        )
        for _ in range(n_sources)
    ]


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


def as_jaxgb_params(sources: list[EccentricBinaryParams]) -> jnp.ndarray:
    return jnp.asarray(
        [
            [
                source.mean_motion / np.pi,
                source.fdot,
                circular_strain_amplitude(source),
                source.lambda_,
                source.beta,
                source.psi,
                np.pi - source.inclination,
                np.pi - 2.0 * source.phi0,
            ]
            for source in sources
        ],
        dtype=jnp.float64,
    )


def median_elapsed_value(func, repeats: int):
    timings = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = func()
        timings.append(time.perf_counter() - start)
    return value, float(np.median(timings))


def xyz_from_links(state, links: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from pytdi.michelson import compute_factorized_michelson

    data = pytdi_data_from_links(state, links)
    return {
        "X": compute_factorized_michelson(data, rot=0, order=3, delay_order=3, generation=1, unit="frequency"),
        "Y": compute_factorized_michelson(data, rot=1, order=3, delay_order=3, generation=1, unit="frequency"),
        "Z": compute_factorized_michelson(data, rot=2, order=3, delay_order=3, generation=1, unit="frequency"),
    }


def jaxgbaccurate_xyz_time_sum(
    model: JaxGBaccurate,
    params: jnp.ndarray,
    *,
    dt: float,
    obs_samples: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, float, float]:
    start = time.perf_counter()
    tdi = model.sum_tdi(params, tdi_generation=1.5, tdi_combination="XYZ")
    fd_s = time.perf_counter() - start

    start = time.perf_counter()
    kmins = np.asarray(model.get_kmin(params[:, 0]))
    kmin = int(np.min(kmins))
    target_length = int(np.rint(model.t_ext / dt))
    crop_start = int(np.rint(model.extension * model.t_obs / dt))
    out = {}
    fd = {}
    for channel, bins in zip(CHANNELS, tdi, strict=True):
        values = np.asarray(bins, dtype=np.complex128).reshape(-1)
        fd[channel] = values
        spectrum = np.zeros(target_length, dtype=np.complex128)
        indices = np.arange(kmin, kmin + values.size, dtype=np.int64) % target_length
        spectrum[indices] = values
        time_series = np.fft.ifft(spectrum / dt, n=target_length)
        out[channel] = time_series[crop_start : crop_start + obs_samples]
    recon_s = time.perf_counter() - start
    return out, fd, kmin, fd_s, recon_s


def best_fit_scale(model: np.ndarray, reference: np.ndarray) -> complex:
    denom = np.vdot(model, model)
    return 0.0j if denom == 0.0 else complex(np.vdot(model, reference) / denom)


def combined_xyz_residual(
    model: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    scaled: bool,
    real_projection: bool = False,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for channel in CHANNELS:
        model_channel = np.asarray(model[channel], dtype=np.complex128)
        reference_channel = np.asarray(reference[channel], dtype=np.complex128)
        if real_projection:
            model_channel = np.real(model_channel).astype(np.complex128)
            reference_channel = np.real(reference_channel).astype(np.complex128)
        if scaled:
            model_channel = best_fit_scale(model_channel, reference_channel) * model_channel
        numerator += np.linalg.norm(model_channel - reference_channel) ** 2
        denominator += np.linalg.norm(reference_channel) ** 2
    return float(np.sqrt(numerator / denominator)) if denominator > 0.0 else np.nan


def fft_common_support(
    xyz: dict[str, np.ndarray],
    *,
    kmin: int,
    bins: int,
    dt: float,
) -> dict[str, np.ndarray]:
    out = {}
    for channel in CHANNELS:
        spectrum = np.fft.fft(np.asarray(xyz[channel], dtype=np.complex128)) * dt
        indices = np.arange(kmin, kmin + bins, dtype=np.int64) % spectrum.size
        out[channel] = spectrum[indices]
    return out


def add_row(
    rows: list[dict[str, object]],
    *,
    source_count: int,
    model_name: str,
    total_s: float,
    link_or_fd_s: float,
    tdi_or_recon_s: float,
    model_xyz: dict[str, np.ndarray],
    reference_xyz: dict[str, np.ndarray],
    model_fd: dict[str, np.ndarray],
    reference_fd: dict[str, np.ndarray],
    duration_days: float,
    samples: int,
    dt: float,
    orbit_kind: str,
    response_bins: int,
    fd_comparable: bool = True,
) -> None:
    raw_fd = combined_xyz_residual(model_fd, reference_fd, scaled=False) if fd_comparable else np.nan
    scaled_fd = combined_xyz_residual(model_fd, reference_fd, scaled=True) if fd_comparable else np.nan
    rows.append(
        {
            "sources": source_count,
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "orbit_kind": orbit_kind,
            "duration_days": duration_days,
            "samples": samples,
            "dt_s": dt,
            "jaxgbaccurate_response_bins": response_bins if model_name == "jaxgbaccurate" else "",
            "wall_s": total_s,
            "link_or_fd_s": link_or_fd_s,
            "tdi_or_reconstruction_s": tdi_or_recon_s,
            "raw_relative_l2_xyz_time": combined_xyz_residual(model_xyz, reference_xyz, scaled=False),
            "scaled_relative_l2_xyz_time": combined_xyz_residual(model_xyz, reference_xyz, scaled=True),
            "raw_relative_l2_xyz_time_real": combined_xyz_residual(model_xyz, reference_xyz, scaled=False, real_projection=True),
            "scaled_relative_l2_xyz_time_real": combined_xyz_residual(model_xyz, reference_xyz, scaled=True, real_projection=True),
            "raw_relative_l2_xyz_common_fd": raw_fd,
            "scaled_relative_l2_xyz_common_fd": scaled_fd,
        }
    )


def plot(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)
    markers = {
        "brute_force_jax": "o",
        "grouped_trilinear_beta": "D",
        "grouped_trilinear_exact_transfer": "^",
        "jaxgbaccurate": "s",
    }
    for model_name, label in MODEL_LABELS.items():
        subset = sorted([row for row in rows if row["model"] == model_name], key=lambda row: int(row["sources"]))
        x = np.array([int(row["sources"]) for row in subset])
        wall = np.array([float(row["wall_s"]) for row in subset])
        axes[0].plot(x, wall, marker=markers[model_name], label=label)
        if model_name != "brute_force_jax":
            residual = np.array([float(row["scaled_relative_l2_xyz_time_real"]) for row in subset])
            axes[1].plot(x, residual, marker=markers[model_name], label=label)

    axes[0].set_title("wall time")
    axes[1].set_title("accuracy vs JAX brute force")
    axes[0].set_ylabel("wall time [s]")
    axes[1].set_ylabel("scaled relative L2 in Re(XYZ), time domain")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("number of circular sources")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict[str, object]]) -> str:
    source_counts = sorted({int(row["sources"]) for row in rows})
    by_key = {(int(row["sources"]), str(row["model"])): row for row in rows}
    lines = [
        "| sources | brute [s] | tri beta [s] | beta speedup | beta residual | tri exact [s] | exact speedup | exact residual | JAXGBacc [s] | JAXGBacc speedup | JAXGBacc residual |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for count in source_counts:
        brute = float(by_key[(count, "brute_force_jax")]["wall_s"])
        beta = by_key[(count, "grouped_trilinear_beta")]
        exact = by_key[(count, "grouped_trilinear_exact_transfer")]
        jaxgb = by_key[(count, "jaxgbaccurate")]
        beta_wall = float(beta["wall_s"])
        exact_wall = float(exact["wall_s"])
        jaxgb_wall = float(jaxgb["wall_s"])
        lines.append(
            "| "
            f"{count} | "
            f"{brute:.4f} | "
            f"{beta_wall:.4f} | "
            f"{brute / beta_wall:.2f}x | "
            f"{float(beta['scaled_relative_l2_xyz_time_real']):.3e} | "
            f"{exact_wall:.4f} | "
            f"{brute / exact_wall:.2f}x | "
            f"{float(exact['scaled_relative_l2_xyz_time_real']):.3e} | "
            f"{jaxgb_wall:.4f} | "
            f"{brute / jaxgb_wall:.2f}x | "
            f"{float(jaxgb['scaled_relative_l2_xyz_time_real']):.3e} |"
        )
    return "\n".join(lines)


def latex_table(rows: list[dict[str, object]]) -> str:
    source_counts = sorted({int(row["sources"]) for row in rows})
    by_key = {(int(row["sources"]), str(row["model"])): row for row in rows}
    lines = [
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"$N$ & brute [s] & beta [s] & $\epsilon_{\rm beta}$ & exact [s] & $\epsilon_{\rm exact}$ & JAXGBacc [s] \\",
        r"\midrule",
    ]
    for count in source_counts:
        brute = float(by_key[(count, "brute_force_jax")]["wall_s"])
        beta = by_key[(count, "grouped_trilinear_beta")]
        exact = by_key[(count, "grouped_trilinear_exact_transfer")]
        jaxgb = by_key[(count, "jaxgbaccurate")]
        lines.append(
            f"{count} & "
            f"{brute:.4f} & "
            f"{float(beta['wall_s']):.4f} & "
            f"{float(beta['scaled_relative_l2_xyz_time_real']):.2e} & "
            f"{float(exact['wall_s']):.4f} & "
            f"{float(exact['scaled_relative_l2_xyz_time_real']):.2e} & "
            f"{float(jaxgb['wall_s']):.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def write_report(rows: list[dict[str, object]], report_dir: Path, csv_path: Path, png_path: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_csv = report_dir / csv_path.name
    report_png = report_dir / png_path.name
    shutil.copyfile(csv_path, report_csv)
    shutil.copyfile(png_path, report_png)

    beta_residuals = [float(row["scaled_relative_l2_xyz_time_real"]) for row in rows if row["model"] == "grouped_trilinear_beta"]
    exact_residuals = [float(row["scaled_relative_l2_xyz_time_real"]) for row in rows if row["model"] == "grouped_trilinear_exact_transfer"]
    jaxgb_residuals = [float(row["scaled_relative_l2_xyz_time_real"]) for row in rows if row["model"] == "jaxgbaccurate"]
    largest_count = max(int(row["sources"]) for row in rows)
    by_key = {(int(row["sources"]), str(row["model"])): row for row in rows}
    brute_large = float(by_key[(largest_count, "brute_force_jax")]["wall_s"])
    beta_large = float(by_key[(largest_count, "grouped_trilinear_beta")]["wall_s"])
    exact_large = float(by_key[(largest_count, "grouped_trilinear_exact_transfer")]["wall_s"])
    jaxgb_large = float(by_key[(largest_count, "jaxgbaccurate")]["wall_s"])
    text = "\n".join(
        [
            "# Circular Grouped Trilinear Study",
            "",
            "This study compares circular sources only (`e = 0`), using the JAX brute-force exact-link pyTDI response as the reference.",
            "The runs use the `esa-trailing` LISA orbit, one day of data, 2048 time samples, median wall times over three repeats, and source counts from 1 to 2048.",
            "",
            "The accuracy number is a scaled relative L2 residual over the real time-domain `X`, `Y`, and `Z` channels:",
            "",
            "```text",
            "epsilon_XYZ = sqrt(sum_c ||a_c Re[h_model,c] - Re[h_brute,c]||_2^2 / sum_c ||Re[h_brute,c]||_2^2)",
            "a_c = <Re[h_model,c], Re[h_brute,c]> / <Re[h_model,c], Re[h_model,c]>",
            "```",
            "",
            "This scale removes one real amplitude/sign factor per channel before measuring the remaining shape mismatch.",
            "",
            "JAXGBaccurate convention note: the eccentric model's circular limit uses an orbital phase, so the JAXGBaccurate gravitational-wave phase is set to `pi - 2 phi0`. The inclination is also mapped as `iota -> pi - iota` to match the eccentric real-polarization convention. The native sparse JAXGBaccurate Fourier bins are not compared directly to the ordinary observation FFT, because JAXGBaccurate uses an extended, windowed frequency grid.",
            "",
            markdown_table(rows),
            "",
            f"Median real time-domain scaled residuals: grouped beta {np.median(beta_residuals):.3e}, grouped exact-transfer {np.median(exact_residuals):.3e}, JAXGBaccurate {np.median(jaxgb_residuals):.3e}.",
            f"At {largest_count} circular sources, the grouped beta path is {brute_large / beta_large:.2f}x faster than brute force, the grouped exact-transfer path is {brute_large / exact_large:.2f}x faster, and JAXGBaccurate is {brute_large / jaxgb_large:.2f}x faster.",
            "",
            f"![Study plot]({report_png.name})",
            "",
        ]
    )
    (report_dir / "circular_grouped_trilinear_jaxgbaccurate_study.md").write_text(text, encoding="utf-8")

    tex = "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=0.8in]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{graphicx}",
            r"\usepackage{amsmath}",
            r"\usepackage{hyperref}",
            r"\title{Circular Trilinear, Brute Force, and JAXGBaccurate Study}",
            r"\author{}",
            r"\date{}",
            r"\begin{document}",
            r"\maketitle",
            r"\section*{Setup}",
            r"This study uses circular sources only, i.e. $e=0$. The reference is the JAX brute-force exact-link calculation followed by pyTDI. The runs use the ESA trailing LISA orbit, one day of data, 2048 time samples, and median wall times over three repeats.",
            r"",
            r"The compared methods are: the brute-force JAX exact-link response, the grouped trilinear beta response, the grouped trilinear exact-transfer response, and JAXGBaccurate with 128 sparse response bins.",
            r"\section*{Accuracy Metric}",
            r"For each model the residual is measured against the real time-domain brute-force JAX response. For $c\in\{X,Y,Z\}$,",
            r"\begin{equation}",
            r"\epsilon_{\rm XYZ} = \left( \frac{\sum_c \|a_c \Re h_{{\rm model},c}-\Re h_{{\rm brute},c}\|_2^2}{\sum_c \|\Re h_{{\rm brute},c}\|_2^2} \right)^{1/2},",
            r"\qquad",
            r"a_c = \frac{\langle \Re h_{{\rm model},c}, \Re h_{{\rm brute},c}\rangle}{\langle \Re h_{{\rm model},c}, \Re h_{{\rm model},c}\rangle}.",
            r"\end{equation}",
            r"The scale $a_c$ removes one real amplitude/sign factor per TDI channel before measuring the remaining shape mismatch.",
            r"\section*{Results}",
            r"\begin{center}",
            latex_table(rows),
            r"\end{center}",
            r"",
            f"Median real time-domain scaled residuals are {np.median(beta_residuals):.3e} for grouped beta, {np.median(exact_residuals):.3e} for grouped exact-transfer, and {np.median(jaxgb_residuals):.3e} for JAXGBaccurate.",
            r"",
            f"At {largest_count} circular sources, grouped beta is {brute_large / beta_large:.2f} times faster than brute force, grouped exact-transfer is {brute_large / exact_large:.2f} times faster, and JAXGBaccurate is {brute_large / jaxgb_large:.2f} times faster.",
            r"\begin{figure}[h]",
            r"\centering",
            rf"\includegraphics[width=\linewidth]{{{report_png.name}}}",
            r"\caption{Wall time and real time-domain scaled residual against the brute-force JAX reference.}",
            r"\end{figure}",
            r"\section*{JAXGBaccurate Convention Note}",
            r"The eccentric model's circular limit uses an orbital phase, while JAXGBaccurate expects the gravitational-wave phase and internally flips that phase. The comparison therefore uses $\phi_{0,\mathrm{JAXGB}}=\pi-2\phi_{0,\mathrm{ecc}}$ and $\iota_{\mathrm{JAXGB}}=\pi-\iota_{\mathrm{ecc}}$. The native sparse JAXGBaccurate Fourier bins are not compared directly to the ordinary observation FFT because JAXGBaccurate uses an extended, windowed frequency grid.",
            r"\end{document}",
            "",
        ]
    )
    (report_dir / "circular_grouped_trilinear_jaxgbaccurate_study.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(__file__).resolve().parent
    report_dir = root / "reports" / "circular_grouped_trilinear_jaxgbaccurate_study"

    orbit_kind = "esa-trailing"
    counts = [1, 8, 32, 128, 512, 2048]
    repeats = 3
    duration = 1.0 * 86400.0
    samples = 2048
    response_bins = 128
    harmonic_samples = 512
    max_batch_size = 128
    dt = duration / samples

    times = np.linspace(0.0, duration, samples, endpoint=False)
    orbits = default_lisaorbits(orbit_kind)
    state = state_from_lisaorbits(orbits, times)
    geometry = precompute_jax_link_geometry(state)
    jaxgb = JaxGBaccurate(orbits=orbits, t_obs=duration, t0=0.0, n=response_bins, extension=0.5)

    warm = make_sources(max_batch_size)
    eccentric_links_jax(warm, geometry, batch_size=max_batch_size)
    warm_links = eccentric_harmonic_links_grouped_jax(
        warm,
        geometry,
        eccentric_order=0,
        harmonic_samples=harmonic_samples,
        batch_size=max_batch_size,
        response_mode="fastgb_transfer",
    )
    xyz_from_links(state, warm_links)
    jaxgbaccurate_xyz_time_sum(jaxgb, as_jaxgb_params(warm), dt=dt, obs_samples=samples)

    rows: list[dict[str, object]] = []
    for count in counts:
        sources = make_sources(count)
        batch_size = min(max_batch_size, count)
        params = as_jaxgb_params(sources)

        eccentric_links_jax(sources, geometry, batch_size=batch_size)
        eccentric_harmonic_links_grouped_jax(
            sources,
            geometry,
            eccentric_order=0,
            harmonic_samples=harmonic_samples,
            batch_size=batch_size,
            response_mode="triliear_beta",
        )

        brute_links, brute_link_s = median_elapsed_value(
            lambda s=sources, b=batch_size: eccentric_links_jax(s, geometry, batch_size=b),
            repeats,
        )
        brute_xyz, brute_tdi_s = median_elapsed_value(lambda: xyz_from_links(state, brute_links), repeats)
        brute_total_s = brute_link_s + brute_tdi_s

        beta_links, beta_link_s = median_elapsed_value(
            lambda s=sources, b=batch_size: eccentric_harmonic_links_grouped_jax(
                s,
                geometry,
                eccentric_order=0,
                harmonic_samples=harmonic_samples,
                batch_size=b,
                response_mode="triliear_beta",
            ),
            repeats,
        )
        beta_xyz, beta_tdi_s = median_elapsed_value(lambda: xyz_from_links(state, beta_links), repeats)

        exact_links, exact_link_s = median_elapsed_value(
            lambda s=sources, b=batch_size: eccentric_harmonic_links_grouped_jax(
                s,
                geometry,
                eccentric_order=0,
                harmonic_samples=harmonic_samples,
                batch_size=b,
                response_mode="fastgb_transfer",
            ),
            repeats,
        )
        exact_xyz, exact_tdi_s = median_elapsed_value(lambda: xyz_from_links(state, exact_links), repeats)

        jaxgb_result, jaxgb_total_s = median_elapsed_value(
            lambda p=params: jaxgbaccurate_xyz_time_sum(jaxgb, p, dt=dt, obs_samples=samples),
            repeats,
        )
        jaxgb_xyz, jaxgb_fd, jaxgb_kmin, jaxgb_fd_s, jaxgb_recon_s = jaxgb_result
        jaxgb_bins = next(iter(jaxgb_fd.values())).size
        brute_fd = fft_common_support(brute_xyz, kmin=jaxgb_kmin, bins=jaxgb_bins, dt=dt)
        beta_fd = fft_common_support(beta_xyz, kmin=jaxgb_kmin, bins=jaxgb_bins, dt=dt)
        exact_fd = fft_common_support(exact_xyz, kmin=jaxgb_kmin, bins=jaxgb_bins, dt=dt)

        add_row(
            rows,
            source_count=count,
            model_name="brute_force_jax",
            total_s=brute_total_s,
            link_or_fd_s=brute_link_s,
            tdi_or_recon_s=brute_tdi_s,
            model_xyz=brute_xyz,
            reference_xyz=brute_xyz,
            model_fd=brute_fd,
            reference_fd=brute_fd,
            duration_days=duration / 86400.0,
            samples=samples,
            dt=dt,
            orbit_kind=orbit_kind,
            response_bins=response_bins,
        )
        add_row(
            rows,
            source_count=count,
            model_name="grouped_trilinear_beta",
            total_s=beta_link_s + beta_tdi_s,
            link_or_fd_s=beta_link_s,
            tdi_or_recon_s=beta_tdi_s,
            model_xyz=beta_xyz,
            reference_xyz=brute_xyz,
            model_fd=beta_fd,
            reference_fd=brute_fd,
            duration_days=duration / 86400.0,
            samples=samples,
            dt=dt,
            orbit_kind=orbit_kind,
            response_bins=response_bins,
        )
        add_row(
            rows,
            source_count=count,
            model_name="grouped_trilinear_exact_transfer",
            total_s=exact_link_s + exact_tdi_s,
            link_or_fd_s=exact_link_s,
            tdi_or_recon_s=exact_tdi_s,
            model_xyz=exact_xyz,
            reference_xyz=brute_xyz,
            model_fd=exact_fd,
            reference_fd=brute_fd,
            duration_days=duration / 86400.0,
            samples=samples,
            dt=dt,
            orbit_kind=orbit_kind,
            response_bins=response_bins,
        )
        add_row(
            rows,
            source_count=count,
            model_name="jaxgbaccurate",
            total_s=jaxgb_total_s,
            link_or_fd_s=jaxgb_fd_s,
            tdi_or_recon_s=jaxgb_recon_s,
            model_xyz=jaxgb_xyz,
            reference_xyz=brute_xyz,
            model_fd=jaxgb_fd,
            reference_fd=brute_fd,
            duration_days=duration / 86400.0,
            samples=samples,
            dt=dt,
            orbit_kind=orbit_kind,
            response_bins=response_bins,
            fd_comparable=False,
        )

        print(
            f"{count:5d} sources | brute {brute_total_s:.4f}s | "
            f"tri beta {beta_link_s + beta_tdi_s:.4f}s r_time={combined_xyz_residual(beta_xyz, brute_xyz, scaled=True, real_projection=True):.3e} | "
            f"tri exact {exact_link_s + exact_tdi_s:.4f}s r_time={combined_xyz_residual(exact_xyz, brute_xyz, scaled=True, real_projection=True):.3e} | "
            f"JAXGBaccurate {jaxgb_total_s:.4f}s r_time={combined_xyz_residual(jaxgb_xyz, brute_xyz, scaled=True, real_projection=True):.3e}",
            flush=True,
        )

    csv_path = out_dir / "circular_grouped_trilinear_jaxgbaccurate_study.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    png_path = out_dir / "circular_grouped_trilinear_jaxgbaccurate_study.png"
    plot(rows, png_path)
    write_report(rows, report_dir, csv_path, png_path)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"wrote {report_dir / 'circular_grouped_trilinear_jaxgbaccurate_study.md'}")


if __name__ == "__main__":
    main()
