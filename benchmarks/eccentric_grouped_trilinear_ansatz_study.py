"""Eccentric grouped-trilinear ansatz wall-time and accuracy study.

The study compares fixed-eccentric sources using:

1. JAX brute-force eccentric links + pyTDI, used as the reference.
2. Grouped JAX harmonic trilinear beta links + pyTDI.
3. Grouped JAX harmonic exact-transfer links + pyTDI.

The harmonic truncation is selected from an eccentricity ansatz:

    n_max(e) = ceil(2.5 + 25 e + 20 e^2),  e > 0,

with only harmonic 2 kept at e = 0.  Since the package's historical
`eccentric_order` convention has order p -> harmonics 1 ... p + 2, this means
`eccentric_order = n_max - 2` for eccentric cases.

Run with:

    python benchmarks/eccentric_grouped_trilinear_ansatz_study.py

Outputs:

    benchmarks/eccentric_grouped_trilinear_ansatz_study.csv
    benchmarks/eccentric_grouped_trilinear_ansatz_study.png
    reports/eccentric_grouped_trilinear_ansatz_study/
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.constants import KILOPARSEC_M
from egb_multi.eccentric import (
    EccentricBinaryParams,
    eccentric_harmonic_angular_frequency,
    eccentric_harmonic_indices,
    eccentric_harmonic_projection_angular_frequency,
    eccentric_polarizations,
)
from egb_multi.eccentric_jax import eccentric_harmonic_links_grouped_jax, eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.tdi import pytdi_data_from_links


CHANNELS = ("X", "Y", "Z")


def harmonic_nmax_ansatz(eccentricity: float) -> int:
    """Return the largest positive harmonic used by this study."""

    if eccentricity == 0.0:
        return 2
    return max(3, int(np.ceil(2.5 + 25.0 * eccentricity + 20.0 * eccentricity * eccentricity)))


def harmonic_order_from_ansatz(eccentricity: float) -> int:
    """Return package `eccentric_order` corresponding to `harmonic_nmax_ansatz`."""

    return 0 if eccentricity == 0.0 else harmonic_nmax_ansatz(eccentricity) - 2


def make_sources(n_sources: int, eccentricity: float, *, seed: int = 1234) -> list[EccentricBinaryParams]:
    rng = np.random.default_rng(seed)
    return [
        EccentricBinaryParams(
            mean_motion=float(np.pi * rng.uniform(1.0e-4, 3.0e-3)),
            eccentricity=eccentricity,
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


def best_fit_scale(model: np.ndarray, reference: np.ndarray) -> complex:
    denom = np.vdot(model, model)
    return 0.0j if denom == 0.0 else complex(np.vdot(model, reference) / denom)


def combined_xyz_residual(
    model: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    scaled: bool,
    real_projection: bool = True,
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


def analytic_reconstruction(signal: np.ndarray, harmonics: tuple[int, ...]) -> np.ndarray:
    coeffs = np.fft.fft(signal) / signal.size
    j = np.arange(signal.size)
    out = np.zeros(signal.size, dtype=np.complex128)
    for harmonic in harmonics:
        out += 2.0 * coeffs[harmonic] * np.exp(2j * np.pi * harmonic * j / signal.size)
    return np.real(out)


def sideband_reconstruction(
    source: EccentricBinaryParams,
    signal: np.ndarray,
    times: np.ndarray,
    harmonics: tuple[int, ...],
    *,
    physics_mode: str = "1pn",
) -> np.ndarray:
    out = np.zeros(signal.size, dtype=np.complex128)
    for harmonic in harmonics:
        angular_frequency = eccentric_harmonic_angular_frequency(source, harmonic, physics_mode=physics_mode)
        coeff = np.mean(signal * np.exp(-1.0j * angular_frequency * times))
        out += coeff * np.exp(1.0j * angular_frequency * times)
    return 2.0 * np.real(out)


def source_harmonic_residual(
    source: EccentricBinaryParams,
    harmonics: tuple[int, ...],
    *,
    samples: int,
    physics_mode: str = "1pn",
) -> float:
    projection_frequency = eccentric_harmonic_projection_angular_frequency(source, physics_mode=physics_mode)
    period = 2.0 * np.pi / projection_frequency
    t = source.t0 + np.linspace(0.0, period, samples, endpoint=False)
    hplus, hcross = eccentric_polarizations(source, t, physics_mode=physics_mode)
    hplus_approx = sideband_reconstruction(source, hplus, t, harmonics, physics_mode=physics_mode)
    hcross_approx = sideband_reconstruction(source, hcross, t, harmonics, physics_mode=physics_mode)
    numerator = np.linalg.norm(hplus - hplus_approx) ** 2 + np.linalg.norm(hcross - hcross_approx) ** 2
    denominator = np.linalg.norm(hplus) ** 2 + np.linalg.norm(hcross) ** 2
    return float(np.sqrt(numerator / denominator))


def plot(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})
    counts = sorted({int(row["sources"]) for row in rows})
    colors = plt.cm.viridis(np.linspace(0.0, 0.88, len(eccentricities)))

    for color, eccentricity in zip(colors, eccentricities, strict=True):
        subset = sorted([row for row in rows if float(row["eccentricity"]) == eccentricity], key=lambda row: int(row["sources"]))
        x = np.array([int(row["sources"]) for row in subset])
        axes[0, 0].plot(
            x,
            [float(row["exact_transfer_speedup_vs_brute"]) for row in subset],
            marker="o",
            color=color,
            label=f"e={eccentricity:g}",
        )

    for count in counts:
        subset = sorted([row for row in rows if int(row["sources"]) == count], key=lambda row: float(row["eccentricity"]))
        x = np.array([float(row["eccentricity"]) for row in subset])
        axes[0, 1].plot(x, [float(row["exact_transfer_scaled_real_residual"]) for row in subset], marker="s", label=f"N={count}")

    largest_count = max(counts)
    largest_subset = sorted([row for row in rows if int(row["sources"]) == largest_count], key=lambda row: float(row["eccentricity"]))
    x = np.array([float(row["eccentricity"]) for row in largest_subset])
    axes[1, 0].plot(x, [float(row["brute_force_wall_s"]) for row in largest_subset], marker="o", label="brute force")
    axes[1, 0].plot(x, [float(row["trilinear_exact_transfer_wall_s"]) for row in largest_subset], marker="^", label="grouped exact-transfer")

    ax_harm = axes[1, 1]
    ax_source = ax_harm.twinx()
    ax_harm.plot(x, [int(row["n_max_harmonic"]) for row in largest_subset], marker="o", color="tab:blue", label="n_max ansatz")
    ax_source.plot(
        x,
        [float(row["source_harmonic_scaled_residual"]) for row in largest_subset],
        marker="s",
        color="tab:orange",
        label="source residual",
    )

    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xlabel("number of sources")
    axes[0, 0].set_ylabel("brute / exact-transfer wall time")
    axes[0, 0].set_title("exact-transfer speedup")

    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("eccentricity")
    axes[0, 1].set_ylabel("scaled Re(XYZ) residual")
    axes[0, 1].set_title("exact-transfer accuracy")

    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("eccentricity")
    axes[1, 0].set_ylabel("wall time [s]")
    axes[1, 0].set_title(f"wall time at N={largest_count}")

    ax_harm.set_xlabel("eccentricity")
    ax_harm.set_title("harmonic ansatz")
    ax_harm.set_ylabel("largest positive harmonic")
    ax_harm.set_yscale("log")
    ax_source.set_ylabel("source harmonic residual")
    ax_source.set_yscale("log")

    for ax in axes.flat:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    ax_source.legend(fontsize=8, loc="lower right")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict[str, object]]) -> str:
    largest_count = max(int(row["sources"]) for row in rows)
    subset = sorted([row for row in rows if int(row["sources"]) == largest_count], key=lambda row: float(row["eccentricity"]))
    lines = [
        f"| e | n_max | order | brute [s] | exact [s] | exact speedup | exact residual |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in subset:
        lines.append(
            "| "
            f"{float(row['eccentricity']):.2f} | "
            f"{int(row['n_max_harmonic'])} | "
            f"{int(row['eccentric_order'])} | "
            f"{float(row['brute_force_wall_s']):.4f} | "
            f"{float(row['trilinear_exact_transfer_wall_s']):.4f} | "
            f"{float(row['exact_transfer_speedup_vs_brute']):.2f}x | "
            f"{float(row['exact_transfer_scaled_real_residual']):.3e} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, object]], report_dir: Path, csv_path: Path, png_path: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_csv = report_dir / csv_path.name
    report_png = report_dir / png_path.name
    shutil.copyfile(csv_path, report_csv)
    shutil.copyfile(png_path, report_png)

    exact_residuals = [float(row["exact_transfer_scaled_real_residual"]) for row in rows]
    exact_speedups = [float(row["exact_transfer_speedup_vs_brute"]) for row in rows]
    text = "\n".join(
        [
            "# Eccentric Grouped Trilinear Ansatz Study",
            "",
            "This study compares fixed-eccentric grouped trilinear responses against the JAX brute-force eccentric link response followed by pyTDI.",
            "",
            "The harmonic cutoff is selected from the ansatz:",
            "",
            "```text",
            "n_max(e) = 2                         for e = 0",
            "n_max(e) = ceil(2.5 + 25 e + 20 e^2) for e > 0",
            "eccentric_order = n_max - 2",
            "```",
            "",
            "Accuracy is the scaled relative L2 residual in the real time-domain `X`, `Y`, and `Z` channels after one real scale/sign fit per channel.",
            "",
            markdown_table(rows),
            "",
            f"Across the full grid, median grouped exact-transfer residual is {np.median(exact_residuals):.3e}, and median exact-transfer speedup over brute force is {np.median(exact_speedups):.2f}x.",
            "",
            f"![Study plot]({report_png.name})",
            "",
        ]
    )
    (report_dir / "eccentric_grouped_trilinear_ansatz_study.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(__file__).resolve().parent
    report_dir = root / "reports" / "eccentric_grouped_trilinear_ansatz_study"

    orbit_kind = "esa-trailing"
    counts = [32, 128, 512, 2048]
    eccentricities = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
    repeats = 3
    duration = 1.0 * 86400.0
    samples = 2048
    harmonic_samples = 1024
    batch_size = 128
    physics_mode = "1pn"

    times = np.linspace(0.0, duration, samples, endpoint=False)
    orbits = default_lisaorbits(orbit_kind)
    state = state_from_lisaorbits(orbits, times)
    geometry = precompute_jax_link_geometry(state)

    warm_sources = make_sources(batch_size, max(eccentricities))
    eccentric_links_jax(warm_sources, geometry, batch_size=batch_size, physics_mode=physics_mode)
    xyz_from_links(state, eccentric_links_jax(warm_sources[:1], geometry, batch_size=batch_size, physics_mode=physics_mode))
    for eccentricity in eccentricities:
        order = harmonic_order_from_ansatz(eccentricity)
        warm = [EccentricBinaryParams(**{**source.__dict__, "eccentricity": eccentricity}) for source in warm_sources]
        for mode in ("triliear_beta", "fastgb_transfer"):
            eccentric_harmonic_links_grouped_jax(
                warm,
                geometry,
                eccentric_order=order,
                harmonic_samples=harmonic_samples,
                batch_size=batch_size,
                response_mode=mode,
                physics_mode=physics_mode,
            )

    rows: list[dict[str, object]] = []
    for eccentricity in eccentricities:
        order = harmonic_order_from_ansatz(eccentricity)
        harmonics = eccentric_harmonic_indices(order)
        n_max = max(harmonics)
        for count in counts:
            sources = make_sources(count, eccentricity)
            source_residual = source_harmonic_residual(
                sources[0],
                harmonics,
                samples=harmonic_samples,
                physics_mode=physics_mode,
            )

            brute_links, brute_links_s = median_elapsed_value(
                lambda s=sources: eccentric_links_jax(s, geometry, batch_size=batch_size, physics_mode=physics_mode),
                repeats,
            )
            brute_xyz, brute_tdi_s = median_elapsed_value(lambda: xyz_from_links(state, brute_links), repeats)
            brute_wall_s = brute_links_s + brute_tdi_s

            beta_links, beta_links_s = median_elapsed_value(
                lambda s=sources, o=order: eccentric_harmonic_links_grouped_jax(
                    s,
                    geometry,
                    eccentric_order=o,
                    harmonic_samples=harmonic_samples,
                    batch_size=batch_size,
                    response_mode="triliear_beta",
                    physics_mode=physics_mode,
                ),
                repeats,
            )
            beta_xyz, beta_tdi_s = median_elapsed_value(lambda: xyz_from_links(state, beta_links), repeats)
            beta_wall_s = beta_links_s + beta_tdi_s

            exact_links, exact_links_s = median_elapsed_value(
                lambda s=sources, o=order: eccentric_harmonic_links_grouped_jax(
                    s,
                    geometry,
                    eccentric_order=o,
                    harmonic_samples=harmonic_samples,
                    batch_size=batch_size,
                    response_mode="fastgb_transfer",
                    physics_mode=physics_mode,
                ),
                repeats,
            )
            exact_xyz, exact_tdi_s = median_elapsed_value(lambda: xyz_from_links(state, exact_links), repeats)
            exact_wall_s = exact_links_s + exact_tdi_s

            row = {
                "sources": count,
                "eccentricity": eccentricity,
                "orbit_kind": orbit_kind,
                "duration_days": duration / 86400.0,
                "samples": samples,
                "dt_s": times[1] - times[0],
                "physics_mode": physics_mode,
                "harmonic_samples": harmonic_samples,
                "n_max_harmonic": n_max,
                "positive_harmonic_count": len(harmonics),
                "positive_harmonics": " ".join(str(harmonic) for harmonic in harmonics),
                "eccentric_order": order,
                "source_harmonic_scaled_residual": source_residual,
                "brute_force_links_s": brute_links_s,
                "brute_force_tdi_s": brute_tdi_s,
                "brute_force_wall_s": brute_wall_s,
                "trilinear_beta_links_s": beta_links_s,
                "trilinear_beta_tdi_s": beta_tdi_s,
                "trilinear_beta_wall_s": beta_wall_s,
                "trilinear_exact_transfer_links_s": exact_links_s,
                "trilinear_exact_transfer_tdi_s": exact_tdi_s,
                "trilinear_exact_transfer_wall_s": exact_wall_s,
                "beta_raw_real_residual": combined_xyz_residual(beta_xyz, brute_xyz, scaled=False),
                "beta_scaled_real_residual": combined_xyz_residual(beta_xyz, brute_xyz, scaled=True),
                "exact_transfer_raw_real_residual": combined_xyz_residual(exact_xyz, brute_xyz, scaled=False),
                "exact_transfer_scaled_real_residual": combined_xyz_residual(exact_xyz, brute_xyz, scaled=True),
                "beta_speedup_vs_brute": brute_wall_s / beta_wall_s,
                "exact_transfer_speedup_vs_brute": brute_wall_s / exact_wall_s,
            }
            rows.append(row)
            print(
                f"e={eccentricity:.2f} N={count:4d} harmonics=1..{n_max} order={order:2d} | "
                f"brute {brute_wall_s:.4f}s | "
                f"beta {beta_wall_s:.4f}s r={row['beta_scaled_real_residual']:.3e} | "
                f"exact {exact_wall_s:.4f}s r={row['exact_transfer_scaled_real_residual']:.3e} "
                f"speedup={row['exact_transfer_speedup_vs_brute']:.2f}x",
                flush=True,
            )

    csv_path = out_dir / "eccentric_grouped_trilinear_ansatz_study.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    png_path = out_dir / "eccentric_grouped_trilinear_ansatz_study.png"
    plot(rows, png_path)
    write_report(rows, report_dir, csv_path, png_path)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"wrote {report_dir / 'eccentric_grouped_trilinear_ansatz_study.md'}")


if __name__ == "__main__":
    main()
