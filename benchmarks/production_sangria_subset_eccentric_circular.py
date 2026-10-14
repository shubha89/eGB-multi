"""Sangria-subset production run for eccentric-vs-circular GB comparisons.

This run samples a reproducible subset from the Sangria training GB catalogs,
maps each catalog amplitude onto the local eccentric source by choosing the
source distance, and computes eccentric-vs-circular TDI-2 A/E metrics.

The dgb/igb Sangria catalogs provide the standard circular GB parameters but
not component masses. We therefore use fixed fiducial component masses for the
eccentric source dynamics and use the distance only as an amplitude calibration
so the circular-limit strain amplitude matches the Sangria catalog amplitude.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from production_eccentric_circular_fdot_tdi2_mismatch_grid import (
    chirp_mass_solar,
    fdot_candidates,
    response_aet,
)
from production_eccentric_circular_mismatch_grid import aet_inner, fd_aet, load_aet_psd

from egb_multi.constants import C_M_PER_S, G_M3_KG_S2, KILOPARSEC_M, SOLAR_MASS_KG
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits


SANGRIA_PATH = Path("/Users/shubhanshutiwari/Downloads/LDC2_sangria_training_v2.h5")
FREQUENCY_CENTERS = np.array([1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 3.0e-3])
ECCENTRICITIES = (1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 5.0e-2, 0.1, 0.2, 0.3)
CHUNK_SIZE = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("smoke", "production"), default="smoke")
    parser.add_argument("--sangria-path", type=Path, default=SANGRIA_PATH)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--sources-per-bin", type=int, default=None)
    parser.add_argument("--duration-days", type=float, default=None)
    parser.add_argument("--dt-s", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--measurement-order", type=int, default=3)
    parser.add_argument("--delay-order", type=int, default=3)
    parser.add_argument("--tdi-generation", type=int, default=2)
    parser.add_argument("--m1-solar", type=float, default=0.5)
    parser.add_argument("--m2-solar", type=float, default=0.5)
    parser.add_argument("--signal-fdot-mode", choices=("none", "catalog"), default="none")
    parser.add_argument("--circular-fdot-mode", choices=("none", "catalog", "grid", "gr"), default="catalog")
    parser.add_argument("--sample-only", action="store_true")
    return parser.parse_args()


def preset_defaults(preset: str) -> dict[str, float | int]:
    if preset == "smoke":
        return {"sources_per_bin": 1, "duration_days": 7.0, "dt_s": 100.0}
    return {"sources_per_bin": 8, "duration_days": 365.25, "dt_s": 50.0}


def frequency_edges(centers: np.ndarray) -> np.ndarray:
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    edges[0] = centers[0] ** 2 / edges[1]
    edges[-1] = centers[-1] ** 2 / edges[-2]
    return edges


def circular_amplitude_at_distance(f0_hz: float, m1_solar: float, m2_solar: float, distance_m: float) -> float:
    chirp_mass_kg = chirp_mass_solar(m1_solar, m2_solar) * SOLAR_MASS_KG
    return float(
        2.0
        * (G_M3_KG_S2 * chirp_mass_kg) ** (5.0 / 3.0)
        * (np.pi * f0_hz) ** (2.0 / 3.0)
        / (C_M_PER_S**4 * distance_m)
    )


def distance_for_catalog_amplitude(f0_hz: float, amplitude: float, m1_solar: float, m2_solar: float) -> float:
    amplitude_at_1kpc = circular_amplitude_at_distance(f0_hz, m1_solar, m2_solar, KILOPARSEC_M)
    return float(KILOPARSEC_M * amplitude_at_1kpc / amplitude)


def empty_reservoir() -> list[list[tuple[float, dict[str, object]]]]:
    return [[] for _ in FREQUENCY_CENTERS]


def catalog_item(
    row: np.void,
    *,
    catalog_name: str,
    catalog_index: int,
    bin_index: int,
    m1_solar: float,
    m2_solar: float,
) -> dict[str, object]:
    f0_hz = float(row["Frequency"])
    amplitude = float(row["Amplitude"])
    return {
        "catalog": catalog_name,
        "catalog_index": int(catalog_index),
        "frequency_bin_center_hz": float(FREQUENCY_CENTERS[int(bin_index)]),
        "f0_hz": f0_hz,
        "amplitude": amplitude,
        "fdot_hz_s": float(row["FrequencyDerivative"]),
        "beta": float(row["EclipticLatitude"]),
        "lambda": float(row["EclipticLongitude"]),
        "inclination": float(row["Inclination"]),
        "phi0": float(row["InitialPhase"]),
        "psi": float(row["Polarization"]),
        "m1_solar": float(m1_solar),
        "m2_solar": float(m2_solar),
        "distance_m": distance_for_catalog_amplitude(f0_hz, amplitude, m1_solar, m2_solar),
    }


def merge_reservoir(
    reservoir: list[tuple[float, dict[str, object]]],
    candidates: list[tuple[float, dict[str, object]]],
    *,
    capacity: int,
) -> list[tuple[float, dict[str, object]]]:
    merged = reservoir + candidates
    merged.sort(key=lambda item: item[0])
    return merged[:capacity]


def sample_sangria_subset(
    h5_path: Path,
    *,
    sources_per_bin: int,
    seed: int,
    m1_solar: float,
    m2_solar: float,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    edges = frequency_edges(FREQUENCY_CENTERS)
    reservoirs = empty_reservoir()

    with h5py.File(h5_path, "r") as h5:
        for catalog_name in ("sky/dgb/cat", "sky/igb/cat"):
            dset = h5[catalog_name]
            for start in range(0, dset.shape[0], CHUNK_SIZE):
                stop = min(start + CHUNK_SIZE, dset.shape[0])
                block = dset[start:stop, 0]
                freq = block["Frequency"]
                amp = block["Amplitude"]
                valid = np.isfinite(freq) & np.isfinite(amp) & (freq > 0.0) & (amp > 0.0)
                if not np.any(valid):
                    continue
                valid_indices = np.nonzero(valid)[0]
                which = np.searchsorted(edges, freq[valid], side="right") - 1
                in_range = (which >= 0) & (which < FREQUENCY_CENTERS.size)
                if not np.any(in_range):
                    continue
                valid_indices = valid_indices[in_range]
                which = which[in_range]
                for bin_index in range(FREQUENCY_CENTERS.size):
                    local_indices = valid_indices[which == bin_index]
                    if local_indices.size == 0:
                        continue
                    keys = rng.random(local_indices.size)
                    if local_indices.size > sources_per_bin:
                        keep = np.argpartition(keys, sources_per_bin - 1)[:sources_per_bin]
                    else:
                        keep = np.arange(local_indices.size)
                    candidates = [
                        (
                            float(keys[i]),
                            catalog_item(
                                block[int(local_indices[i])],
                                catalog_name=catalog_name,
                                catalog_index=start + int(local_indices[i]),
                                bin_index=bin_index,
                                m1_solar=m1_solar,
                                m2_solar=m2_solar,
                            ),
                        )
                        for i in keep
                    ]
                    reservoirs[bin_index] = merge_reservoir(
                        reservoirs[bin_index],
                        candidates,
                        capacity=sources_per_bin,
                    )

    samples: list[dict[str, object]] = []
    for bucket in reservoirs:
        samples.extend(item for _, item in bucket)
    samples.sort(key=lambda row: (float(row["frequency_bin_center_hz"]), str(row["catalog"]), int(row["catalog_index"])))
    return samples


def write_sample_catalog(samples: list[dict[str, object]], path: Path) -> None:
    if not samples:
        raise ValueError("no Sangria samples selected")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)


def make_source(sample: dict[str, object], *, eccentricity: float, fdot: float) -> EccentricBinaryParams:
    return EccentricBinaryParams(
        mean_motion=np.pi * float(sample["f0_hz"]),
        eccentricity=eccentricity,
        m1_solar=float(sample["m1_solar"]),
        m2_solar=float(sample["m2_solar"]),
        distance_m=float(sample["distance_m"]),
        beta=float(sample["beta"]),
        lambda_=float(sample["lambda"]),
        psi=float(sample["psi"]),
        inclination=float(sample["inclination"]),
        phi0=float(sample["phi0"]),
        fdot=fdot,
    )


def fdot_grid_from_catalog(mode: str, catalog_fdot: float) -> list[float]:
    if mode == "none":
        return [0.0]
    if mode in {"catalog", "gr"}:
        return [catalog_fdot]
    if catalog_fdot == 0.0 or not np.isfinite(catalog_fdot):
        return [0.0]
    candidates = [0.0, 0.25 * catalog_fdot, 0.5 * catalog_fdot, catalog_fdot, 2.0 * catalog_fdot, 4.0 * catalog_fdot]
    return sorted(set(float(x) for x in candidates))


def compare_aet_with_absolute_residual(
    eccentric_aet: dict[str, np.ndarray],
    circular_aet: dict[str, np.ndarray],
    psd: dict[str, np.ndarray],
    dt: float,
    frequency_mask: np.ndarray,
) -> dict[str, float]:
    df = 1.0 / (np.asarray(eccentric_aet["A"]).size * dt)
    ecc_fd = fd_aet(eccentric_aet, dt)
    circ_fd = fd_aet(circular_aet, dt)
    diff_fd = {channel: ecc_fd[channel] - circ_fd[channel] for channel in ("A", "E", "T")}
    ee = max(aet_inner(ecc_fd, ecc_fd, psd, df, frequency_mask).real, 0.0)
    cc = max(aet_inner(circ_fd, circ_fd, psd, df, frequency_mask).real, 0.0)
    dd = max(aet_inner(diff_fd, diff_fd, psd, df, frequency_mask).real, 0.0)
    ec = aet_inner(ecc_fd, circ_fd, psd, df, frequency_mask)
    fitting_factor = float(abs(ec) / np.sqrt(ee * cc)) if ee > 0.0 and cc > 0.0 else 0.0
    return {
        "signal_snr_AE": float(np.sqrt(ee)),
        "circular_snr_AE": float(np.sqrt(cc)),
        "residual_snr_AE": float(np.sqrt(dd)),
        "relative_noise_weighted_residual_AE": float(np.sqrt(dd / ee)) if ee > 0.0 else np.nan,
        "fitting_factor_AE": fitting_factor,
        "mismatch_AE": 1.0 - fitting_factor,
    }


def plot_summary(rows: list[dict[str, object]], path: Path) -> None:
    positive_rows = [row for row in rows if float(row["eccentricity"]) > 0.0]
    bins = sorted({float(row["frequency_bin_center_hz"]) for row in positive_rows})
    eccs = sorted({float(row["eccentricity"]) for row in positive_rows})
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    panels = [
        ("mismatch_AE", "A/E mismatch, $1-\\mathrm{FF}$"),
        ("residual_snr_AE", "residual SNR in A/E"),
        ("relative_noise_weighted_residual_AE", "relative noise-weighted residual"),
        ("signal_snr_AE", "signal SNR in A/E"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True, sharex=True)
    for ax, (column, ylabel) in zip(axes.flat, panels, strict=True):
        for color, bin_center in zip(colors, bins, strict=True):
            medians = []
            lows = []
            highs = []
            for ecc in eccs:
                values = np.array(
                    [
                        float(row[column])
                        for row in positive_rows
                        if float(row["frequency_bin_center_hz"]) == bin_center and float(row["eccentricity"]) == ecc
                    ],
                    dtype=np.float64,
                )
                values = values[np.isfinite(values) & (values > 0.0)]
                medians.append(np.median(values) if values.size else np.nan)
                lows.append(np.percentile(values, 16.0) if values.size else np.nan)
                highs.append(np.percentile(values, 84.0) if values.size else np.nan)
            x = np.array(eccs)
            y = np.array(medians)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=3.8,
                color=color,
                label=f"$f_0\\sim {bin_center:.0e}\\,$Hz",
            )
            ax.fill_between(x, lows, highs, color=color, alpha=0.10, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.28)
    for ax in axes[-1, :]:
        ax.set_xlabel("injected eccentricity")
    axes[0, 0].legend(fontsize=7.5, ncol=2, frameon=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    defaults = preset_defaults(args.preset)
    sources_per_bin = int(args.sources_per_bin or defaults["sources_per_bin"])
    duration_days = float(args.duration_days or defaults["duration_days"])
    dt = float(args.dt_s or defaults["dt_s"])
    out_dir = Path(__file__).resolve().parent
    out_prefix = out_dir / (
        f"production_sangria_subset_eccentric_circular_tdi{args.tdi_generation}_"
        f"signal_{args.signal_fdot_mode}_template_{args.circular_fdot_mode}_"
        f"{args.preset}_seed{args.seed}_n{sources_per_bin}perbin"
    )

    samples = sample_sangria_subset(
        args.sangria_path,
        sources_per_bin=sources_per_bin,
        seed=args.seed,
        m1_solar=args.m1_solar,
        m2_solar=args.m2_solar,
    )
    sample_catalog_path = out_prefix.with_name(out_prefix.name + "_sample_catalog.csv")
    write_sample_catalog(samples, sample_catalog_path)
    print(f"selected_sources={len(samples)}; wrote {sample_catalog_path}", flush=True)
    if args.sample_only:
        return

    psd_freq, psd_base = load_aet_psd(out_dir / "lisainstrument_welch_psd.csv")
    duration = duration_days * 86400.0
    times = np.arange(0.0, duration, dt, dtype=np.float64)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)
    freq = np.fft.rfftfreq(times.size, dt)
    psd = {channel: np.interp(freq, psd_freq, psd_base[channel], left=np.inf, right=np.inf) for channel in ("A", "E", "T")}

    compile_source = make_source(samples[0], eccentricity=0.0, fdot=float(samples[0]["fdot_hz_s"]))
    response_aet(
        compile_source,
        geometry,
        state,
        batch_size=args.batch_size,
        generation=args.tdi_generation,
        measurement_order=args.measurement_order,
        delay_order=args.delay_order,
    )

    total_cases = len(samples) * len(ECCENTRICITIES)
    print(
        f"preset={args.preset}, sources={len(samples)}, eccentricities={len(ECCENTRICITIES)}, "
        f"total_cases={total_cases}, duration_days={duration_days}, dt_s={dt}, samples={times.size}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    csv_path = out_prefix.with_suffix(".csv")
    csv_file = None
    writer = None
    start_all = time.perf_counter()
    case_index = 0
    try:
        for source_index, sample in enumerate(samples):
            catalog_fdot = float(sample["fdot_hz_s"])
            signal_fdot = catalog_fdot if args.signal_fdot_mode == "catalog" else 0.0
            circular_templates = []
            circular_start = time.perf_counter()
            for candidate_fdot in fdot_grid_from_catalog(args.circular_fdot_mode, catalog_fdot):
                circular_source = make_source(sample, eccentricity=0.0, fdot=float(candidate_fdot))
                circular_templates.append(
                    (
                        float(candidate_fdot),
                        response_aet(
                            circular_source,
                            geometry,
                            state,
                            batch_size=args.batch_size,
                            generation=args.tdi_generation,
                            measurement_order=args.measurement_order,
                            delay_order=args.delay_order,
                        ),
                    )
                )
            circular_wall = time.perf_counter() - circular_start
            f0_hz = float(sample["f0_hz"])
            band_mask = (freq >= 0.20 * f0_hz) & (freq <= min(freq[-1], 10.0 * f0_hz))

            for eccentricity in ECCENTRICITIES:
                case_index += 1
                eccentric_source = make_source(sample, eccentricity=float(eccentricity), fdot=signal_fdot)
                start = time.perf_counter()
                eccentric_aet = response_aet(
                    eccentric_source,
                    geometry,
                    state,
                    batch_size=args.batch_size,
                    generation=args.tdi_generation,
                    measurement_order=args.measurement_order,
                    delay_order=args.delay_order,
                )
                eccentric_wall = time.perf_counter() - start

                best_metrics = None
                best_fdot = None
                for candidate_fdot, circular_aet in circular_templates:
                    metrics = compare_aet_with_absolute_residual(eccentric_aet, circular_aet, psd, dt, band_mask.copy())
                    if best_metrics is None or metrics["fitting_factor_AE"] > best_metrics["fitting_factor_AE"]:
                        best_metrics = metrics
                        best_fdot = candidate_fdot
                assert best_metrics is not None
                assert best_fdot is not None

                row = {
                    "preset": args.preset,
                    "seed": args.seed,
                    "source_index": source_index,
                    "catalog": sample["catalog"],
                    "catalog_index": sample["catalog_index"],
                    "frequency_bin_center_hz": sample["frequency_bin_center_hz"],
                    "tdi_generation": args.tdi_generation,
                    "duration_days": duration_days,
                    "dt_s": dt,
                    "samples": times.size,
                    "f0_hz": sample["f0_hz"],
                    "catalog_amplitude": sample["amplitude"],
                    "catalog_fdot_hz_s": catalog_fdot,
                    "signal_fdot_mode": args.signal_fdot_mode,
                    "signal_fdot_hz_s": signal_fdot,
                    "circular_fdot_mode": args.circular_fdot_mode,
                    "best_circular_fdot_hz_s": best_fdot,
                    "circular_template_count": len(circular_templates),
                    "eccentricity": eccentricity,
                    "inclination": sample["inclination"],
                    "beta": sample["beta"],
                    "lambda": sample["lambda"],
                    "psi": sample["psi"],
                    "phi0": sample["phi0"],
                    "m1_solar": sample["m1_solar"],
                    "m2_solar": sample["m2_solar"],
                    "chirp_mass_solar": chirp_mass_solar(float(sample["m1_solar"]), float(sample["m2_solar"])),
                    "distance_m": sample["distance_m"],
                    "circular_wall_s": circular_wall,
                    "eccentric_wall_s": eccentric_wall,
                    **best_metrics,
                }
                if writer is None:
                    csv_file = csv_path.open("w", newline="")
                    writer = csv.DictWriter(csv_file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                csv_file.flush()
                rows.append(row)
                print(
                    f"[{case_index:04d}/{total_cases}] src={source_index:03d} "
                    f"f0={f0_hz:.3e} e={eccentricity:.3f} "
                    f"SNR={best_metrics['signal_snr_AE']:.3e} resSNR={best_metrics['residual_snr_AE']:.3e} "
                    f"FF={best_metrics['fitting_factor_AE']:.8f} mismatch={best_metrics['mismatch_AE']:.3e} "
                    f"wall={eccentric_wall:.2f}s",
                    flush=True,
                )
    finally:
        if csv_file is not None:
            csv_file.close()

    plot_path = out_prefix.with_suffix(".png")
    plot_summary(rows, plot_path)
    print(f"total_wall_s={time.perf_counter() - start_all:.3f}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
