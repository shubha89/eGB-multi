"""Production grid: eccentric fixed-n,e vs circular fixed-n full LISA response.

The grid compares exact-retarded time-domain eccentric waveforms against their
quasi-circular counterparts after pyTDI Michelson projection. The default
``pilot`` preset is intentionally moderate so it can be run interactively. The
``production`` preset expands the same axes for the paper-quality run.

Run examples:

    python benchmarks/production_eccentric_circular_mismatch_grid.py --preset pilot
    python benchmarks/production_eccentric_circular_mismatch_grid.py --preset production
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jax_bruteforce_links_walltime import xyz_from_links

from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.noise import aet_from_xyz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("pilot", "coarse", "production"), default="pilot")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--measurement-order", type=int, default=3)
    parser.add_argument("--delay-order", type=int, default=3)
    return parser.parse_args()


def grid_for_preset(preset: str) -> dict[str, object]:
    if preset == "pilot":
        return {
            "duration_days": 30.0,
            "dt_s": 100.0,
            "f0_hz": [2.0e-4, 1.0e-3],
            "eccentricity": [0.0, 0.01, 0.05, 0.1, 0.3, 0.5],
            "inclination": [0.4, 1.2],
            "sky": [(0.3, -0.7), (-0.5, 1.4)],
            "masses": [(0.5, 0.5), (0.8, 0.6)],
        }
    if preset == "coarse":
        return {
            "duration_days": 90.0,
            "dt_s": 50.0,
            "f0_hz": [1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3],
            "eccentricity": [0.0, 1.0e-3, 1.0e-2, 3.0e-2, 5.0e-2, 0.1, 0.2, 0.3, 0.5],
            "inclination": [0.4, 1.2],
            "sky": [(0.3, -0.7), (-0.5, 1.4), (0.0, 2.8)],
            "masses": [(0.5, 0.5), (0.8, 0.6)],
        }
    return {
        "duration_days": 365.25,
        "dt_s": 50.0,
        "f0_hz": [1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 3.0e-3],
        "eccentricity": [0.0, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 5.0e-2, 0.1, 0.2, 0.3, 0.5],
        "inclination": [0.2, 0.8, 1.2],
        "sky": [(0.2, -1.0), (0.6, 0.5), (-0.5, 1.8), (0.0, 2.8)],
        "masses": [(0.5, 0.5), (0.8, 0.6), (1.0, 0.3)],
    }


def load_aet_psd(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = list(csv.DictReader(path.open()))
    freq = np.array([float(row["frequency_hz"]) for row in rows], dtype=np.float64)
    return freq, {
        "A": np.array([float(row["PSD_A"]) for row in rows], dtype=np.float64),
        "E": np.array([float(row["PSD_E"]) for row in rows], dtype=np.float64),
        "T": np.array([float(row["PSD_T"]) for row in rows], dtype=np.float64),
    }


def real_rfft(series: np.ndarray, dt: float) -> np.ndarray:
    return np.fft.rfft(np.real(np.asarray(series))) * dt


def channel_inner(
    left_fd: np.ndarray,
    right_fd: np.ndarray,
    psd: np.ndarray,
    df: float,
    mask: np.ndarray,
) -> complex:
    valid = mask & np.isfinite(psd) & (psd > 0.0)
    if valid.size:
        valid[0] = False
    return complex(4.0 * np.sum(left_fd[valid] * np.conjugate(right_fd[valid]) / psd[valid]) * df)


def aet_inner(
    left_fd: dict[str, np.ndarray],
    right_fd: dict[str, np.ndarray],
    psd: dict[str, np.ndarray],
    df: float,
    mask: np.ndarray,
    channels: tuple[str, ...] = ("A", "E"),
) -> complex:
    total = 0.0j
    for channel in channels:
        total += channel_inner(left_fd[channel], right_fd[channel], psd[channel], df, mask.copy())
    return total


def fd_aet(aet: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    return {channel: real_rfft(aet[channel], dt) for channel in ("A", "E", "T")}


def compare_aet(
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
    ee = aet_inner(ecc_fd, ecc_fd, psd, df, frequency_mask).real
    cc = aet_inner(circ_fd, circ_fd, psd, df, frequency_mask).real
    dd = aet_inner(diff_fd, diff_fd, psd, df, frequency_mask).real
    ec = aet_inner(ecc_fd, circ_fd, psd, df, frequency_mask)
    ff = float(abs(ec) / np.sqrt(ee * cc)) if ee > 0.0 and cc > 0.0 else 0.0

    numerator = sum(np.linalg.norm(np.real(eccentric_aet[ch] - circular_aet[ch])) ** 2 for ch in ("A", "E"))
    denominator = sum(np.linalg.norm(np.real(eccentric_aet[ch])) ** 2 for ch in ("A", "E"))
    return {
        "relative_l2_residual_AE": float(np.sqrt(numerator / denominator)) if denominator > 0.0 else np.nan,
        "noise_weighted_residual_AE": float(np.sqrt(max(dd, 0.0) / ee)) if ee > 0.0 else np.nan,
        "fitting_factor_AE": ff,
        "mismatch_AE": 1.0 - ff,
    }


def make_source(
    *,
    f0_hz: float,
    eccentricity: float,
    inclination: float,
    beta: float,
    lambda_: float,
    m1_solar: float,
    m2_solar: float,
) -> EccentricBinaryParams:
    return EccentricBinaryParams(
        mean_motion=np.pi * f0_hz,
        eccentricity=eccentricity,
        m1_solar=m1_solar,
        m2_solar=m2_solar,
        beta=beta,
        lambda_=lambda_,
        psi=0.4,
        inclination=inclination,
        phi0=0.2,
    )


def response_aet(source: EccentricBinaryParams, geometry, state, *, batch_size: int) -> dict[str, np.ndarray]:
    links = eccentric_links_jax(source, geometry, batch_size=batch_size, physics_mode="1pn")
    xyz = xyz_from_links(state, links)
    return aet_from_xyz(xyz)


def main() -> None:
    args = parse_args()
    config = grid_for_preset(args.preset)
    out_dir = Path(__file__).resolve().parent
    out_prefix = out_dir / f"production_eccentric_circular_mismatch_grid_{args.preset}"
    psd_freq, psd_base = load_aet_psd(out_dir / "lisainstrument_welch_psd.csv")

    duration = float(config["duration_days"]) * 86400.0
    dt = float(config["dt_s"])
    times = np.arange(0.0, duration, dt, dtype=np.float64)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)
    freq = np.fft.rfftfreq(times.size, dt)
    psd = {channel: np.interp(freq, psd_freq, psd_base[channel], left=np.inf, right=np.inf) for channel in ("A", "E", "T")}

    axes = list(
        itertools.product(
            config["f0_hz"],
            config["inclination"],
            config["sky"],
            config["masses"],
        )
    )
    total_cases = len(axes) * len(config["eccentricity"])
    print(
        f"preset={args.preset}, samples={times.size}, duration_days={config['duration_days']}, "
        f"dt={dt}, base_points={len(axes)}, eccentricities={len(config['eccentricity'])}, total_cases={total_cases}"
    )

    # Compile once at fixed shape.
    compile_source = make_source(
        f0_hz=float(config["f0_hz"][0]),
        eccentricity=0.0,
        inclination=float(config["inclination"][0]),
        beta=float(config["sky"][0][0]),
        lambda_=float(config["sky"][0][1]),
        m1_solar=float(config["masses"][0][0]),
        m2_solar=float(config["masses"][0][1]),
    )
    response_aet(compile_source, geometry, state, batch_size=args.batch_size)

    csv_path = out_prefix.with_suffix(".csv")
    rows = []
    csv_file = None
    writer = None
    start_all = time.perf_counter()
    case_index = 0
    try:
        for f0_hz, inclination, sky, masses in axes:
            beta, lambda_ = sky
            m1_solar, m2_solar = masses
            circular_source = make_source(
                f0_hz=float(f0_hz),
                eccentricity=0.0,
                inclination=float(inclination),
                beta=float(beta),
                lambda_=float(lambda_),
                m1_solar=float(m1_solar),
                m2_solar=float(m2_solar),
            )
            circular_start = time.perf_counter()
            circular_aet = response_aet(circular_source, geometry, state, batch_size=args.batch_size)
            circular_wall = time.perf_counter() - circular_start

            band_mask = (freq >= 0.20 * float(f0_hz)) & (freq <= min(freq[-1], 10.0 * float(f0_hz)))
            for eccentricity in config["eccentricity"]:
                case_index += 1
                eccentric_source = make_source(
                    f0_hz=float(f0_hz),
                    eccentricity=float(eccentricity),
                    inclination=float(inclination),
                    beta=float(beta),
                    lambda_=float(lambda_),
                    m1_solar=float(m1_solar),
                    m2_solar=float(m2_solar),
                )
                start = time.perf_counter()
                eccentric_aet = response_aet(eccentric_source, geometry, state, batch_size=args.batch_size)
                eccentric_wall = time.perf_counter() - start
                metrics = compare_aet(eccentric_aet, circular_aet, psd, dt, band_mask.copy())
                row = {
                    "preset": args.preset,
                    "duration_days": config["duration_days"],
                    "dt_s": dt,
                    "samples": times.size,
                    "f0_hz": f0_hz,
                    "eccentricity": eccentricity,
                    "inclination": inclination,
                    "beta": beta,
                    "lambda": lambda_,
                    "m1_solar": m1_solar,
                    "m2_solar": m2_solar,
                    "circular_wall_s": circular_wall,
                    "eccentric_wall_s": eccentric_wall,
                    **metrics,
                }
                if writer is None:
                    csv_file = csv_path.open("w", newline="")
                    writer = csv.DictWriter(csv_file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                csv_file.flush()
                rows.append(row)
                print(
                    f"[{case_index:04d}/{total_cases}] f0={float(f0_hz):.1e} e={float(eccentricity):.3f} "
                    f"i={float(inclination):.2f} beta={float(beta):.2f} lam={float(lambda_):.2f} "
                    f"m={float(m1_solar):.1f}+{float(m2_solar):.1f} "
                    f"FF={metrics['fitting_factor_AE']:.8f} mismatch={metrics['mismatch_AE']:.3e} "
                    f"wall={eccentric_wall:.2f}s",
                    flush=True,
                )
    finally:
        if csv_file is not None:
            csv_file.close()

    plot_summary(rows, out_prefix.with_suffix(".png"))
    print(f"total_wall_s={time.perf_counter() - start_all:.3f}")
    print(f"wrote {csv_path}")
    print(f"wrote {out_prefix.with_suffix('.png')}")


def plot_summary(rows: list[dict[str, object]], path: Path) -> None:
    frequencies = sorted({float(row["f0_hz"]) for row in rows})
    fig, axes = plt.subplots(1, len(frequencies), figsize=(5.4 * len(frequencies), 4.2), constrained_layout=True, squeeze=False)
    for ax, f0_hz in zip(axes[0], frequencies, strict=True):
        subset_f = [row for row in rows if float(row["f0_hz"]) == f0_hz]
        eccentricities = sorted({float(row["eccentricity"]) for row in subset_f})
        med = []
        lo = []
        hi = []
        for eccentricity in eccentricities:
            values = np.array(
                [max(float(row["mismatch_AE"]), 1.0e-16) for row in subset_f if float(row["eccentricity"]) == eccentricity],
                dtype=np.float64,
            )
            med.append(np.median(values))
            lo.append(np.percentile(values, 16.0))
            hi.append(np.percentile(values, 84.0))
        x = np.array(eccentricities)
        med_arr = np.array(med)
        ax.plot(x, med_arr, marker="o", label="median over sky/inclination/mass")
        ax.fill_between(x, lo, hi, alpha=0.25, label="16-84 percentile")
        ax.set_xscale("symlog", linthresh=1.0e-3)
        ax.set_yscale("log")
        ax.set_xlabel("eccentricity")
        ax.set_title(f"f0={f0_hz:.1e} Hz")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("A/E noise-weighted mismatch, 1 - FF")
    fig.suptitle("Eccentric fixed-n,e vs circular fixed-n, full time-domain LISA response")
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
