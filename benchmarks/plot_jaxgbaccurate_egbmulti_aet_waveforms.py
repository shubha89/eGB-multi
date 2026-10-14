"""Plot JAXGBaccurate and eGB-multi A/E/T response time series."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
FIGURES_FINAL = ROOT / "reports" / "Effect_of_eccentricity_for_GB_in_LISA-final" / "figures"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BENCHMARKS))

from jaxgb.jaxgb import JaxGBaccurate

from egb_multi.eccentric import peters_mathews_derivatives
from egb_multi.eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.noise import aet_from_xyz

from jaxgbaccurate_calibrated_eccentric_study import as_jaxgb_params, jaxgbaccurate_xyz_time, make_source, xyz_from_links


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0", type=float, default=1.0e-3, help="Circular GW frequency in Hz.")
    parser.add_argument("--duration-days", type=float, default=7.0, help="Observation duration in days.")
    parser.add_argument("--samples", type=int, default=4096, help="Number of time samples.")
    parser.add_argument("--response-bins", type=int, default=128, help="JAXGBaccurate sparse response bins.")
    parser.add_argument("--window-days", type=float, default=0.08, help="Time window shown in the plot.")
    parser.add_argument(
        "--plot-mode",
        choices=("zoom", "full-band"),
        default="zoom",
        help="'zoom' plots the first window-days; 'full-band' plots binned min/max bands over the full duration.",
    )
    parser.add_argument("--band-bins", type=int, default=1200, help="Number of time bins for full-band plotting.")
    parser.add_argument("--physics-mode", choices=("newtonian", "1pn"), default="1pn")
    parser.add_argument(
        "--evolution-mode",
        choices=("fixed", "peters_mathews", "peters_mathews_orbital_only", "peters_mathews_eccentricity_only", "auto"),
        default="fixed",
        help="eGB-multi source evolution mode.",
    )
    parser.add_argument("--pm-mismatch-tolerance", type=float, default=1.0e-2)
    parser.add_argument(
        "--jaxgb-fdot-mode",
        choices=("none", "pm-initial"),
        default="none",
        help="Frequency derivative assigned to the circular JAXGBaccurate model.",
    )
    parser.add_argument(
        "--jaxgb-real-factor",
        type=float,
        default=2.0,
        help="Factor applied to Re[JAXGBaccurate ifft] to restore the physical real waveform from one-sided bins.",
    )
    parser.add_argument("--output-prefix", default="jaxgbaccurate_egbmulti_aet_waveforms")
    return parser.parse_args()


def elapsed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def source_with_pm_initial_fdot(source):
    ndot, _ = peters_mathews_derivatives(source.mean_motion, source.eccentricity, source.total_mass_kg, source.symmetric_mass_ratio)
    return replace(source, fdot=float(ndot / np.pi))


def write_csv(
    path: Path,
    days: np.ndarray,
    jaxgb_aet: dict[str, np.ndarray],
    egb_aet: dict[str, np.ndarray],
    *,
    jaxgb_real_factor: float,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_days",
                "jaxgbaccurate_A",
                "jaxgbaccurate_E",
                "jaxgbaccurate_T",
                "egbmulti_A",
                "egbmulti_E",
                "egbmulti_T",
            ]
        )
        for i, t in enumerate(days):
            writer.writerow(
                [
                    t,
                    jaxgb_real_factor * np.real(jaxgb_aet["A"][i]),
                    jaxgb_real_factor * np.real(jaxgb_aet["E"][i]),
                    jaxgb_real_factor * np.real(jaxgb_aet["T"][i]),
                    np.real(egb_aet["A"][i]),
                    np.real(egb_aet["E"][i]),
                    np.real(egb_aet["T"][i]),
                ]
            )


def binned_minmax(x: np.ndarray, y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    count = min(bins, x.size)
    edges = np.linspace(0, x.size, count + 1, dtype=np.int64)
    centers = np.empty(count, dtype=np.float64)
    lows = np.empty(count, dtype=np.float64)
    highs = np.empty(count, dtype=np.float64)
    for i in range(count):
        sl = slice(edges[i], edges[i + 1])
        centers[i] = 0.5 * (x[edges[i]] + x[edges[i + 1] - 1])
        ys = y[sl]
        lows[i] = np.min(ys)
        highs[i] = np.max(ys)
    return centers, lows, highs


def main() -> None:
    args = parse_args()
    duration = args.duration_days * 86400.0
    dt = duration / args.samples
    times = np.linspace(0.0, duration, args.samples, endpoint=False)
    days = times / 86400.0

    source = make_source(args.f0, 0.0)
    jaxgb_source = source_with_pm_initial_fdot(source) if args.jaxgb_fdot_mode == "pm-initial" else source
    orbits = default_lisaorbits("equal")
    state = state_from_lisaorbits(orbits, times)
    geometry = precompute_jax_link_geometry(state)
    jaxgb = JaxGBaccurate(orbits=orbits, t_obs=duration, t0=0.0, n=args.response_bins, extension=0.5)

    # Warm the JAX paths before timing/evaluation.
    eccentric_links_jax(
        source,
        geometry,
        batch_size=1,
        physics_mode=args.physics_mode,
        evolution_mode=args.evolution_mode,
        pm_mismatch_tolerance=args.pm_mismatch_tolerance,
    )
    jaxgbaccurate_xyz_time(jaxgb, as_jaxgb_params(jaxgb_source), dt=dt, obs_samples=args.samples)

    jaxgb_xyz, jaxgb_s = elapsed(
        lambda: jaxgbaccurate_xyz_time(jaxgb, as_jaxgb_params(jaxgb_source), dt=dt, obs_samples=args.samples)
    )
    links, link_s = elapsed(
        lambda: eccentric_links_jax(
            source,
            geometry,
            batch_size=1,
            physics_mode=args.physics_mode,
            evolution_mode=args.evolution_mode,
            pm_mismatch_tolerance=args.pm_mismatch_tolerance,
        )
    )
    egb_xyz, tdi_s = elapsed(lambda: xyz_from_links(state, links))

    jaxgb_aet = aet_from_xyz(jaxgb_xyz)
    egb_aet = aet_from_xyz(egb_xyz)

    csv_path = BENCHMARKS / f"{args.output_prefix}.csv"
    png_path = BENCHMARKS / f"{args.output_prefix}.png"
    paper_png_path = FIGURES_FINAL / png_path.name
    write_csv(csv_path, days, jaxgb_aet, egb_aet, jaxgb_real_factor=args.jaxgb_real_factor)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.2), sharex=True, constrained_layout=True)
    for ax, channel in zip(axes, ("A", "E", "T"), strict=True):
        jaxgb_real = args.jaxgb_real_factor * np.real(jaxgb_aet[channel])
        egb_real = np.real(egb_aet[channel])
        if args.plot_mode == "full-band":
            years = days / 365.25
            x_j, j_low, j_high = binned_minmax(years, jaxgb_real, args.band_bins)
            x_e, e_low, e_high = binned_minmax(years, egb_real, args.band_bins)
            ax.fill_between(x_j, j_low, j_high, color="#0072B2", alpha=0.32, linewidth=0.0, label="JAXGBaccurate")
            ax.fill_between(x_e, e_low, e_high, color="#D55E00", alpha=0.30, linewidth=0.0, label="eGB-multi TD")
            ax.plot(x_j, 0.5 * (j_low + j_high), color="#0072B2", linewidth=0.7, alpha=0.75)
            ax.plot(x_e, 0.5 * (e_low + e_high), color="#D55E00", linewidth=0.7, alpha=0.75)
        else:
            mask = days <= args.window_days
            ax.plot(days[mask], jaxgb_real[mask], color="#0072B2", linewidth=1.5, label="JAXGBaccurate")
            ax.plot(days[mask], egb_real[mask], color="#D55E00", linewidth=1.15, alpha=0.9, label="eGB-multi TD")
        ax.set_ylabel(channel)
        ax.grid(True, alpha=0.28)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[0].set_title(
        rf"A/E/T response comparison, $f_0={args.f0:.0e}\,\mathrm{{Hz}}$, "
        rf"$e_0=0$, {args.duration_days:g} d"
    )
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    axes[-1].set_xlabel("time [yr]" if args.plot_mode == "full-band" else "time [days]")
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    FIGURES_FINAL.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png_path, paper_png_path)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"wrote {paper_png_path}")
    print(f"JAXGBaccurate wall time: {jaxgb_s:.6f} s")
    print(f"eGB-multi wall time: {link_s + tdi_s:.6f} s (links {link_s:.6f} s + TDI {tdi_s:.6f} s)")
    print(f"eGB-multi evolution mode: {args.evolution_mode}")
    print(f"JAXGBaccurate fdot: {jaxgb_source.fdot:.6e} Hz/s ({args.jaxgb_fdot_mode})")


if __name__ == "__main__":
    main()
