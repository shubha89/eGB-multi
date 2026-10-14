"""Production grid: can circular chirping templates absorb eccentricity?

This is the conservative follow-up to
``production_eccentric_circular_mismatch_grid.py``. It keeps the eccentric
signal model in the JAX brute-force time-domain response, but projects both
signal and circular template through second-generation pyTDI Michelson XYZ.

The circular template can include vacuum-GR circular ``fdot`` computed from
the component masses. The ``grid`` mode additionally tries a small set of
multiplicative factors around the GR value and keeps the best fitting factor.

Run examples:

    python benchmarks/production_eccentric_circular_fdot_tdi2_mismatch_grid.py --preset smoke
    python benchmarks/production_eccentric_circular_fdot_tdi2_mismatch_grid.py --preset production --circular-fdot-mode gr
    python benchmarks/production_eccentric_circular_fdot_tdi2_mismatch_grid.py --preset production \
        --signal-fdot-mode none --signal-evolution-mode auto --pm-mismatch-tolerance 1e-2 \
        --circular-fdot-mode wide_grid
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from production_eccentric_circular_mismatch_grid import (
    compare_aet,
    grid_for_preset as base_grid_for_preset,
    load_aet_psd,
    plot_summary,
)

from egb_multi.constants import C_M_PER_S, G_M3_KG_S2, SOLAR_MASS_KG
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry, resolve_response_evolution_modes
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.noise import aet_from_xyz
from egb_multi.tdi import pytdi_data_from_links


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("smoke", "pilot", "coarse", "production"), default="smoke")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--measurement-order", type=int, default=3)
    parser.add_argument("--delay-order", type=int, default=3)
    parser.add_argument("--tdi-generation", type=int, default=2)
    parser.add_argument("--circular-fdot-mode", choices=("none", "gr", "grid", "wide_grid"), default="gr")
    parser.add_argument("--signal-fdot-mode", choices=("none", "gr"), default="gr")
    parser.add_argument("--signal-evolution-mode", choices=("fixed", "peters_mathews", "auto"), default="fixed")
    parser.add_argument("--pm-mismatch-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--resume", action="store_true", help="Append to an existing CSV and skip completed cases.")
    return parser.parse_args()


def grid_for_preset(preset: str) -> dict[str, object]:
    if preset != "smoke":
        return base_grid_for_preset(preset)
    return {
        "duration_days": 7.0,
        "dt_s": 100.0,
        "f0_hz": [1.0e-3, 3.0e-3],
        "eccentricity": [0.0, 0.01, 0.05],
        "inclination": [0.8],
        "sky": [(0.3, -0.7)],
        "masses": [(0.5, 0.5), (0.8, 0.6)],
    }


def chirp_mass_solar(m1_solar: float, m2_solar: float) -> float:
    total = m1_solar + m2_solar
    return (m1_solar * m2_solar) ** (3.0 / 5.0) / total ** (1.0 / 5.0)


def circular_gr_fdot_hz_s(f0_hz: float, m1_solar: float, m2_solar: float) -> float:
    """Return vacuum-GR circular ``fdot`` for GW frequency ``f0_hz``."""

    chirp_mass_kg = chirp_mass_solar(m1_solar, m2_solar) * SOLAR_MASS_KG
    return float(
        (96.0 / 5.0)
        * np.pi ** (8.0 / 3.0)
        * (G_M3_KG_S2 * chirp_mass_kg / C_M_PER_S**3) ** (5.0 / 3.0)
        * f0_hz ** (11.0 / 3.0)
    )


def make_source(
    *,
    f0_hz: float,
    eccentricity: float,
    inclination: float,
    beta: float,
    lambda_: float,
    m1_solar: float,
    m2_solar: float,
    fdot: float = 0.0,
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
        fdot=fdot,
    )


def xyz_from_links_generation(
    state,
    links: dict[str, np.ndarray],
    *,
    generation: int,
    measurement_order: int,
    delay_order: int,
) -> dict[str, np.ndarray]:
    from pytdi.michelson import compute_factorized_michelson

    data = pytdi_data_from_links(state, links)
    return {
        "X": compute_factorized_michelson(
            data,
            rot=0,
            order=measurement_order,
            delay_order=delay_order,
            generation=generation,
            unit="frequency",
        ),
        "Y": compute_factorized_michelson(
            data,
            rot=1,
            order=measurement_order,
            delay_order=delay_order,
            generation=generation,
            unit="frequency",
        ),
        "Z": compute_factorized_michelson(
            data,
            rot=2,
            order=measurement_order,
            delay_order=delay_order,
            generation=generation,
            unit="frequency",
        ),
    }


def response_aet(
    source: EccentricBinaryParams,
    geometry,
    state,
    *,
    batch_size: int,
    generation: int,
    measurement_order: int,
    delay_order: int,
    evolution_mode: str = "fixed",
    pm_mismatch_tolerance: float = 1.0e-2,
) -> dict[str, np.ndarray]:
    links = eccentric_links_jax(
        source,
        geometry,
        batch_size=batch_size,
        physics_mode="1pn",
        evolution_mode=evolution_mode,
        pm_mismatch_tolerance=pm_mismatch_tolerance,
    )
    xyz = xyz_from_links_generation(
        state,
        links,
        generation=generation,
        measurement_order=measurement_order,
        delay_order=delay_order,
    )
    return aet_from_xyz(xyz)


def fdot_candidates(mode: str, gr_fdot: float) -> list[float]:
    if mode == "none":
        return [0.0]
    if mode == "gr":
        return [gr_fdot]
    if gr_fdot == 0.0:
        return [0.0]
    factors = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    if mode == "wide_grid":
        factors.extend([8.0, 16.0, 32.0])
    return [factor * gr_fdot for factor in factors]


def tolerance_label(tolerance: float) -> str:
    return f"{tolerance:.0e}".replace("+", "p").replace("-", "m")


def case_key(
    f0_hz: float,
    eccentricity: float,
    inclination: float,
    beta: float,
    lambda_: float,
    m1_solar: float,
    m2_solar: float,
) -> tuple[float, ...]:
    return (
        float(f0_hz),
        float(eccentricity),
        float(inclination),
        float(beta),
        float(lambda_),
        float(m1_solar),
        float(m2_solar),
    )


def row_case_key(row: dict[str, object]) -> tuple[float, ...]:
    return case_key(
        float(row["f0_hz"]),
        float(row["eccentricity"]),
        float(row["inclination"]),
        float(row["beta"]),
        float(row["lambda"]),
        float(row["m1_solar"]),
        float(row["m2_solar"]),
    )


def main() -> None:
    args = parse_args()
    config = grid_for_preset(args.preset)
    out_dir = Path(__file__).resolve().parent
    out_prefix = out_dir / (
        f"production_eccentric_circular_fdot_tdi{args.tdi_generation}_"
        f"signal_{args.signal_fdot_mode}_{args.signal_evolution_mode}_pm{tolerance_label(args.pm_mismatch_tolerance)}_"
        f"template_{args.circular_fdot_mode}_{args.preset}"
    )
    psd_freq, psd_base = load_aet_psd(out_dir / "lisainstrument_welch_psd.csv")

    duration = float(config["duration_days"]) * 86400.0
    dt = float(config["dt_s"])
    times = np.arange(0.0, duration, dt, dtype=np.float64)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)
    freq = np.fft.rfftfreq(times.size, dt)
    psd = {channel: np.interp(freq, psd_freq, psd_base[channel], left=np.inf, right=np.inf) for channel in ("A", "E", "T")}

    axes = list(itertools.product(config["f0_hz"], config["inclination"], config["sky"], config["masses"]))
    total_cases = len(axes) * len(config["eccentricity"])
    print(
        f"preset={args.preset}, tdi_generation={args.tdi_generation}, signal_fdot_mode={args.signal_fdot_mode}, "
        f"signal_evolution_mode={args.signal_evolution_mode}, pm_mismatch_tolerance={args.pm_mismatch_tolerance}, "
        f"circular_fdot_mode={args.circular_fdot_mode}, "
        f"samples={times.size}, duration_days={config['duration_days']}, dt={dt}, "
        f"base_points={len(axes)}, eccentricities={len(config['eccentricity'])}, total_cases={total_cases}",
        flush=True,
    )

    compile_source = make_source(
        f0_hz=float(config["f0_hz"][0]),
        eccentricity=0.0,
        inclination=float(config["inclination"][0]),
        beta=float(config["sky"][0][0]),
        lambda_=float(config["sky"][0][1]),
        m1_solar=float(config["masses"][0][0]),
        m2_solar=float(config["masses"][0][1]),
    )
    response_aet(
        compile_source,
        geometry,
        state,
        batch_size=args.batch_size,
        generation=args.tdi_generation,
        measurement_order=args.measurement_order,
        delay_order=args.delay_order,
    )

    csv_path = out_prefix.with_suffix(".csv")
    rows: list[dict[str, object]] = []
    completed_cases: set[tuple[float, ...]] = set()
    existing_fieldnames: list[str] | None = None
    if args.resume and csv_path.exists():
        with csv_path.open(newline="") as existing_file:
            reader = csv.DictReader(existing_file)
            existing_fieldnames = reader.fieldnames
            rows = list(reader)
        completed_cases = {row_case_key(row) for row in rows}
        print(f"resume=true, loaded_completed_cases={len(completed_cases)} from {csv_path}", flush=True)

    csv_file = None
    writer = None
    start_all = time.perf_counter()
    case_index = 0
    try:
        for f0_hz, inclination, sky, masses in axes:
            beta, lambda_ = sky
            m1_solar, m2_solar = masses
            axis_case_keys = [
                case_key(
                    float(f0_hz),
                    float(eccentricity),
                    float(inclination),
                    float(beta),
                    float(lambda_),
                    float(m1_solar),
                    float(m2_solar),
                )
                for eccentricity in config["eccentricity"]
            ]
            if axis_case_keys and all(key in completed_cases for key in axis_case_keys):
                case_index += len(axis_case_keys)
                print(
                    f"[{case_index:04d}/{total_cases}] skip completed axis f0={float(f0_hz):.1e} "
                    f"m={float(m1_solar):.1f}+{float(m2_solar):.1f}",
                    flush=True,
                )
                continue

            gr_fdot = circular_gr_fdot_hz_s(float(f0_hz), float(m1_solar), float(m2_solar))
            signal_fdot = gr_fdot if args.signal_fdot_mode == "gr" else 0.0
            circular_templates = []
            circular_start = time.perf_counter()
            for candidate_fdot in fdot_candidates(args.circular_fdot_mode, gr_fdot):
                circular_source = make_source(
                    f0_hz=float(f0_hz),
                    eccentricity=0.0,
                    inclination=float(inclination),
                    beta=float(beta),
                    lambda_=float(lambda_),
                    m1_solar=float(m1_solar),
                    m2_solar=float(m2_solar),
                    fdot=float(candidate_fdot),
                )
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
            band_mask = (freq >= 0.20 * float(f0_hz)) & (freq <= min(freq[-1], 10.0 * float(f0_hz)))

            for eccentricity in config["eccentricity"]:
                case_index += 1
                current_key = case_key(
                    float(f0_hz),
                    float(eccentricity),
                    float(inclination),
                    float(beta),
                    float(lambda_),
                    float(m1_solar),
                    float(m2_solar),
                )
                if current_key in completed_cases:
                    print(
                        f"[{case_index:04d}/{total_cases}] skip f0={float(f0_hz):.1e} "
                        f"e={float(eccentricity):.3f} m={float(m1_solar):.1f}+{float(m2_solar):.1f}",
                        flush=True,
                    )
                    continue

                eccentric_source = make_source(
                    f0_hz=float(f0_hz),
                    eccentricity=float(eccentricity),
                    inclination=float(inclination),
                    beta=float(beta),
                    lambda_=float(lambda_),
                    m1_solar=float(m1_solar),
                    m2_solar=float(m2_solar),
                    fdot=signal_fdot,
                )
                start = time.perf_counter()
                resolved_signal_evolution_mode = resolve_response_evolution_modes(
                    eccentric_source,
                    times,
                    physics_mode="1pn",
                    evolution_mode=args.signal_evolution_mode,
                    pm_mismatch_tolerance=args.pm_mismatch_tolerance,
                )[0]
                eccentric_aet = response_aet(
                    eccentric_source,
                    geometry,
                    state,
                    batch_size=args.batch_size,
                    generation=args.tdi_generation,
                    measurement_order=args.measurement_order,
                    delay_order=args.delay_order,
                    evolution_mode=resolved_signal_evolution_mode,
                    pm_mismatch_tolerance=args.pm_mismatch_tolerance,
                )
                eccentric_wall = time.perf_counter() - start

                best_metrics = None
                best_fdot = None
                for candidate_fdot, circular_aet in circular_templates:
                    metrics = compare_aet(eccentric_aet, circular_aet, psd, dt, band_mask.copy())
                    if best_metrics is None or metrics["fitting_factor_AE"] > best_metrics["fitting_factor_AE"]:
                        best_metrics = metrics
                        best_fdot = candidate_fdot
                assert best_metrics is not None
                assert best_fdot is not None

                row = {
                    "preset": args.preset,
                    "tdi_generation": args.tdi_generation,
                    "signal_fdot_mode": args.signal_fdot_mode,
                    "signal_evolution_mode": args.signal_evolution_mode,
                    "resolved_signal_evolution_mode": resolved_signal_evolution_mode,
                    "pm_mismatch_tolerance": args.pm_mismatch_tolerance,
                    "circular_fdot_mode": args.circular_fdot_mode,
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
                    "chirp_mass_solar": chirp_mass_solar(float(m1_solar), float(m2_solar)),
                    "circular_gr_fdot_hz_s": gr_fdot,
                    "signal_fdot_hz_s": signal_fdot,
                    "best_circular_fdot_hz_s": best_fdot,
                    "circular_template_count": len(circular_templates),
                    "circular_wall_s": circular_wall,
                    "eccentric_wall_s": eccentric_wall,
                    **best_metrics,
                }
                if writer is None:
                    if args.resume and csv_path.exists():
                        csv_file = csv_path.open("a", newline="")
                        writer = csv.DictWriter(csv_file, fieldnames=existing_fieldnames or list(row))
                    else:
                        csv_file = csv_path.open("w", newline="")
                        writer = csv.DictWriter(csv_file, fieldnames=list(row))
                        writer.writeheader()
                writer.writerow(row)
                csv_file.flush()
                rows.append(row)
                completed_cases.add(current_key)
                print(
                    f"[{case_index:04d}/{total_cases}] f0={float(f0_hz):.1e} e={float(eccentricity):.3f} "
                    f"m={float(m1_solar):.1f}+{float(m2_solar):.1f} evo={resolved_signal_evolution_mode} "
                    f"fdot_best={best_fdot:.3e} "
                    f"FF={best_metrics['fitting_factor_AE']:.8f} mismatch={best_metrics['mismatch_AE']:.3e} "
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


if __name__ == "__main__":
    main()
