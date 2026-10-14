"""Preserve the legacy comparison and audit the Task 2.1/2.2 conventions.

Run from this checkout with ``python benchmarks/referee_task_2_audit.py``.
Waveforms are imported exclusively from this checkout's src/egb_multi.
This is a definitions/parameter audit, not a production fitting-factor search.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# Imported legacy plotting modules need a writable cache even for this audit.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/referee/.cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs/referee/.cache"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import egb_multi
import numpy as np
from scipy.optimize import minimize_scalar

from egb_multi.constants import C_M_PER_S, G_M3_KG_S2
from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain, peters_mathews_derivatives
from egb_multi.eccentric_jax import (
    eccentric_complex_strain_batch_jax,
    precompute_jax_link_geometry,
    resolve_response_evolution_modes,
)
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from production_eccentric_circular_fdot_tdi2_mismatch_grid import circular_gr_fdot_hz_s, make_source, response_aet
from production_eccentric_circular_mismatch_grid import aet_inner, compare_aet, fd_aet, load_aet_psd

LEGACY_CSV = ROOT / "benchmarks/production_eccentric_circular_fdot_tdi2_signal_none_auto_pm1em02_template_wide_grid_production.csv"
PSD_CSV = ROOT / "benchmarks/lisainstrument_welch_psd.csv"
BASELINE_CASES = ((1e-4, .1), (1e-3, 0.), (1e-3, .1), (3e-3, .1), (3e-3, .3))


def assert_local_package() -> None:
    """Reject an installed or alternate-checkout eGB implementation."""
    expected = (ROOT / "src/egb_multi").resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "egb_multi" or name.startswith("egb_multi."):
            module_file = getattr(module, "__file__", None)
            if module_file is not None and not Path(module_file).resolve().is_relative_to(expected):
                raise RuntimeError(f"Nonlocal eGB module: {name}: {module_file}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rereference_circular(source: EccentricBinaryParams, new_t0: float) -> EccentricBinaryParams:
    """Change a circular fixed-JAX source epoch without changing its signal.

    phi0 is orbital phase; the GW phase is 2*phi0. Detector times stay fixed.
    This identity applies to the prescribed linear-fdot JAX model, not to PM.
    """
    if source.eccentricity != 0.0:
        raise ValueError("This epoch identity is only for circular sources")
    delta = new_t0 - source.t0
    new_frequency = source.mean_motion + np.pi * source.fdot * delta
    if not np.isfinite(new_frequency) or new_frequency <= 0:
        raise ValueError("The re-referenced frequency must be finite and positive")
    phase = source.phi0 + source.mean_motion * delta + .5 * np.pi * source.fdot * delta**2
    return replace(source, mean_motion=float(new_frequency), phi0=float(phase % np.pi), t0=float(new_t0))


def parameter_contract(source: EccentricBinaryParams, duration: float, f_low: float, f_high: float) -> dict:
    """Explicit initial domain for the paper's detached WD-like comparisons.

    Bounds are analysis choices, not a claimed universal astrophysical prior.
    Search-bound expansion and sampling convergence remain Tasks 2.6/2.8.
    """
    f0 = source.mean_motion / np.pi
    if not (np.isfinite(duration) and duration > 0 and 0 < f_low < f_high and f0 > 0):
        raise ValueError("Invalid observation duration or frequency band")
    gr = circular_gr_fdot_hz_s(f0, source.m1_solar, source.m2_solar)
    ndot, _ = peters_mathews_derivatives(source.mean_motion, source.eccentricity, source.total_mass_kg, source.symmetric_mass_ratio)
    bound = max(64.0 * gr, 2.0 * f_high / f0 * abs(float(ndot) / np.pi), 4.0 / duration**2)
    return {
        "status": "defined_initial_domain_not_yet_optimization_validated",
        "waveform_package": str(Path(egb_multi.__file__).resolve()),
        "physics_mode": "1pn_periastron",
        "template_evolution_mode": "fixed",
        "template_backend": "eccentric_links_jax_fixed_kernel",
        "eccentricity": 0.0,
        "observation_duration_s": duration,
        "reference_epoch_s": source.t0 + .5 * duration,
        "frequency_is": "azimuthal_GW_carrier_mean_motion_over_pi",
        "phase_is": "orbital_phi0_GW_phase_is_twice_phi0",
        "band_hz": [f_low, f_high],
        "band_rule": "fixed_per_injection_independent_of_candidate_and_cadence",
        "bounds": {
            "carrier_frequency_hz": [f_low, f_high],
            "fdot_hz_s": [-bound, bound],
            "sky_latitude_rad": [-.5 * np.pi, .5 * np.pi],
            "sky_longitude_rad": [0., 2. * np.pi],
            "cos_inclination": [-1., 1.],
            "polarization_rad": [0., np.pi],
            "orbital_phase_rad": [0., np.pi],
            "component_masses_solar": [.1, 1.4],
        },
        "periodic_upper_endpoints_excluded": ["sky_longitude_rad", "polarization_rad", "orbital_phase_rad"],
        "mass_ordering": "m1_solar >= m2_solar",
        "mass_treatment": "retained_in_exact_1pn_chirping_family_reduction_requires_error_budget",
        "amplitude_treatment": "positive_common_distance_scale_profiled_using_real_AE_inner_product",
        "fdot_bound_rule": "max(64*injection_circular_GR_fdot,2*(f_high/f0)*injection_PM_fdot,4/T**2)",
        "candidate_validity": "positive_instantaneous_carrier_over_all_retarded_evaluation_times",
        "relative_time_treatment": "source_epoch_gauge_fixed_detector_epoch_not_shifted",
        "coherence": "same_source_parameters_in_A_and_E",
        "followup_gates": ["phase_and_orientation_maximization", "search_domain_expansion", "cadence_bandwidth_TDI_convergence"],
    }


def physical_phase_maximum(gram: np.ndarray, projections: np.ndarray, signal_norm: float) -> tuple[float, float]:
    """Diagnostic physical phase search for h(theta)=R+C cos(theta)+S sin(theta).

    Includes a phase-independent term and the full real Gram matrix. Only the
    circular phase is varied; this is not a fitting factor or sky search.
    """
    gram = np.asarray(gram, dtype=float)
    projections = np.asarray(projections, dtype=float)
    if gram.shape != (3, 3) or projections.shape != (3,) or not signal_norm > 0:
        raise ValueError("Invalid phase-basis dimensions or signal norm")

    def overlap(theta):
        theta = np.asarray(theta)
        coeff = np.stack((np.ones_like(theta), np.cos(theta), np.sin(theta)), axis=-1)
        norm = np.einsum("...i,ij,...j->...", coeff, gram, coeff)
        if np.any(norm <= 0) or np.any(~np.isfinite(norm)):
            raise ValueError("Nonpositive phase-template norm")
        return (coeff @ projections) / np.sqrt(signal_norm * norm)

    step = 2. * np.pi / 4096
    grid = np.arange(4096) * step
    values = overlap(grid)
    maxima = np.flatnonzero((values >= np.roll(values, 1)) & (values >= np.roll(values, -1)))
    # Keep the best sampled point as a candidate even when the objective is flat.
    candidates = [(float(values.max()), float(grid[values.argmax()]))]
    for index in sorted(maxima, key=lambda i: values[i], reverse=True)[:8]:
        result = minimize_scalar(lambda theta: -float(overlap(theta)), bounds=(grid[index] - step, grid[index] + step), method="bounded", options={"xatol": 1e-13})
        if not result.success:
            raise RuntimeError("Diagnostic phase refinement failed")
        candidates.append((-float(result.fun), float(result.x % (2. * np.pi))))
    return max(candidates)


def create_snapshot(out: Path, rows: list[dict], reference_documents: list[Path] | None = None) -> dict:
    inputs = [LEGACY_CSV, PSD_CSV, ROOT / "pyproject.toml", ROOT / "README.md"]
    inputs += sorted((ROOT / "src/egb_multi").glob("*.py"))
    inputs += sorted((ROOT / "benchmarks").glob("*.py"))
    inputs += [ROOT / "reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex", ROOT / "reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.bib"]
    hashes = {}
    for path in inputs:
        relative = path.relative_to(ROOT)
        target = out / "snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = sha256(path)
    selected = [row for row in rows if float(row["inclination"]) == .2 and float(row["beta"]) == .2 and float(row["lambda"]) == -1. and float(row["f0_hz"]) in (1e-4, 1e-3, 3e-3) and float(row["eccentricity"]) in (0., .001, .1, .3, .5)]
    write_csv(out / "preserved_representative_rows.csv", selected)
    documents = {}
    for path in reference_documents or []:
        path = path.resolve()
        target = out / "snapshot/reference_documents" / path.name
        if target.exists():
            raise ValueError(f"Duplicate reference document filename: {path.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        documents[path.name] = {"source_path": str(path), "sha256": sha256(path)}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "egb_module": str(Path(egb_multi.__file__).resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "versions": {name: version(name) for name in ("numpy", "scipy", "jax", "jaxlib", "lisaorbits", "pytdi", "matplotlib")},
        "original_csv_rows": len(rows),
        "preserved_representative_rows": len(selected),
        "sha256": hashes,
        "reference_documents": documents,
        "original_code_commit": None,
        "provenance_note": "Checkout has no git metadata; hashes capture current files, not historical authoring provenance. Local TeX differs from submitted PDF.",
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def run_audit(out: Path, rows: list[dict]) -> dict:
    duration = 365.25 * 86400.
    dt = 50.
    times = np.arange(0., duration, dt)
    # This is an archival reproduction, including the historical frame
    # omission. Corrected production runs use the default ecliptic positions.
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times,frame='icrs')
    geometry = precompute_jax_link_geometry(state)
    freq = np.fft.rfftfreq(times.size, dt)
    psd_freq, psd_base = load_aet_psd(PSD_CSV)
    psd = {ch: np.interp(freq, psd_freq, psd_base[ch], left=np.inf, right=np.inf) for ch in ("A", "E", "T")}
    df = 1. / duration

    def response(source, mode="fixed"):
        result = response_aet(source, geometry, state, batch_size=1, generation=2, measurement_order=3, delay_order=3, evolution_mode=mode)
        if any(np.any(~np.isfinite(series)) for series in result.values()):
            raise ValueError("Nonfinite response")
        return result

    def inner(left, right, mask):
        return aet_inner(left, right, psd, df, mask.copy()).real

    def fitted_residual(reference, candidate, mask):
        ee = inner(reference, reference, mask)
        cc = inner(candidate, candidate, mask)
        scale = inner(reference, candidate, mask) / cc
        residual = {ch: reference[ch] - scale * candidate[ch] for ch in reference}
        return float(scale), float(np.sqrt(inner(residual, residual, mask) / ee))

    baseline = []
    contracts = []
    phase_evidence = []
    for f0, eccentricity in BASELINE_CASES:
        row = next(r for r in rows if float(r["f0_hz"]) == f0 and float(r["eccentricity"]) == eccentricity and float(r["inclination"]) == .2 and float(r["beta"]) == .2 and float(r["lambda"]) == -1. and float(r["m1_solar"]) == .5 and float(r["m2_solar"]) == .5)
        signal = make_source(f0_hz=f0, eccentricity=eccentricity, inclination=.2, beta=.2, lambda_=-1., m1_solar=.5, m2_solar=.5)
        template = replace(signal, eccentricity=0., fdot=float(row["best_circular_fdot_hz_s"]))
        mode = resolve_response_evolution_modes(signal, times, evolution_mode="auto", pm_mismatch_tolerance=.01)[0]
        if mode != row["resolved_signal_evolution_mode"]:
            raise RuntimeError("Legacy signal evolution selection changed")
        print(f"baseline f0={f0:g} e={eccentricity:g} mode={mode}", flush=True)
        start = time.perf_counter()
        data_aet = response(signal, mode)
        circ_aet = response(template)
        mask = (freq >= .2 * f0) & (freq <= min(freq[-1], 10. * f0))
        metrics = compare_aet(data_aet, circ_aet, psd, dt, mask.copy())
        data_fd, circ_fd = fd_aet(data_aet, dt), fd_aet(circ_aet, dt)
        real_overlap = inner(data_fd, circ_fd, mask) / np.sqrt(inner(data_fd, data_fd, mask) * inner(circ_fd, circ_fd, mask))
        baseline.append({"f0_hz": f0, "eccentricity": eccentricity, "signal_mode": mode, "stored_best_fdot_hz_s": template.fdot, "stored_legacy_score": float(row["fitting_factor_AE"]), "rerun_legacy_score": metrics["fitting_factor_AE"], "legacy_score_absolute_difference": abs(metrics["fitting_factor_AE"] - float(row["fitting_factor_AE"])), "real_overlap": float(real_overlap), "absolute_real_overlap": float(abs(real_overlap)), "rerun_raw_weighted_residual": metrics["noise_weighted_residual_AE"], "stored_raw_weighted_residual": float(row["noise_weighted_residual_AE"]), "wall_s": time.perf_counter() - start})
        shifted = [fd_aet(response(replace(template, phi0=template.phi0 + delta)), dt) for delta in (.25 * np.pi, .5 * np.pi)]
        offset = {ch: .5 * (circ_fd[ch] + shifted[1][ch]) for ch in circ_fd}
        cos_basis = {ch: .5 * (circ_fd[ch] - shifted[1][ch]) for ch in circ_fd}
        sin_basis = {ch: shifted[0][ch] - offset[ch] for ch in circ_fd}
        basis = [offset, cos_basis, sin_basis]
        gram = np.array([[inner(a, b, mask) for b in basis] for a in basis])
        projections = np.array([inner(data_fd, b, mask) for b in basis])
        phase_max, theta = physical_phase_maximum(gram, projections, inner(data_fd, data_fd, mask))
        phase_evidence.append({"f0_hz": f0, "eccentricity": eccentricity, "legacy_complex_magnitude": metrics["fitting_factor_AE"], "physical_phase_maximum_at_stored_fdot": phase_max, "difference_physical_minus_legacy": phase_max - metrics["fitting_factor_AE"], "best_orbital_phase_shift_rad": .5 * theta, "quadrature_norm_ratio": float(gram[2, 2] / gram[1, 1]), "quadrature_normalized_cross_inner": float(gram[1, 2] / np.sqrt(gram[1, 1] * gram[2, 2]))})
        if f0 == 1e-3 and eccentricity == .1:
            explicit = fd_aet(response(replace(template, phi0=template.phi0 + .413)), dt)
            reconstructed = {ch: offset[ch] + np.cos(2. * .413) * cos_basis[ch] + np.sin(2. * .413) * sin_basis[ch] for ch in circ_fd}
            _, reconstruction_error = fitted_residual(explicit, reconstructed, mask)
            phase_evidence[-1]["explicit_phase_basis_weighted_relative_error"] = reconstruction_error
        contracts.append({"injection": asdict(signal), "domain": parameter_contract(signal, duration, .2 * f0, 10. * f0)})
        write_csv(out / "baseline_rerun.csv", baseline)
        write_json(out / "phase_convention_audit.json", phase_evidence)
        write_json(out / "circular_parameter_contract.json", contracts)
        print(f"  legacy delta={baseline[-1]['legacy_score_absolute_difference']:.3g}; phase-minus-legacy={phase_evidence[-1]['difference_physical_minus_legacy']:.3g}", flush=True)
        del data_aet, circ_aet, data_fd, circ_fd, shifted, basis, offset, cos_basis, sin_basis
        gc.collect()

    # Use the same full-year, same-cadence TDI path for parameter identities.
    f0 = 3e-3
    source = make_source(f0_hz=f0, eccentricity=0., inclination=.8, beta=.2, lambda_=-1., m1_solar=.5, m2_solar=.5)
    chirp = 32. * circular_gr_fdot_hz_s(f0, .5, .5)
    mask = (freq >= .2 * f0) & (freq <= min(freq[-1], 10. * f0))
    mass_rows = []
    epoch_results = {}
    for fdot in (0., chirp):
        base = replace(source, fdot=fdot)
        base_fd = fd_aet(response(base), dt)
        for m1, m2 in ((.8, .6), (1., .3), (1.4, .1), (1.4, 1.4)):
            print(f"mass audit fdot={fdot:.3g} m={m1:g}+{m2:g}", flush=True)
            candidate = replace(base, m1_solar=m1, m2_solar=m2)
            other_fd = fd_aet(response(candidate), dt)
            scale, residual = fitted_residual(base_fd, other_fd, mask)
            mass_rows.append({"f0_hz": f0, "fdot_hz_s": fdot, "base_m1_solar": .5, "base_m2_solar": .5, "candidate_m1_solar": m1, "candidate_m2_solar": m2, "best_positive_amplitude_scale": scale, "weighted_relative_residual_after_amplitude_fit": residual})
            write_csv(out / "mass_shape_audit.csv", mass_rows)
        if fdot != 0.:
            shifted_epoch = rereference_circular(base, .5 * duration)
            _, epoch_error = fitted_residual(base_fd, fd_aet(response(shifted_epoch), dt), mask)
            _, distance_error = fitted_residual(base_fd, fd_aet(response(replace(base, distance_m=2. * base.distance_m)), dt), mask)
            numpy_strain = eccentric_complex_strain(base, times)
            jax_strain = eccentric_complex_strain_batch_jax(base, times)[0]
            zero_jax = eccentric_complex_strain_batch_jax(replace(base, fdot=0.), times)[0]
            epoch_results = {"f0_hz": f0, "fdot_hz_s": fdot, "new_epoch_s": shifted_epoch.t0, "new_f0_hz": shifted_epoch.mean_motion / np.pi, "new_orbital_phase_rad": shifted_epoch.phi0, "epoch_weighted_relative_residual": epoch_error, "distance_weighted_relative_residual_after_amplitude_fit": distance_error, "numpy_vs_jax_prescribed_fdot_source_relative_l2": float(np.linalg.norm(numpy_strain - jax_strain) / np.linalg.norm(jax_strain)), "numpy_vs_zero_fdot_jax_source_relative_l2": float(np.linalg.norm(numpy_strain - zero_jax) / np.linalg.norm(zero_jax)), "numpy_fdot_contract": "NumPy fixed eccentric_dynamics ignores fdot; use JAX for prescribed circular chirp"}
            write_json(out / "epoch_and_backend_audit.json", epoch_results)
        del base_fd
        gc.collect()

    failures = []
    if max(r["legacy_score_absolute_difference"] for r in baseline) > 1e-8:
        failures.append("Stored-best-template legacy reproduction exceeds 1e-8 absolute score tolerance")
    if phase_evidence[2]["explicit_phase_basis_weighted_relative_error"] > 1e-8:
        failures.append("Physical phase basis reconstruction exceeds 1e-8 relative tolerance")
    if epoch_results["epoch_weighted_relative_residual"] > 1e-7:
        failures.append("Source epoch identity exceeds 1e-7 weighted relative tolerance")
    if epoch_results["distance_weighted_relative_residual_after_amplitude_fit"] > 1e-10:
        failures.append("Distance scaling exceeds 1e-10 weighted relative tolerance")
    result = {"status": "passed_audit_checks" if not failures else "discrepancy_requires_investigation", "failures": failures, "duration_days": 365.25, "dt_s": dt, "samples": times.size, "tdi_generation": 2, "measurement_order": 3, "delay_order": 3, "baseline_cases": len(baseline), "max_legacy_score_absolute_difference": max(r["legacy_score_absolute_difference"] for r in baseline), "max_phase_score_absolute_difference": max(abs(r["difference_physical_minus_legacy"]) for r in phase_evidence), "mass_no_chirp_max_relative_residual": max(r["weighted_relative_residual_after_amplitude_fit"] for r in mass_rows if r["fdot_hz_s"] == 0.), "mass_chirp_max_relative_residual": max(r["weighted_relative_residual_after_amplitude_fit"] for r in mass_rows if r["fdot_hz_s"] != 0.), "epoch_and_backend": epoch_results, "limitations": ["Stored best templates rerun; nine-point grid and full 2160-case campaign not reoptimized", "50 s legacy cadence and Nyquist-truncated legacy masks deliberately preserved, not convergence certified", "Diagnostic phase maximization only, not full template fitting factor", "Mass identities tested at selected 3 mHz cases, not over every future search-domain point", "Initial parameter bounds require expansion checks before production"]}
    write_json(out / "audit_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/referee/task_2_1_2_2_20261007")
    parser.add_argument("--archive-only", action="store_true", help="Preserve inputs and optional reference PDFs without numerical reruns.")
    parser.add_argument("--reference-document", type=Path, action="append", default=[], help="Copy a supplied reference into the local snapshot and record its hash.")
    args = parser.parse_args()
    assert_local_package()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("All audit outputs must be inside this project")
    out.mkdir(parents=True, exist_ok=False)
    rows = list(csv.DictReader(LEGACY_CSV.open()))
    create_snapshot(out, rows, args.reference_document)
    if args.archive_only:
        print(f"Preserved snapshot at {out}", flush=True)
        return
    result = run_audit(out, rows)
    assert_local_package()
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
