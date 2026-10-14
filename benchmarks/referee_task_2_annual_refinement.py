"""Reoptimize saved annual circular-family fits under declared refinements.

Each level has one immutable physical band, covariance and observation. All
six intrinsic coordinates stay free, and old candidates are re-evaluated.
Local refinement is explicitly distinguished from a new full multibasin run.
Snapshots, input hashes, per-evaluation logs and native winner checks are saved.
Passing these selected controls does not certify the production grid.
"""

import argparse
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
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/referee/.mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs/referee/.cache"))

import numpy as np
from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.circular_family_search import CircularFamilySearch, FrequencyBasin, equal_area_sky, select_frequency_basins
from egb_multi.circular_seeds import rank_envelope_frequency_seeds
from egb_multi.circular_spectral import CircularSpectralTDI
from egb_multi.circular_search import circular_source_at_epoch
from egb_multi.band_bounds import expanded_band_maximum_increase_bound
from egb_multi.comparison import AESeries, AEPSD, AEInnerProduct, TDIConvention
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits
from egb_multi.native_validation import ChunkedNativeResponse
from referee_task_2_objective_orientation import assert_local, write_json


def default_plan():
    base = dict(dt_s=10., measurement_order=81, envelope_intervals=8192,
                pm_ode_rtol=1e-10, pm_ode_phase_atol=1e-8,
                band_hz=[2e-5, .04], chirp_bound_scaled=16., width_factor=1.,
                include_primary_laser=False, diagonal_weighting=False,
                full_multibasin_search=False)
    changes = [
        ("baseline_reopt", {}),
        ("envelope_16384", dict(envelope_intervals=16384)),
        ("interpolation_101", dict(measurement_order=101, envelope_intervals=16384)),
        ("cadence_5", dict(dt_s=5., measurement_order=101, envelope_intervals=16384)),
        ("cadence_2_5", dict(dt_s=2.5, measurement_order=101, envelope_intervals=32768)),
        ("tight_PM", dict(pm_ode_rtol=1e-12, pm_ode_phase_atol=1e-10)),
        ("primary_laser", dict(include_primary_laser=True)),
        ("diagonal_AE", dict(diagonal_weighting=True)),
        ("expanded_domain", dict(width_factor=2., chirp_bound_scaled=32., full_multibasin_search=True)),
    ]
    return [dict(base, name=name, **change) if not change else dict(base, **change, name=name)
            for name, change in changes]


def validate_spectral_control(path):
    summary = json.loads((path / "validation_summary.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    if summary["status"] != "passed_selected_annual_spectral_objective":
        raise ValueError("Spectral preparation control did not pass")
    for name in ("spline_fourier.py", "circular_spectral.py", "circular_envelope.py", "circular_fit.py", "comparison.py"):
        relative = "src/egb_multi/" + name
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != manifest["sha256"][relative]:
            raise ValueError("Spectral control source differs: " + name)


def run(args):
    assert_local()
    out, inputs, source_run, control = [p.resolve() for p in
                                       (args.output, args.inputs, args.source_run, args.spectral_validation)]
    if not all(p.is_relative_to(ROOT) for p in (out, inputs, source_run, control)):
        raise ValueError("Keep all calculations and inputs in this checkout")
    validate_spectral_control(control)
    old_file = source_run / f"{args.case}_annual_search.json"
    while not (source_run/'validation_summary.json').exists():
        time.sleep(20)
    baseline_summary=json.loads((source_run/'validation_summary.json').read_text())
    if baseline_summary['status']!='annual_baseline_completed':
        raise ValueError('Completed successful baseline required')
    if args.retain_refinement:
        while not (args.retain_refinement/'validation_summary.json').exists():
            time.sleep(20)
        retained_summary=json.loads((args.retain_refinement/'validation_summary.json').read_text())
        if retained_summary['status']!='selected_annual_reoptimization_completed':
            raise ValueError('Completed successful retained refinement required')
    old = json.loads(old_file.read_text())  # A completed baseline is required.
    independent_reference = None
    reference_file = reference_manifest_file = None
    if args.independent_reference_refinement:
        reference_folder=args.independent_reference_refinement.resolve()
        if not reference_folder.is_relative_to(ROOT):raise ValueError('Use a local independent reference')
        reference_summary=json.loads((reference_folder/'validation_summary.json').read_text())
        if reference_summary['status']!='selected_annual_reoptimization_completed':
            raise ValueError('Completed successful independent comparison reference required')
        reference_file=reference_folder/'annual_refinement.json'
        reference_manifest_file=reference_folder/'manifest.json'
        reference_rows=json.loads(reference_file.read_text())
        if args.independent_reference_level:
            reference_rows=[r for r in reference_rows if r['level']['name']==args.independent_reference_level]
        if len(reference_rows)!=1:raise ValueError('Select exactly one independent comparison reference level')
        independent_reference=reference_rows[0]
        reference_manifest=json.loads(reference_manifest_file.read_text())
        # Native overlaps can only be compared under the same injection and
        # weighting. A change of metric is not an optimizer deficit.
        required_inputs=[inputs/f'{args.case}_source.json',inputs/f'{args.case}_signal.npz',inputs/'annual_covariance.npz']
        if args.laser_covariance:required_inputs.append(args.laser_covariance.resolve())
        for path in required_inputs:
            expected={v for k,v in reference_manifest['inputs'].items() if Path(k).name==path.name}
            actual=hashlib.sha256(path.read_bytes()).hexdigest()
            if expected!={actual}:raise ValueError('Independent reference has different physical input: '+path.name)
    meta_file = inputs / f"{args.case}_source.json"
    meta = json.loads(meta_file.read_text())
    truth = EccentricBinaryParams(**meta["source"])
    duration = meta["duration_s"]
    old_convention = TDIConvention(**meta["convention"])
    plan = json.loads(args.plan.read_text()) if args.plan else default_plan()
    if args.plan is None:
        baseline_config = json.loads((source_run / "manifest.json").read_text())["configuration"]
        baseline_chirp = baseline_config["chirp"]
        for row in plan:
            row["chirp_bound_scaled"] = max(row["chirp_bound_scaled"], baseline_chirp * (2. if row["full_multibasin_search"] else 1.))
    if args.levels:
        names = {row["name"] for row in plan}
        if set(args.levels) - names:
            raise ValueError("Unknown requested refinement level")
        plan = [row for row in plan if row["name"] in args.levels]
    if not plan or len({row["name"] for row in plan}) != len(plan):
        raise ValueError("Nonempty refinement plan with distinct level names required")
    laser_file = args.laser_covariance.resolve() if args.laser_covariance else None
    if any(row["include_primary_laser"] for row in plan) and laser_file is None:
        raise ValueError("Supply physical primary-laser covariance for laser levels")
    if laser_file and not laser_file.is_relative_to(ROOT):
        raise ValueError("Use a local laser covariance")
    out.mkdir(parents=True, exist_ok=False)
    (out / "input_baseline.json").write_bytes(old_file.read_bytes())
    write_json(out / "refinement_plan.json", plan)
    hashes = {}
    for path in [*list((ROOT / "src/egb_multi").glob("*.py")), Path(__file__),
                 ROOT / "benchmarks/referee_task_2_objective_orientation.py"]:
        relative = path.relative_to(ROOT)
        target = out / "snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    input_files = [old_file, meta_file, inputs / f"{args.case}_signal.npz", inputs / "annual_covariance.npz",
                   control / "manifest.json", control / "validation_summary.json"]
    if independent_reference is not None:
        frozen=out/'input_independent_reference_refinement.json'
        shutil.copy2(reference_file,frozen)
        input_files.extend([reference_file,reference_manifest_file,frozen])
    if laser_file:
        input_files.append(laser_file)
    if args.plan:
        input_files.append(args.plan.resolve())
    if args.retain_refinement:
        retained_file = args.retain_refinement.resolve() / "annual_refinement.json"
        if not retained_file.is_relative_to(ROOT):
            raise ValueError("Use local retained refinement results")
        input_files.append(retained_file)
    write_json(out / "manifest.json", dict(created_utc=datetime.now(timezone.utc).isoformat(), sha256=hashes,
        inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files},
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        environment={name: version(name) for name in ("numpy", "scipy", "jax", "pytdi", "lisaorbits")},
        total_absolute_FF_target=.001, waveform_allowance=.0002, optimizer_allowance=.0001,
        scope="Selected annual reoptimization controls; full multibasin status recorded per level"))
    with np.load(inputs / "annual_covariance.npz") as data:
        frequency, covariance = data["frequency_hz"], data["covariance"]
    laser_frequency = laser_covariance = None
    if laser_file:
        with np.load(laser_file) as data:
            laser_frequency, laser_covariance = data["frequency_hz"], data["covariance"]
    candidates = [EccentricBinaryParams(**row["best"]["source"]) for row in old["search"]["basins_results"]]
    candidates.append(EccentricBinaryParams(**old["refinement"]["best"]["source"]))
    if args.retain_refinement:
        prior = json.loads((args.retain_refinement / "annual_refinement.json").read_text())
        candidates.extend(EccentricBinaryParams(**row["search"]["best"]["source"]) for row in prior)
    original_basins = [FrequencyBasin(**row) for row in old["selection"]["basins"]]
    orbit = default_lisaorbits("esa-trailing")
    records, failures = [], []
    for level in plan:
        name = level["name"]
        started = time.perf_counter()
        dt = level["dt_s"]
        times = np.arange(round(duration / dt)) * dt
        convention = replace(old_convention, measurement_order=level["measurement_order"])
        native = ChunkedNativeResponse(orbit, times, convention, block_samples=65536, cached_blocks=2)
        if dt == meta["dt_s"] and convention == old_convention and level["pm_ode_rtol"] == 1e-10 and level["pm_ode_phase_atol"] == 1e-8:
            with np.load(inputs / f"{args.case}_signal.npz") as data:
                signal = AESeries(times, {ch: data[ch] for ch in ("A", "E")}, convention)
        else:
            signal = native(truth, meta["evolution"], pm_ode_rtol=level["pm_ode_rtol"],
                            pm_ode_phase_atol=level["pm_ode_phase_atol"])
        cov = covariance.copy()
        if level["include_primary_laser"]:
            if laser_frequency[0] > level["band_hz"][0] or laser_frequency[-1] < level["band_hz"][1]:
                raise ValueError("Laser spectrum does not cover the full analysis band")
            for a in range(2):
                for b in range(2):
                    cov[:, a, b] += np.interp(frequency, laser_frequency, laser_covariance[:, a, b])
        provenance = dict(sidedness="one_sided", scaling="density", CSD_convention="S_AE=E[tilde_A*conj(tilde_E)]",
            noise_model="native_secondary_plus_adiabatic_primary" if level["include_primary_laser"] else "native_secondary_diagnostic")
        psd = AEPSD(frequency, {ch: cov[:, i, i].real for i, ch in enumerate(("A", "E"))}, convention,
                    provenance, None if level["diagonal_weighting"] else cov[:, 0, 1])
        metric = AEInnerProduct(signal, psd, tuple(level["band_hz"]))
        response = InterpolatedCircularTDI(orbit, times, convention, envelope_intervals=level["envelope_intervals"])
        maximum_radius = max(max(np.linalg.norm(g[3], axis=-1).max(), np.linalg.norm(g[4], axis=-1).max()) for _, _, g in response.terms)
        maximum_ltt = max(g[1].max() for _, _, g in response.terms)
        if maximum_radius >= 1000. or maximum_ltt >= 20.:
            raise ValueError("Retarded geometry exceeds the declared all-sky guard")
        spectral = CircularSpectralTDI(response, metric)
        basins = [replace(b, half_width_cycles=b.half_width_cycles * level["width_factor"]) for b in original_basins]
        selection = None
        new_indices = []
        if level.get("reselect_frequency_basins", False):
            # Expanded physical bands can contain new harmonics. Preserve the
            # old basins as seeds and select additional peaks in the new metric.
            # Reusing only the old centers would not test this possibility.
            basins, selection = select_frequency_basins(metric, signal,
                half_width_cycles=max(b.half_width_cycles for b in basins),
                minimum_peak_fraction=.005, target_coverage=.999,
                extra_frequencies=[b.center_hz for b in basins])
            new_indices = [i for i,b in enumerate(basins) if b.label == "weighted_response_peak"]
            write_json(out / f"{name}_basin_selection.json", selection)
        log = out / f"{name}_evaluations.jsonl"
        def progress(entry):
            with log.open("a") as stream:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
            if entry["unique_intrinsic_evaluation"] % 40 == 0:
                print(args.case, name, "evaluations", entry["unique_intrinsic_evaluation"],
                      "best", search.result()["best"]["profile"]["overlap"], flush=True)
        independent_control = bool(level.get("independent_starts_only", False))
        current_epoch = float(.5*(times[0]+times[-1]))
        prototype = (replace(truth, eccentricity=0., phi0=0., psi=0., inclination=1.,
                     beta=-.6, lambda_=.8, m1_solar=.5, m2_solar=.4, fdot=0.,
                     t0=current_epoch)
                     if independent_control else replace(circular_source_at_epoch(candidates[-1],current_epoch),
                         eccentricity=0., phi0=0.))
        search = CircularFamilySearch(metric, prototype, spectral, basins=basins,
            chirp_scaled_bounds=(-level["chirp_bound_scaled"], level["chirp_bound_scaled"]),
            evaluation_time_bounds_s=(times[0]-2180., duration+2000.), progress=progress)
        retained = []
        if not independent_control:
            for source in candidates:
                retained.append(search.retain(source))
        retained_winners = {}
        for item in retained:
            index = min(range(len(basins)), key=lambda i: abs(basins[i].center_hz - item["frequency_hz"]))
            if index not in retained_winners or item["profile"]["overlap"] > retained_winners[index]["profile"]["overlap"]:
                retained_winners[index] = item
        ranking = sorted(retained_winners, key=lambda i: retained_winners[i]["profile"]["overlap"], reverse=True)
        if independent_control:
            indices = level.get("basin_indices", list(range(len(basins))))
            if not indices or any(i < 0 or i >= len(basins) for i in indices):
                raise ValueError("Valid selected independent-control basin indices required")
        else:
            indices = list(range(len(basins))) if level["full_multibasin_search"] else sorted(set(ranking[:args.local_basins]) | set(new_indices))
        # On the centered observation, a chirp error delta_q contributes
        # pi*delta_q*(t/T)^2. A <=4 spacing bounds the nearest-seed endpoint
        # phase error by pi/2; a five-node expanded chart cannot guarantee it.
        seed_chirp_points = int(level.get("seed_chirp_points",
            max(5, int(np.ceil(2*level["chirp_bound_scaled"]/4))+1) if independent_control else 5))
        runs = []
        for index in indices:
            trials = []
            if index in retained_winners:
                point = search.coordinates_of(EccentricBinaryParams(**retained_winners[index]["source"]), index)
                trials.append((retained_winners[index]["profile"]["overlap"], point, "retained_basin_winner"))
            if level["full_multibasin_search"] or index in new_indices or independent_control:
                seeds = rank_envelope_frequency_seeds(metric, response, search.prototype, basins[index],
                    sky=equal_area_sky(args.sky_points), chirp_scaled_values=np.linspace(-level["chirp_bound_scaled"], level["chirp_bound_scaled"], seed_chirp_points),
                    seed_samples=2048, window_half_width_cycles=256)
                write_json(out / f"{name}_basin{index}_seed_proposals.json", [dict(row, source=asdict(row["source"])) for row in seeds])
                for seed in seeds[:args.exact_seeds]:
                    point = search.coordinates_of(seed["source"], index)
                    score = search.evaluate(point, index)["profile"]["overlap"]
                    trials.append((score, point, "independent_envelope_proposal"))
            if not trials:
                raise ValueError("No retained or independent start for a requested basin")
            center = max(trials, key=lambda row: row[0])[1]
            # Additional physical mass starts keep both masses active in every
            # local optimization, including weakly identifiable directions.
            for mass in ((0., 0.), (0., 1.), (1., 0.), (.5, .5)):
                point = center.copy(); point[4:] = mass
                score = search.evaluate(point, index)["profile"]["overlap"]
                trials.append((score, point, "additional_physical_mass_start"))
            starts = sorted(trials, key=lambda row: row[0], reverse=True)
            if not independent_control:
                blind=[row for row in starts if row[2]=='independent_envelope_proposal']
                # A retained local maximum must not consume all restarts in
                # a purported independent expanded-domain search.
                starts=blind[:1]+starts
            selected = []
            for _, point, origin in starts:
                if independent_control:
                    def direction(p):
                        return np.array([np.sqrt(1-p[2]**2)*np.cos(p[3]), np.sqrt(1-p[2]**2)*np.sin(p[3]), p[2]])
                    distinct = all(np.linalg.norm(direction(point)-direction(p[0])) > .05 for p in selected)
                else:
                    distinct = all(np.linalg.norm(point[2:4]-p[0][2:4]) > .25 or np.linalg.norm(point[4:]-p[0][4:]) > .25 for p in selected)
                if not selected or distinct:
                    selected.append((point, origin))
                if len(selected) == args.starts:
                    break
            for point, origin in selected:
                fit, endpoint, conditioning = search._local_fit(point, index, ftol=args.ftol,
                    maxiter=args.maxiter, finite_difference_step=args.fd_step)
                item = search.evaluate(endpoint, index)
                runs.append(dict(basin_index=index, start=point.tolist(), start_origin=origin,
                    coordinates=endpoint.tolist(), overlap=item["profile"]["overlap"], success=bool(fit.success),
                    message=str(fit.message), iterations=int(fit.nit), function_calls=int(fit.nfev), conditioning=conditioning))
                write_json(out / f"{name}_local_runs.json", runs)
        result = search.result()
        recovered = EccentricBinaryParams(**result["best"]["source"])
        explicit_candidate = native(recovered)
        explicit = metric.compare(explicit_candidate)
        # Use the same broad metric for both parts of each orthogonal split.
        # These are measured signal/template powers, not a global FF bound.
        prepared_candidate = metric.prepare(explicit_candidate)
        power_splits = []
        for split in ((2e-5, .04), (1e-5, .04), (1e-5, .06)):
            if split[0] < metric.band_hz[0] or split[1] > metric.band_hz[1]:
                continue
            inside = (metric.frequency_hz >= split[0]) & (metric.frequency_hz <= split[1])
            reference_inside = float(np.sum(abs(metric.reference[:, inside])**2))
            candidate_inside = float(np.sum(abs(prepared_candidate[:, inside])**2))
            cross_inside = float(np.real(np.vdot(prepared_candidate[:, inside], metric.reference[:, inside])))
            signal_outside = float(np.sum(abs(metric.reference[:, ~inside])**2)/metric.reference_norm)
            template_outside = float(np.sum(abs(prepared_candidate[:, ~inside])**2)/explicit["candidate_norm"])
            inner_candidate_overlap = cross_inside/np.sqrt(reference_inside*candidate_inside)
            power_splits.append(dict(inner_band_hz=list(split), outer_band_hz=list(metric.band_hz),
                signal_outside_fraction=signal_outside, template_outside_fraction=template_outside,
                same_physical_candidate_inner_overlap=inner_candidate_overlap,
                true_same_family_maximum_increase_bound=expanded_band_maximum_increase_bound(max(0.,inner_candidate_overlap),signal_outside),
                scope="Increase bound compares true maxima over one unchanged family; not an absolute FF upper bound, decrease bound or guarantee for a wider template search"))
        del explicit_candidate, prepared_candidate
        difference = explicit["overlap"] - result["best"]["profile"]["overlap"]
        independent_baseline_deficit = None
        if independent_control:
            reference_overlap=old['native_winner_metrics']['overlap']
            if independent_reference is not None:
                for key in ('dt_s','measurement_order','envelope_intervals','pm_ode_rtol','pm_ode_phase_atol',
                            'band_hz','include_primary_laser','diagonal_weighting'):
                    if level[key]!=independent_reference['level'][key]:
                        raise ValueError('Independent reference metric/numerical configuration differs: '+key)
                reference_overlap=independent_reference['native_winner_metrics']['overlap']
            independent_baseline_deficit = reference_overlap - explicit["overlap"]
            if independent_baseline_deficit > 1e-4:
                failures.append(name + ": independent starts miss the archived winning branch by more than the optimizer allowance")
            elif independent_baseline_deficit < -1e-4:
                failures.append(name + ": independent starts improve the archived winner beyond the optimizer allowance; baseline refinements need repeating")
        if abs(difference) > 2e-4:
            failures.append(name + ": native/spectral winner discrepancy")
        winning_index = min(range(len(basins)), key=lambda i: abs(basins[i].center_hz-result["best"]["frequency_hz"]))
        point = search.coordinates_of(recovered, winning_index)
        boundary = min(basins[winning_index].half_width_cycles-abs(point[0]), level["chirp_bound_scaled"]-abs(point[1]))
        if boundary < .01:
            failures.append(name + ": winning frequency/chirp boundary is active")
        row = dict(level=level, search=result, local_runs=runs, native_winner_metrics=explicit,
            native_minus_spectral=difference, winning_basin=winning_index, frequency_chirp_boundary_distance_scaled=float(boundary),
            full_multibasin_search=level["full_multibasin_search"], reoptimized_basins=list(indices),
            reselected_basins=selection, newly_selected_basins=new_indices,
            band_power_splits=power_splits,
            retained_candidate_count=len(retained), all_six_intrinsic_coordinates_free=True,
            independent_starts_only=independent_control,
            independent_baseline_deficit=independent_baseline_deficit,
            independent_reference=(dict(path=str(reference_file.relative_to(ROOT)),
                level=independent_reference['level']['name'],native_overlap=reference_overlap,
                same_physical_metric_inputs_checked=True) if independent_control and independent_reference is not None else None),
            effective_seed_chirp_points=seed_chirp_points,
            physical_epochs_s=dict(injection=truth.t0, template=recovered.t0), wall_s=time.perf_counter()-started)
        records.append(row)
        discovered_winners = {}
        for item in search.cache.values():
            index = min(range(len(basins)), key=lambda i: abs(basins[i].center_hz-item["frequency_hz"]))
            if index not in discovered_winners or item["profile"]["overlap"] > discovered_winners[index]["profile"]["overlap"]:
                discovered_winners[index] = item
        candidates.extend(EccentricBinaryParams(**item["source"]) for item in discovered_winners.values())
        # Keep the union compact without rounding any physical coordinate.
        candidates = list({tuple(asdict(source).values()): source for source in candidates}.values())
        write_json(out / "annual_refinement.json", records)
        print(args.case, name, "native overlap", explicit["overlap"], "elapsed_s", row["wall_s"], flush=True)
        del native, response, spectral, metric, signal, search
        gc.collect()
    write_json(out / "validation_summary.json", dict(status="selected_annual_reoptimization_completed" if not failures else "requires_investigation",
        failures=failures, case=args.case, levels_completed=len(records), levels_requested=len(plan), production_certified=False,
        limitations=["Local levels reoptimize declared retained basins; independent full multibasin levels recorded separately",
                     "Spectral control at other cadences requires native winner checks, which are included",
                     "No automatic uniform FF error bound or contour certification inferred from selected differences"]))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inputs", type=Path, default=ROOT/"outputs/referee/task_2_8_20261008_annual_weighting_pilot")
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--case", default="near_contour")
    p.add_argument("--spectral-validation", type=Path, required=True)
    p.add_argument("--laser-covariance", type=Path)
    p.add_argument("--plan", type=Path)
    p.add_argument("--retain-refinement", type=Path)
    p.add_argument('--independent-reference-refinement',type=Path,
        help='Completed refinement in the same physical metric for an independent-start comparison')
    p.add_argument('--independent-reference-level',
        help='Select one reference level when the reference refinement contains several levels')
    p.add_argument("--levels", nargs="+")
    p.add_argument("--local-basins", type=int, default=2)
    p.add_argument("--starts", type=int, default=2)
    p.add_argument("--sky-points", type=int, default=48)
    p.add_argument("--exact-seeds", type=int, default=12)
    p.add_argument("--maxiter", type=int, default=30)
    p.add_argument("--ftol", type=float, default=1e-9)
    p.add_argument("--fd-step", type=float, default=1e-6)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        if args.output.exists() and not (args.output/"validation_summary.json").exists():
            write_json(args.output/"validation_summary.json", dict(status="failed_refinement_run", production_certified=False,
                exception_type=type(exc).__name__, exception=str(exc)))
        raise
