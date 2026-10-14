"""Resumable Task 2.9 campaign using the frozen local eGB production reference.

One memory-bounded worker runs at a time. The first two original-grid points
are resource/workflow pilots, followed by the remaining 2160-point base grid
and median FF=0.9 brackets. Native sampled signals and all search evidence are
retained. This launcher never declares scientific completion from exit codes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/referee"
FROZEN = BASE / "task_2_8_20261012_completion/frozen_production_configuration.json"
CONTROL = BASE / "task_2_8_20261008_spectral_objective_provenance_control"
sys.path.insert(0, str(ROOT / "benchmarks"))


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def environment():
    return dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (ROOT / "src", ROOT / "benchmarks", BASE / ".precision_deps"))),
                MPLCONFIGDIR=str(BASE / ".mplconfig"), XDG_CACHE_HOME=str(BASE / ".cache"),
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")


def case_id(point):
    return "p_" + hashlib.sha256(json.dumps(point, sort_keys=True).encode()).hexdigest()[:16]


def original_grid():
    axes = dict(f0_hz=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3],
                eccentricity=[0., .001, .003, .01, .03, .05, .1, .2, .3, .5],
                inclination=[.2, .8, 1.2], sky=[[.2, -1.], [.6, .5], [-.5, 1.8], [0., 2.8]],
                masses=[[.5, .5], [.8, .6], [1., .3]])
    points = [dict(f0_hz=f, eccentricity=e, inclination=i, beta=sky[0], lambda_=sky[1],
                   m1_solar=m[0], m2_solar=m[1])
              for f, e, i, sky, m in itertools.product(*axes.values())]
    assert len(points) == 2160 and len({case_id(p) for p in points}) == 2160
    # Pilot points belong to the original grid; they are not extra diagnostics.
    def priority(p):
        fiducial = (p["f0_hz"] == .001 and p["inclination"] == .8 and p["beta"] == .2
                    and p["lambda_"] == -1. and p["m1_solar"] == .5 and p["m2_solar"] == .5)
        return (0 if fiducial and p["eccentricity"] == 0 else
                1 if fiducial and p["eccentricity"] == .1 else 2,
                p["f0_hz"], p["m1_solar"], p["m2_solar"], p["inclination"], p["beta"], p["eccentricity"])
    points.sort(key=priority)
    return axes, [dict(id=case_id(p), point=p, stage="base_grid", pilot=i < 2) for i, p in enumerate(points)]


def verify_inputs(out):
    manifest = read(out / "manifest.json")
    for relative, expected in manifest["input_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError("Production input changed: " + relative)
    if digest(out / "shared/annual_covariance.npz") != manifest["combined_covariance_sha256"]:
        raise RuntimeError("Combined covariance changed")
    return manifest


def prepare(out):
    if not out.is_relative_to(ROOT):
        raise ValueError("Keep the campaign in the authorized eGB workspace")
    if (out / "manifest.json").exists():
        verify_inputs(out)
        return
    from referee_task_2_8_completion import reviewed_completion
    decision = reviewed_completion()
    if not decision or not decision["accepted"]:
        raise RuntimeError("Task 2.8 reviewed frozen configuration is not intact")
    cfg = read(FROZEN)
    if not (cfg["waveform_dt_s"] == 2.5 and cfg["circular_envelope_intervals"] == 32768
            and cfg["analysis_band_hz"] == [1e-5, .08] and cfg["template_family"]["both_masses_remain_free"]):
        raise RuntimeError("Unexpected frozen production contract")
    out.mkdir(parents=True, exist_ok=False)
    shared = out / "shared"
    shared.mkdir()
    import numpy as np
    secondary = ROOT / cfg["reference"]["covariance_file"]
    primary = ROOT / cfg["reference"]["laser_covariance_file"]
    if digest(secondary) != cfg["reference"]["covariance_file_sha256"] or digest(primary) != cfg["reference"]["laser_covariance_file_sha256"]:
        raise RuntimeError("Frozen physical noise covariance changed")
    with np.load(secondary) as data:
        frequency, covariance = data["frequency_hz"], data["covariance"].copy()
    with np.load(primary) as data:
        lf, lc = data["frequency_hz"], data["covariance"]
    if lf[0] > cfg["analysis_band_hz"][0] or lf[-1] < cfg["analysis_band_hz"][1]:
        raise RuntimeError("Primary laser grid does not cover the analysis band")
    for a in range(2):
        for b in range(2):
            covariance[:, a, b] += np.interp(frequency, lf, lc[:, a, b])
    mask=(frequency>=cfg["analysis_band_hz"][0]) & (frequency<=cfg["analysis_band_hz"][1])
    current=covariance[mask]
    scale=np.maximum(current[:,0,0].real,current[:,1,1].real)
    hermitian_error=float(np.max(np.abs(current-current.conj().transpose(0,2,1))/scale[:,None,None]))
    coherence=np.abs(current[:,0,1])**2/(current[:,0,0].real*current[:,1,1].real)
    if (not np.all(np.isfinite(current)) or np.any(scale<=0)
        or np.any(current[:,0,0].real<=0) or np.any(current[:,1,1].real<=0)
        or hermitian_error>1e-10 or np.any(coherence>=1)):
        raise RuntimeError("Combined production covariance is not finite Hermitian positive definite")
    save(shared/"covariance_contract_check.json",dict(hermitian_relative_error=hermitian_error,
        maximum_AE_coherence=float(coherence.max()),band_hz=cfg["analysis_band_hz"],
        secondary_file=str(secondary.relative_to(ROOT)),primary_file=str(primary.relative_to(ROOT)),
        assembly="Exact same secondary-plus-interpolated-primary sum as the final Task 2.8 metric"))
    np.savez_compressed(shared / "annual_covariance.npz", frequency_hz=frequency, covariance=covariance)
    axes, queue = original_grid()
    save(out / "queue.json", queue)
    save(out / "grid_definition.json", dict(axes=axes, base_points=len(queue), source_psi=.4, source_phi0=.2,
        source_epoch_s=0., injection_evolution="peters_mathews", contour_target_ff=.9,
        contour_bracket_width=.001, maximum_contour_bisections=10,
        aggregation="Median of individual FF over 12 sky/inclination points per mass, and over all 36 for frequency-only curves",
        waveform_storage="Retain generated inputs and native search evidence; stop if less than 25 GiB disk remains"))
    input_paths = [FROZEN, BASE / "task_2_8_20261012_completion/completion_decision.json",
        CONTROL / "manifest.json", CONTROL / "validation_summary.json", secondary, primary,
        *sorted((ROOT / "src/egb_multi").glob("*.py")), Path(__file__),
        ROOT / "benchmarks/referee_task_2_annual_search.py",
        ROOT / "benchmarks/referee_task_2_annual_refinement.py",
        ROOT / "benchmarks/referee_task_2_objective_orientation.py"]
    for path in input_paths:
        relative = path.relative_to(ROOT)
        if path.suffix == ".py" or path == FROZEN:
            target = out / "snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    save(out / "manifest.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in input_paths},
        combined_covariance_sha256=digest(shared / "annual_covariance.npz"),
        reviewed_completion=decision, frozen_configuration=cfg,
        worker_count=1, machine_memory_bytes=16 * 2**30, base_grid_points=2160,
        source_root=str(ROOT / "src/egb_multi"), git_commit=None,
        search=dict(sky_points=96, chirp_points=25, exact_seeds=32, starts_per_basin=4,
                    maxiter=90, seed_samples=2048, seed_window_cycles=256),
        status="prepared_not_yet_launched",
        scope="Full science-grid calculation under the representative validated idealized reference; no universal global-search or mission-likelihood certificate"))
    save(out / "work_state.json", dict(status="prepared", completed_points=0, base_points=2160,
        production_grid_complete=False, task_2_9_complete=False))


def width_for(point):
    if point["f0_hz"] >= .002 and point["eccentricity"] >= .3:
        return 192.
    return 96. if point["eccentricity"] >= .2 else 24.


def worker(out, entry):
    manifest = verify_inputs(out)
    cfg, point = manifest["frozen_configuration"], entry["point"]
    folder = out / "points" / entry["id"]
    folder.mkdir(parents=True, exist_ok=True)
    inputs = folder / "inputs"
    inputs.mkdir(exist_ok=True)
    import numpy as np
    from dataclasses import asdict, replace
    import gc
    from importlib.metadata import version
    from egb_multi.comparison import TDIConvention, AEPSD
    from egb_multi.eccentric import EccentricBinaryParams, peters_mathews_derivatives
    from egb_multi.external import default_lisaorbits
    from egb_multi.native_validation import ChunkedNativeResponse
    from egb_multi.circular_search import circular_source_at_epoch
    import referee_task_2_annual_search as annual
    annual.assert_local()
    save(folder / "runtime.json", dict(python=sys.version, executable=sys.executable,
        distributions={name:version(name) for name in ("numpy", "scipy", "jax", "pytdi", "lisaorbits")},
        source_root=str(ROOT / "src/egb_multi")))
    duration, dt = cfg["duration_s"], cfg["waveform_dt_s"]
    convention = TDIConvention(**cfg["tdi_convention"])
    source = EccentricBinaryParams(mean_motion=np.pi * point["f0_hz"], eccentricity=point["eccentricity"],
        m1_solar=point["m1_solar"], m2_solar=point["m2_solar"], inclination=point["inclination"],
        beta=point["beta"], lambda_=point["lambda_"], psi=.4, phi0=.2, fdot=0., t0=0.)
    series_file = inputs / "production_signal.npz"
    meta_file = inputs / "production_source.json"
    if not series_file.exists():
        started = time.monotonic()
        times = np.arange(round(duration / dt)) * dt
        if len(times) * dt != duration:
            raise RuntimeError("Observation is not exactly preserved")
        print("Generating annual native PM signal", entry["id"], len(times), "samples", flush=True)
        native = ChunkedNativeResponse(default_lisaorbits("esa-trailing"), times, convention,
                                       block_samples=65536, cached_blocks=2)
        signal = native(source, "peters_mathews", pm_ode_rtol=cfg["pm_ode_rtol"],
                        pm_ode_phase_atol=cfg["pm_ode_phase_atol"])
        temp = inputs / "production_signal.tmp.npz"
        np.savez_compressed(temp, A=signal.channels["A"], E=signal.channels["E"])
        temp.replace(series_file)
        save(meta_file, dict(source=asdict(source), evolution="peters_mathews", convention=asdict(convention),
            duration_s=duration, dt_s=dt, signal_sha256=digest(series_file),
            signal_array_sha256={ch:hashlib.sha256(signal.channels[ch].tobytes()).hexdigest() for ch in ("A", "E")},
            pm_ode_rtol=cfg["pm_ode_rtol"], pm_ode_phase_atol=cfg["pm_ode_phase_atol"],
            injection_wall_s=time.monotonic()-started))
        del signal, native, times
        gc.collect()
        print("Native injection complete", round(time.monotonic()-started, 1), "s", flush=True)
    elif not meta_file.exists() or digest(series_file) != read(meta_file)["signal_sha256"]:
        raise RuntimeError("Incomplete or changed injection checkpoint")
    covariance_link = inputs / "annual_covariance.npz"
    if not covariance_link.exists():
        covariance_link.symlink_to(out / "shared/annual_covariance.npz")
    circular = replace(source, eccentricity=0.)
    fdot = float(peters_mathews_derivatives(circular.mean_motion, 0., circular.total_mass_kg,
                                          circular.symmetric_mass_ratio)[0] / np.pi)
    midpoint = .5 * (duration - dt)
    candidates = [circular_source_at_epoch(replace(circular, fdot=factor * fdot), midpoint)
                  for factor in (0., .25, .5, 1., 2., 4., 8., 16., 32.)]
    save(inputs / "annual_weighting_comparison.json", [dict(case="production", source=asdict(source),
        candidates=[dict(source=asdict(c), origin="legacy_circular_derivative_candidate_in_current_metric") for c in candidates])])
    # Reuse the validated search implementation. These two adapters supply
    # frozen-reference metadata and add the explicitly retained carriers using
    # the existing validated basin selector; no waveform/optimizer is replaced.
    original_selector = annual.select_frequency_basins
    annual.select_frequency_basins = lambda *args, **kwargs: original_selector(*args, **kwargs,
        extra_frequencies=[c.mean_motion/np.pi for c in candidates])
    def reference_psd(frequency, channels, convention, provenance, cross):
        provenance = dict(provenance, noise_model="frozen_annual_moving_secondary_plus_primary_ideal_reference",
                          frozen_configuration=str(FROZEN.relative_to(ROOT)), production_certified=False)
        return AEPSD(frequency, channels, convention, provenance, cross)
    annual.AEPSD = reference_psd
    attempts = sorted(folder.glob("attempt_*"))
    complete = [p for p in attempts if (p / "production_annual_search.json").exists()
                and (p / "validation_summary.json").exists() and read(p / "validation_summary.json")["status"] == "annual_baseline_completed"]
    if complete:
        search_out = complete[-1]
    else:
        resume = next((p for p in reversed(attempts) if (p / "production_search_baseline.json").exists()), None)
        search_out = folder / ("attempt_%03d" % len(attempts))
        settings = dict(output=search_out, inputs=inputs, cases=["production"], band=cfg["analysis_band_hz"],
            envelope_intervals=cfg["circular_envelope_intervals"], width=width_for(point), chirp=48.,
            chirp_points=25, sky_points=96, starts=4, maxiter=90, ftol=1e-11, fd_step=5e-6,
            peak_fraction=.005, coverage=.999, retain_run=None, seed_method="envelope", seed_samples=2048,
            seed_window=256, exact_seeds=32, spectral_objective=True, spectral_validation=CONTROL,
            retain_evaluations=None, resume_run=resume, independent_control=False, retain_candidates=None,
            injection_sky_seeds=True)
        annual.run(SimpleNamespace(**settings))
    result = read(search_out / "production_annual_search.json")
    best = result["refinement"]["best"]
    ff = result["native_winner_metrics"]["overlap"]
    basins = result["search"]["basins_results"]
    winner = min(basins, key=lambda b:abs(b["basin"]["center_hz"] - best["frequency_hz"]))
    x = winner["best_coordinates"]
    final_source = EccentricBinaryParams(**best["source"])
    # Recompute boundary distance at the FINAL candidate, not the baseline.
    frequency_distance = winner["basin"]["half_width_cycles"] - abs((final_source.mean_motion/np.pi-winner["basin"]["center_hz"])*duration)
    chirp_distance = 48. - abs(final_source.fdot*duration**2)
    checks = dict(native_spectral_agreement=abs(result["native_minus_envelope"]) <= .0002,
        inactive_frequency_chirp_boundaries=min(frequency_distance, chirp_distance) > .01,
        final_refinement_success=result["refinement"]["refinement"]["success"],
        final_refinement_shift=abs(best["profile"]["overlap"]-result["search"]["best"]["profile"]["overlap"]) <= .0001,
        retains_legacy_candidates=len(result["retained_pool_candidates"]) == 9,
        no_legacy_candidate_exceeds_winner=all(r["profile"]["overlap"] <= ff+.0002 for r in result["retained_pool_candidates"]),
        all_basins_completed=len(basins) == result["search"]["requested_basins"],
        blind_restart_in_every_basin=all(b["blind_restart_reserved"] and any(r["start_origin"] == "independent_envelope_seed"
            for r in b["independent_start_results"]) for b in basins),
        coherent_full_AE_weighting=result["search"]["noise_weighting"] == "full_Hermitian_AE")
    # New winning harmonic branches and contour points receive a separate
    # six-start blind check using the existing annual refinement implementation.
    if abs(ff-.9) < .015 or best["frequency_hz"] > 1.25*point["f0_hz"]:
        import referee_task_2_annual_refinement as refinement
        # Same adapter: the cache already contains both noise contributions.
        refinement.AEPSD = reference_psd
        plan = [dict(name="production_blind_winning_basin", dt_s=dt, measurement_order=101,
            envelope_intervals=32768, pm_ode_rtol=1e-12, pm_ode_phase_atol=1e-10,
            band_hz=[1e-5,.08], chirp_bound_scaled=48., width_factor=1.,
            include_primary_laser=False, diagonal_weighting=False, full_multibasin_search=False,
            independent_starts_only=True, basin_indices=[winner["basin_index"]], seed_chirp_points=25)]
        plan_file = folder / "blind_plan.json"
        save(plan_file, plan)
        blind_out = folder / "blind_check"
        if not (blind_out / "validation_summary.json").exists():
            refinement.run(SimpleNamespace(output=blind_out, inputs=inputs, source_run=search_out,
                spectral_validation=CONTROL, case="production", retain_refinement=None,
                independent_reference_refinement=None, independent_reference_level=None,
                plan=plan_file, levels=None, laser_covariance=None, local_basins=1, starts=6,
                sky_points=96, exact_seeds=32, maxiter=90, ftol=1e-12, fd_step=1e-6))
        blind = read(blind_out / "annual_refinement.json")[0]
        checks["independent_winning_branch_agreement"] = abs(blind["native_winner_metrics"]["overlap"]-ff) <= .0001
    save(folder / "result.json", dict(id=entry["id"], point=point, stage=entry["stage"], fitting_factor=ff,
        mismatch=1-ff, native_minus_spectral=result["native_minus_envelope"],
        checks=checks, status="passed_point_controls" if all(checks.values()) else "requires_investigation",
        winning_source=best["source"], winning_basin=winner["basin_index"],
        frequency_chirp_boundary_distance_scaled=min(frequency_distance, chirp_distance),
        search_output=str(search_out.relative_to(ROOT)), wall_s=result["wall_s"],
        native_winner_metrics=result["native_winner_metrics"],
        scope="Empirical finite search under frozen idealized reference; no global-maximum proof"))
    if not all(checks.values()):
        raise RuntimeError("Production point requires investigation: " + str(checks))
    print("Production point passed", entry["id"], "FF", ff, flush=True)


def completed_results(out, queue):
    rows = []
    for entry in queue:
        path = out / "points" / entry["id"] / "result.json"
        if path.exists():
            row = read(path)
            if row["status"] == "passed_point_controls":
                rows.append(row)
    return rows


def contour_next(out, queue, rows):
    """Bracket medians at actual shared eccentricities; never average crossings."""
    axes = read(out / "grid_definition.json")["axes"]
    index = {r["id"]:r for r in rows}
    brackets, pending = [], []
    for frequency in axes["f0_hz"]:
        for mass_group in [None] + axes["masses"]:
            masses = axes["masses"] if mass_group is None else [mass_group]
            relevant = [e for e in queue if e["point"]["f0_hz"] == frequency
                and [e["point"]["m1_solar"],e["point"]["m2_solar"]] in masses]
            by_e = {}
            for entry in relevant:
                by_e.setdefault(entry["point"]["eccentricity"], {})[entry["id"]] = entry
            medians = []
            for eccentricity, entries in by_e.items():
                if len(entries) == 12*len(masses) and all(k in index for k in entries):
                    medians.append((eccentricity,statistics.median(index[k]["fitting_factor"] for k in entries)))
            medians.sort()
            crossings = [(a,b) for a,b in zip(medians,medians[1:]) if (a[1]-.9)*(b[1]-.9) <= 0]
            if not crossings:
                brackets.append(dict(f0_hz=frequency,masses=mass_group,status="no_crossing_in_computed_range",medians=medians))
            for lower,upper in crossings:
                width = upper[0]-lower[0]
                brackets.append(dict(f0_hz=frequency,masses=mass_group,eccentricity_bracket=[lower[0],upper[0]],
                    endpoint_FF=[lower[1],upper[1]],status="bracketed" if width<=.001 else "needs_refinement"))
                if width > .001:
                    middle = .5*(lower[0]+upper[0])
                    for mass, inclination, sky in itertools.product(masses,axes["inclination"],axes["sky"]):
                        p=dict(f0_hz=frequency,eccentricity=middle,inclination=inclination,beta=sky[0],lambda_=sky[1],m1_solar=mass[0],m2_solar=mass[1])
                        ident=case_id(p)
                        if ident not in {e["id"] for e in queue}:
                            pending.append(dict(id=ident,point=p,stage="contour_refinement",pilot=False))
    save(out / "contour_brackets.json", brackets)
    return list({e["id"]:e for e in pending}.values())


def campaign(out):
    manifest = verify_inputs(out)
    lock = out / "controller.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    except FileExistsError:
        old_pid=int(lock.read_text())
        try:os.kill(old_pid,0)
        except ProcessLookupError:lock.unlink();return campaign(out)
        raise RuntimeError("A production controller is already running")
    try:
        queue=read(out / "queue.json")
        rounds=0
        while True:
            for entry in queue:
                result_file=out / "points" / entry["id"] / "result.json"
                if result_file.exists() and read(result_file)["status"] == "passed_point_controls":
                    continue
                verify_inputs(out)
                if shutil.disk_usage(out).free < 25*2**30:
                    raise RuntimeError("Less than 25 GiB disk remains; retained-waveform storage requires review")
                folder=out / "points" / entry["id"]
                folder.mkdir(parents=True,exist_ok=True)
                save(folder / "entry.json",entry)
                command=[sys.executable,str(Path(__file__).resolve()),"--output",str(out),"--worker",str(folder/"entry.json")]
                log=folder/"run.log"
                with log.open("a") as stream:
                    child=subprocess.Popen(command,cwd=ROOT,env=environment(),stdout=stream,stderr=subprocess.STDOUT)
                    save(out / "work_state.json",dict(status="running_resource_pilot" if entry.get("pilot") else "running_production_grid",
                        controller_pid=os.getpid(),worker_pid=child.pid,current_case=entry,log=str(log.relative_to(ROOT)),
                        completed_points=len(completed_results(out,queue)),queued_points=len(queue),base_points=2160,
                        production_grid_complete=False,task_2_9_complete=False,updated_utc=datetime.now(timezone.utc).isoformat()))
                    print("Running",entry["id"],entry["stage"],"worker",child.pid,flush=True)
                    code=child.wait()
                if code:
                    raise RuntimeError("Worker failed; inspect " + str(log.relative_to(ROOT)))
                rows=completed_results(out,queue)
                save(out / "results.json",rows)
                elapsed=sum(r["wall_s"] for r in rows)
                save(out / "resource_pilot.json",dict(completed_points=len(rows),mean_search_wall_s=elapsed/len(rows),
                    linear_base_grid_search_estimate_days=(elapsed/len(rows))*2160/86400,
                    caveat="Early pilot excludes injection/control costs and harder harmonic searches; not a completion-time promise"))
            rows=completed_results(out,queue)
            extra=contour_next(out,queue,rows)
            if not extra:break
            rounds+=1
            if rounds>10:raise RuntimeError("Contour refinement exceeded ten rounds; review crossings")
            queue.extend(extra);save(out / "queue.json",queue)
            print("Added",len(extra),"contour points",flush=True)
        save(out / "work_state.json",dict(status="grid_and_brackets_computed_pending_review_and_figures",
            completed_points=len(rows),queued_points=len(queue),base_points=2160,
            production_grid_complete=True,task_2_9_complete=False,
            remaining="Review per-point evidence/contour precision and regenerate figures/table; manuscript work deferred"))
    except BaseException as error:
        state=read(out / "work_state.json")
        save(out / "work_state.json",dict(state,status="requires_investigation",exception_type=type(error).__name__,
            exception=str(error),production_grid_complete=False,task_2_9_complete=False))
        raise
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=BASE/"task_2_9_20261012_production")
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--launch",action="store_true")
    parser.add_argument("--worker",type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if args.worker:
        worker(out,read(args.worker))
    else:
        prepare(out)
        if args.launch:
            lock=out/"controller.lock"
            if lock.exists():
                try:os.kill(int(lock.read_text()),0)
                except ProcessLookupError:pass
                else:raise RuntimeError("Production is already running; duplicate launch refused")
            with (out/"controller.log").open("a") as stream:
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--output",str(out)],
                    cwd=ROOT,env=environment(),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            wake=subprocess.Popen(["/usr/bin/caffeinate","-i","-w",str(child.pid)],stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            save(out/"launch.json",dict(controller_pid=child.pid,caffeinate_pid=wake.pid,
                launched_utc=datetime.now(timezone.utc).isoformat(),output=str(out),worker_count=1,
                launch_authorized="User requested launch now; supersedes earlier target start time"))
            print(json.dumps(read(out/"launch.json"),indent=2))
        elif args.prepare:
            print(json.dumps(dict(status="prepared",base_points=2160,output=str(out)),indent=2))
        else:
            campaign(out)
