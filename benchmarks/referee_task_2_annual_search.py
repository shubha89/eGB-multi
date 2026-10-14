"""Annual circular-family optimization with reusable native clean injections.

One fixed full A/E PSD, observation, band and source epoch define each run.
Both component masses and the entire sky remain free. Pool candidates are
retained but do not count as independent starts. This baseline is one gate of
2.8, not by itself a production or contour-convergence certificate.
"""

import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/referee/.mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/"outputs/referee/.cache"))

import numpy as np
from egb_multi.comparison import AESeries,AEPSD,AEInnerProduct,TDIConvention
from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.circular_family_search import CircularFamilySearch,select_frequency_basins,equal_area_sky
from egb_multi.circular_seeds import rank_envelope_frequency_seeds
from egb_multi.circular_spectral import CircularSpectralTDI
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits
from egb_multi.native_validation import ChunkedNativeResponse
from referee_task_2_objective_orientation import assert_local,write_json


def run(args):
    assert_local();out=args.output.resolve();inputs=args.inputs.resolve()
    if not out.is_relative_to(ROOT) or not inputs.is_relative_to(ROOT):raise ValueError("Use this checkout")
    resume_manifest=None
    if args.resume_run:
        args.resume_run=args.resume_run.resolve()
        if not args.resume_run.is_relative_to(ROOT) or args.seed_method!="envelope" or len(args.cases)!=1:
            raise ValueError("Resume a single local envelope-search checkpoint")
        if args.retain_evaluations:raise ValueError("Resume supplies its own immutable retained-log prefix")
        resume_manifest=json.loads((args.resume_run/"manifest.json").read_text())
        old_config=resume_manifest["configuration"]
        if bool(old_config.get("independent_control",False))!=args.independent_control:
            raise ValueError("Resume independent-control setting differs")
        for key in ("cases","band","envelope_intervals","width","chirp","chirp_points","sky_points","starts",
                    "maxiter","ftol","fd_step","peak_fraction","coverage","seed_method","seed_samples",
                    "seed_window","exact_seeds","spectral_objective"):
            if old_config[key]!=getattr(args,key):raise ValueError("Resume search configuration differs: "+key)
        for relative,expected in resume_manifest["sha256"].items():
            if relative.startswith("src/egb_multi/") and relative!="src/egb_multi/circular_family_search.py":
                if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
                    raise ValueError("Resume waveform or metric source differs: "+relative)
        for relative,expected in resume_manifest["inputs"].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
                raise ValueError("Previous run input changed: "+relative)
    if args.spectral_objective:
        if args.spectral_validation is None:raise ValueError("Validate the spectral objective before using it")
        summary=json.loads((args.spectral_validation/"validation_summary.json").read_text())
        manifest=json.loads((args.spectral_validation/"manifest.json").read_text())
        if summary["status"]!="passed_selected_annual_spectral_objective":raise ValueError("Spectral validation did not pass")
        for name in ("spline_fourier.py","circular_spectral.py","circular_envelope.py","circular_fit.py","comparison.py"):
            relative="src/egb_multi/"+name
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=manifest["sha256"][relative]:
                raise ValueError("Spectral validation source differs: "+name)
    out.mkdir(parents=True,exist_ok=False);hashes={}
    for path in [*list((ROOT/"src/egb_multi").glob("*.py")),Path(__file__),ROOT/"benchmarks/referee_task_2_objective_orientation.py"]:
        rel=path.relative_to(ROOT);target=out/"snapshot"/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    input_files=[inputs/"annual_covariance.npz",inputs/"annual_weighting_comparison.json"]
    for case in args.cases:input_files += [inputs/f"{case}_source.json",inputs/f"{case}_signal.npz"]
    if args.retain_run:
        for case in args.cases:
            path=args.retain_run.resolve()/f"{case}_annual_search.json"
            if not path.is_relative_to(ROOT):raise ValueError("Use local retained runs")
            frozen=out/f"input_retained_{case}_annual_search.json"
            shutil.copy2(path,frozen);input_files.extend([path,frozen])
    if args.retain_candidates:
        path=args.retain_candidates.resolve()
        if not path.is_relative_to(ROOT):raise ValueError("Use local retained candidates")
        shutil.copy2(path,out/"input_retained_candidates.json")
        input_files.extend([path,out/"input_retained_candidates.json"])
    if args.spectral_validation:input_files += [args.spectral_validation.resolve()/"manifest.json",args.spectral_validation.resolve()/"validation_summary.json"]
    if args.retain_evaluations:
        path=args.retain_evaluations.resolve()
        if not path.is_relative_to(ROOT):raise ValueError("Use local retained evaluations")
        frozen=path.read_bytes()
        if not frozen.endswith(b"\n"):frozen=frozen[:frozen.rfind(b"\n")+1]
        (out/"input_retained_evaluations.jsonl").write_bytes(frozen)
        input_files.append(out/"input_retained_evaluations.jsonl")
    if args.resume_run:
        case=args.cases[0]
        for source_name,destination_name in ((f"{case}_search_baseline.json","input_resume_search_baseline.json"),
            (f"{case}_evaluations.jsonl","input_retained_evaluations.jsonl"),
            (f"{case}_basin_selection.json","input_resume_basin_selection.json"),("manifest.json","input_resume_manifest.json")):
            destination=out/destination_name
            shutil.copy2(args.resume_run/source_name,destination);input_files.append(destination)
    write_json(out/"manifest.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),sha256=hashes,
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files},
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scope="Annual baseline intrinsic optimization; separate numerical/domain/noise gates remain"))
    with np.load(inputs/"annual_covariance.npz") as data:
        frequency=data["frequency_hz"];covariance=data["covariance"]
    pool=json.loads((inputs/"annual_weighting_comparison.json").read_text());records=[];failures=[]
    orbit=default_lisaorbits("esa-trailing")
    for case in args.cases:
        started=time.perf_counter();meta=json.loads((inputs/f"{case}_source.json").read_text())
        convention=TDIConvention(**meta["convention"]);duration=meta["duration_s"];dt=meta["dt_s"]
        times=np.arange(round(duration/dt))*dt
        with np.load(inputs/f"{case}_signal.npz") as data:
            signal=AESeries(times,{ch:data[ch] for ch in ("A","E")},convention)
        provenance=dict(sidedness="one_sided",scaling="density",noise_model="native_annual_secondary_diagnostic",
            CSD_convention="S_AE=E[tilde_A*conj(tilde_E)]",production_certified=False)
        psd=AEPSD(frequency,{ch:covariance[:,i,i].real for i,ch in enumerate(("A","E"))},convention,provenance,covariance[:,0,1])
        metric=AEInnerProduct(signal,psd,tuple(args.band))
        original=next(r for r in pool if r["case"]==case)
        prototype=EccentricBinaryParams(mean_motion=original["source"]["mean_motion"],eccentricity=0.,
            t0=.5*(times[0]+times[-1]),beta=-.6,lambda_=.8,m1_solar=.5,m2_solar=.5,
            inclination=.6,phi0=0.,psi=.2)
        response=InterpolatedCircularTDI(orbit,times,convention,envelope_intervals=args.envelope_intervals)
        # Conservative light-time bounds include all skies, nine atomic arms,
        # and >3 times the orbit's ~500 s barycentric radius. Native geometry
        # arrays are checked against the declared radius guard.
        maximum_radius=max(max(np.linalg.norm(g[3],axis=-1).max(),np.linalg.norm(g[4],axis=-1).max()) for _,_,g in response.terms)
        maximum_ltt=max(g[1].max() for _,_,g in response.terms)
        if maximum_radius>=1000 or maximum_ltt>=20:raise ValueError("Geometry exceeds conservative all-sky guard")
        bounds=(times[0]-2000-9*20,times[-1]+2000)
        try:
            basins,selection=select_frequency_basins(metric,signal,half_width_cycles=args.width,
                minimum_peak_fraction=args.peak_fraction,target_coverage=args.coverage)
        except ValueError as exc:
            failures.append(f"{case}: {exc}")
            write_json(out/f"{case}_validation_summary.json",dict(status="failed_basin_band_control",exception=str(exc),production_certified=False))
            continue
        write_json(out/f"{case}_basin_selection.json",selection)
        resumed_stages=[]
        if args.resume_run:
            previous_selection=json.loads((out/"input_resume_basin_selection.json").read_text())
            if selection!=previous_selection:raise ValueError("Resume basin selection or coverage differs")
            prior_checkpoint=json.loads((out/"input_resume_search_baseline.json").read_text())
            resumed_stages=prior_checkpoint["basins_results"]
            if [r["basin_index"] for r in resumed_stages]!=list(range(len(resumed_stages))):
                raise ValueError("Resume checkpoint must contain a contiguous completed basin prefix")
            if prior_checkpoint["requested_basins"]!=len(basins):raise ValueError("Resume requested basin count differs")
        print(case,"selected",len(basins),"annual basins",flush=True)
        log=out/f"{case}_evaluations.jsonl"
        def progress(entry):
            with log.open("a") as stream:stream.write(json.dumps(entry,allow_nan=False)+"\n")
            if entry["unique_intrinsic_evaluation"]%20==0:
                best=max(r["overlap"] for r in search.history)
                print(case,entry["unique_intrinsic_evaluation"],"basin",entry["basin_index"],"best",best,"wall_s",round(time.perf_counter()-started,1),flush=True)
        search_response=CircularSpectralTDI(response,metric) if args.spectral_objective else response
        search=CircularFamilySearch(metric,prototype,search_response,basins=basins,
            chirp_scaled_bounds=(-args.chirp,args.chirp),evaluation_time_bounds_s=bounds,progress=progress)
        retained=[];rejected=[]
        independent_circular_control=bool((case=="circular" or args.independent_control) and args.retain_run is None)
        if args.independent_control and meta["source"]["eccentricity"]!=0.:
            raise ValueError("Independent unit-overlap control requires an exactly circular injection")
        for candidate in ([] if independent_circular_control else original["candidates"]):
            source=EccentricBinaryParams(**candidate["source"])
            try:retained.append(search.retain(source))
            except ValueError as exc:rejected.append(dict(source=candidate["source"],reason=str(exc)))
        if args.retain_run:
            prior=json.loads((out/f"input_retained_{case}_annual_search.json").read_text())
            for basin in prior["search"]["basins_results"]:
                retained.append(search.retain(EccentricBinaryParams(**basin["best"]["source"])))
            retained.append(search.retain(EccentricBinaryParams(**prior["refinement"]["best"]["source"])))
        if args.retain_candidates:
            for candidate in json.loads((out/"input_retained_candidates.json").read_text()):
                source=EccentricBinaryParams(**candidate["source"])
                try:retained.append(search.retain(source))
                except ValueError as exc:rejected.append(dict(source=candidate["source"],reason=str(exc)))
        retained_log_controls=[]
        if args.retain_evaluations or args.resume_run:
            prior=[json.loads(line) for line in (out/"input_retained_evaluations.jsonl").read_text().splitlines()]
            winners={}
            for entry in prior:
                index=entry["basin_index"]
                if index not in winners or entry["overlap"]>winners[index]["overlap"]:winners[index]=entry
            for entry in winners.values():
                try:
                    item=search.retain(EccentricBinaryParams(**entry["source"]));retained.append(item)
                    retained_log_controls.append(dict(previous_overlap=entry["overlap"],current_overlap=item["profile"]["overlap"],
                        delta_overlap=item["profile"]["overlap"]-entry["overlap"],source=entry["source"]))
                except ValueError as exc:rejected.append(dict(source=entry["source"],reason=str(exc)))
            write_json(out/f"{case}_retained_log_controls.json",retained_log_controls)
            if args.spectral_objective and any(abs(r["delta_overlap"])>1e-8 for r in retained_log_controls):
                raise ValueError("Retained sampled/spectral objective disagreement")
        resume_controls=[]
        for stage in resumed_stages:
            item=search.retain(EccentricBinaryParams(**stage["best"]["source"]))
            difference=item["profile"]["overlap"]-stage["best"]["profile"]["overlap"]
            resume_controls.append(dict(basin_index=stage["basin_index"],previous_overlap=stage["best"]["profile"]["overlap"],
                current_overlap=item["profile"]["overlap"],delta_overlap=difference))
            if abs(difference)>1e-8:raise ValueError("Completed checkpoint basin changed its physical objective")
            stage["checkpoint_origin"]=str(args.resume_run.relative_to(ROOT))
            stage["checkpoint_winner_rechecked_overlap"]=item["profile"]["overlap"]
        if args.resume_run:write_json(out/f"{case}_resume_controls.json",resume_controls)
        if args.seed_method=="coarse":
            result=search.run(sky_points=args.sky_points,coarse_shape=(2*int(args.width)+1,args.chirp_points),
                starts_per_basin=args.starts,ftol=args.ftol,maxiter=args.maxiter,finite_difference_step=args.fd_step)
        else:
            stages=list(resumed_stages)
            if stages:
                search.stage_results=stages
                write_json(out/f"{case}_search_baseline.json",dict(search.result(),basins_results=stages,
                    completed_basins=len(stages),requested_basins=len(basins),settings=prior_checkpoint["settings"]))
                print(case,"resumed",len(stages),"completed basins after objective checks",flush=True)
            for index,basin in enumerate(basins):
                if index<len(resumed_stages):continue
                seeds=rank_envelope_frequency_seeds(metric,response,prototype,basin,
                    sky=equal_area_sky(args.sky_points),chirp_scaled_values=np.linspace(-args.chirp,args.chirp,args.chirp_points),
                    seed_samples=args.seed_samples,window_half_width_cycles=args.seed_window)
                for seed in seeds:seed['start_origin']='independent_envelope_seed'
                informed_seeds=[]
                if args.injection_sky_seeds:
                    # In a fitting-factor calculation the injection is known.
                    # Its sky supplies extra starts only: all six intrinsic
                    # coordinates remain free in every subsequent local fit.
                    # Blind controls deliberately do not use this option.
                    latitude=float(np.sin(meta['source']['beta']))
                    longitude=float(meta['source']['lambda_']%(2*np.pi))
                    informed_seeds=rank_envelope_frequency_seeds(metric,response,prototype,basin,
                        sky=[(latitude,longitude),(-latitude,(longitude+np.pi)%(2*np.pi))],
                        chirp_scaled_values=np.linspace(-args.chirp,args.chirp,args.chirp_points),
                        seed_samples=args.seed_samples,window_half_width_cycles=args.seed_window)
                    for seed in informed_seeds:seed['start_origin']='known_injection_sky_start_all_coordinates_free'
                write_json(out/f"{case}_basin{index}_approximate_seeds.json",
                    [dict(r,source=asdict(r["source"]),quantity="approximate_oscillatory_seed_score_not_overlap")
                        for r in sorted(seeds+informed_seeds,key=lambda r:r['seed_score'],reverse=True)])
                trials=[];starts=[];runs=[]
                for seed in seeds[:args.exact_seeds]+informed_seeds[:4]:
                    point=search.coordinates_of(seed["source"],index)
                    item=search.evaluate(point,index)
                    trials.append((item["profile"]["overlap"],point,seed['start_origin']))
                for item in list(search.cache.values()):
                    point=search.coordinates_of(EccentricBinaryParams(**item["source"]),index)
                    if abs(point[0])<=basin.half_width_cycles:
                        trials.append((item["profile"]["overlap"],point,"retained_full_objective_candidate"))
                ranked=sorted(trials,key=lambda r:r[0],reverse=True)
                # Reserve a blind start even when retained candidates have a
                # higher initial score. Otherwise every restart can remain in
                # the same old local basin, hiding a better solution.
                independent_trials=[r for r in ranked if r[2]=='independent_envelope_seed']
                prioritized=independent_trials[:1]+ranked
                for score,point,origin in prioritized:
                    direction=np.array([np.sqrt(1-point[2]**2)*np.cos(point[3]),np.sqrt(1-point[2]**2)*np.sin(point[3]),point[2]])
                    if not starts or all(np.linalg.norm(direction-other[1])>.05 or
                        abs(point[0]-other[0][0])>1. or abs(point[1]-other[0][1])>4. for other in starts):
                        starts.append((point,direction,origin))
                    if len(starts)==args.starts:break
                for point,_,origin in starts:
                    fit,x,conditioning=search._local_fit(point,index,ftol=args.ftol,maxiter=args.maxiter,finite_difference_step=args.fd_step)
                    item=search.evaluate(x,index);trials.append((item["profile"]["overlap"],x,"local_refinement"))
                    runs.append(dict(start=point.tolist(),start_origin=origin,coordinates=x.tolist(),overlap=item["profile"]["overlap"],
                        success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),function_calls=int(fit.nfev),conditioning=conditioning))
                _,point,_=max(trials,key=lambda r:r[0]);best=search.evaluate(point,index)
                stages.append(dict(basin_index=index,basin=asdict(basin),best=best,best_coordinates=point.tolist(),
                    independent_start_results=runs,approximate_seeds=len(seeds),
                    blind_restart_reserved=True,
                    injection_sky_start_proposals=len(informed_seeds),
                    exact_seed_evaluations=min(args.exact_seeds,len(seeds))+min(4,len(informed_seeds)),
                    frequency_chirp_boundary_distance_scaled=float(min(basin.half_width_cycles-abs(point[0]),
                        point[1]+args.chirp,args.chirp-point[1]))))
                search.stage_results=stages
                result=dict(search.result(),basins_results=stages,completed_basins=len(stages),requested_basins=len(basins),
                    settings=dict(seed_method="coarse_envelope_for_proposals_only",sky_points=args.sky_points,
                        seed_samples=args.seed_samples,seed_window_cycles=args.seed_window,starts_per_basin=args.starts))
                write_json(out/f"{case}_search_baseline.json",result)
                print(case,"completed basin",index,"score",best["profile"]["overlap"],flush=True)
        write_json(out/f"{case}_search_baseline.json",result)
        refined=search.refine_best(ftol=args.ftol/10,finite_difference_step=args.fd_step/5,maxiter=args.maxiter)
        recovered=EccentricBinaryParams(**refined["best"]["source"])
        native=ChunkedNativeResponse(orbit,times,convention,block_samples=65536,cached_blocks=2)
        explicit=metric.compare(native(recovered))
        difference=explicit["overlap"]-refined["best"]["profile"]["overlap"]
        if abs(difference)>2e-4:failures.append(f"{case}: native/envelope winner mismatch")
        if (case=="circular" or args.independent_control) and explicit["overlap"]<1-2e-7:failures.append("Annual circular independent recovery failed")
        record=dict(case=case,source=original["source"],evolution=meta["evolution"],selection=selection,
            search=result,refinement=refined,native_winner_metrics=explicit,native_minus_envelope=difference,
            retained_pool_candidates=retained,rejected_pool_candidates=rejected,all_sky_time_bounds_s=bounds,
            sampled_maximum_barycentric_radius_s=maximum_radius,wall_s=time.perf_counter()-started)
        record["retained_log_controls"]=retained_log_controls
        if args.resume_run:record["resume"]=dict(previous_run=str(args.resume_run.relative_to(ROOT)),
            completed_checkpoint_basins=len(resumed_stages),previous_candidates_reevaluated=True,controls=resume_controls,
            same_noise_band_waveform_and_physical_domain=True)
        record["circular_control_used_injection_pool_seed"]=bool((case=="circular" or args.independent_control) and retained)
        record['known_injection_sky_used_for_starts_only']=args.injection_sky_seeds
        record['all_six_intrinsic_coordinates_free']=True
        write_json(out/f"{case}_annual_search.json",record);records.append(record)
        print(case,"annual optimized candidate",explicit["overlap"],"evaluations",len(search.cache),flush=True)
        del native,response,search,search_response,metric,signal;gc.collect()
    write_json(out/"validation_summary.json",dict(status="annual_baseline_completed" if not failures else "annual_baseline_requires_investigation",
        failures=failures,production_certified=False,cases_completed=len(records),cases_requested=len(args.cases),
        limitations=["Finite basin/start agreement remains empirical", "Covariance is secondary-only diagnostic until laser gate closes",
            "Annual numerical, band, noise-model and contour refinements remain required"]))
    if failures:raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inputs",type=Path,default=ROOT/"outputs/referee/task_2_8_20261008_annual_weighting_pilot")
    p.add_argument("--cases",nargs="+",default=["near_contour"])
    p.add_argument("--band",type=float,nargs=2,default=[2e-5,.04])
    p.add_argument("--envelope-intervals",type=int,default=8192)
    p.add_argument("--width",type=float,default=12.)
    p.add_argument("--chirp",type=float,default=8.)
    p.add_argument("--chirp-points",type=int,default=3)
    p.add_argument("--sky-points",type=int,default=24)
    p.add_argument("--starts",type=int,default=2)
    p.add_argument("--maxiter",type=int,default=90)
    p.add_argument("--ftol",type=float,default=1e-11)
    p.add_argument("--fd-step",type=float,default=5e-6)
    p.add_argument("--peak-fraction",type=float,default=.005)
    p.add_argument("--coverage",type=float,default=.999)
    p.add_argument("--retain-run",type=Path)
    p.add_argument("--seed-method",choices=["coarse","envelope"],default="coarse")
    p.add_argument("--seed-samples",type=int,default=2048)
    p.add_argument("--seed-window",type=int,default=256)
    p.add_argument("--exact-seeds",type=int,default=12)
    p.add_argument("--spectral-objective",action="store_true")
    p.add_argument("--spectral-validation",type=Path)
    p.add_argument("--retain-evaluations",type=Path)
    p.add_argument("--resume-run",type=Path,help="Resume the completed basin prefix and retain all discovered basin winners")
    p.add_argument("--independent-control",action="store_true",help="Exclude injection-pool seeds and require circular native recovery")
    p.add_argument("--retain-candidates",type=Path,help="Explicit physical templates to re-evaluate in the current metric")
    p.add_argument('--injection-sky-seeds',action='store_true',
        help='Add known-injection and antipodal sky starts; all intrinsic coordinates remain free')
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        if args.output.exists() and not (args.output/"validation_summary.json").exists():
            write_json(args.output/"validation_summary.json",dict(status="failed_annual_search",production_certified=False,
                exception_type=type(exc).__name__,exception=str(exc),completed_original_optimizer_procedure=False,
                limitation="Saved checkpoint and evaluation log preserve completed work; no final FF inferred"))
        raise
