"""Independent optimizer restarts and circular-limit checks for Task 2.7.

Uses only the local exact circular family and the adopted provisional PSD.
The archived 2.6 winners are retained for comparison, never counted as an
independent recovery. DE seeds use only declared bounds. All evaluated points
are retained even when an optimizer stops unsuccessfully.
"""

import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/referee/.cache'))
import numpy as np
from scipy.optimize import differential_evolution,minimize
from egb_multi.circular_family_search import CircularFamilySearch,FrequencyBasin,select_frequency_basins
from egb_multi.comparison import AEInnerProduct
from egb_multi.eccentric import EccentricBinaryParams,peters_mathews_derivatives
from referee_task_2_frequency_chirp import load_psd,prepared_response
from referee_task_2_sky_mass_harmonics import all_sky_source_time_bounds
from referee_task_2_objective_orientation import assert_local,make_clean_response,write_json
from egb_multi.external import default_lisaorbits,link_delays_from_lisaorbits,state_from_lisaorbits
from egb_multi.circular_refinement import refine_sky_partners,refine_with_waveform_jacobian


def provenance(out,args):
    files=list((ROOT/'src/egb_multi').glob('*.py'))+list((ROOT/'tests').glob('test_*.py'))
    files += [Path(__file__),ROOT/'benchmarks/referee_task_2_frequency_chirp.py',
        ROOT/'benchmarks/referee_task_2_sky_mass_harmonics.py',ROOT/'benchmarks/referee_task_2_objective_orientation.py',
        args.psd_run/'tdi2_psd.csv',args.psd_run/'psd_metadata.json']
    hashes={}
    for p in files:
        relative=p.relative_to(ROOT);target=out/'snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        hashes[str(relative)]=hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'root':str(ROOT),'python':sys.executable,'sha256':hashes,
        'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'predetermined_gates':{'independent_best_score_agreement':args.optimizer_error,
            'circular_overlap_deficit':2e-7,'reference_retention':1e-10,
            'small_e_waveform_convergence':'residual decreases as eccentricity decreases',
            'native_profile_score_difference':1e-9},
        'scope':'short native optimizer audit; production settings and one-year coverage are separate gates'})


def archived_cases(names):
    result=[]
    for name in names:
        folder=ROOT/f'outputs/referee/task_2_6_conditioned_20261007_{name}'
        path=folder/'sky_mass_harmonic_validation.json'
        record=json.loads(path.read_text())[0]
        result.append((name,EccentricBinaryParams(**record['injection']),
            EccentricBinaryParams(**record['prototype']),record['evolution'],record,path))
    return result


def restart_coverage_complete(records, basin_count, seeds):
    """A native recovery alone cannot pass a skipped restart experiment."""
    expected={(i,seed) for i in range(basin_count) for seed in seeds}
    actual={(row['basin_index'],row['seed']) for row in records}
    return bool(expected) and actual==expected and len(records)==len(expected)


def run(out,args):
    assert_local();psd,conv=load_psd(args.psd_run)
    orbit=default_lisaorbits('esa-trailing')
    times=np.arange(args.samples)*args.dt
    delays=link_delays_from_lisaorbits(orbit,times)
    halo=int(np.ceil(8*max(d.max() for d in delays.values())/args.dt))+4*(conv.measurement_order+1)
    if len(times)<=2*halo+16: raise ValueError('Insufficient retained samples')
    response,geometry=prepared_response(orbit,times,conv,halo)
    native,_=make_clean_response(orbit,times,conv,halo)
    duration=(len(times)-2*halo)*args.dt;epoch=(times[halo]+times[-halo-1])/2
    bounds=all_sky_source_time_bounds(geometry)
    sky_state=state_from_lisaorbits(orbit,np.array([epoch-3600,epoch,epoch+3600]),frame='ecliptic')
    r=sky_state.barycenter[1];v=(sky_state.barycenter[2]-sky_state.barycenter[0])/7200
    acceleration=(sky_state.barycenter[2]-2*r+sky_state.barycenter[0])/3600**2
    plane=sky_state.positions[1];normal=np.cross(plane[1]-plane[0],plane[2]-plane[0])
    if args.mode=='restart':
        definitions=archived_cases(args.cases)
        if args.samples!=8192 or args.dt!=10: raise ValueError('Archived comparisons require identical N=8192, dt=10')
    else:
        # Sky and masses differ from the source used to initialize the search.
        prototype=EccentricBinaryParams(mean_motion=np.pi*.001,eccentricity=0,t0=epoch,
            beta=-.6,lambda_=.8,m1_solar=.5,m2_solar=.5,inclination=.6,phi0=.1,psi=.2)
        definitions=[]
        for e in args.small_e_values:
            truth=replace(prototype,eccentricity=e,mean_motion=np.pi*(.001+.713/duration),
                fdot=-.517/duration**2,beta=.43,lambda_=6.08,m1_solar=1.12,m2_solar=.32,
                inclination=.91,phi0=.77,psi=.46)
            definitions.append((f'small_e_{e:g}',truth,prototype,'fixed',None,None))
    records,failures=[],[]
    circular_reference=None
    for name,truth,prototype,evolution,old,old_path in definitions:
        started=time.perf_counter();signal=native(truth,evolution)
        nominal=prototype.mean_motion/np.pi
        metric=AEInnerProduct(signal,psd,(args.band_low_factor*nominal,args.band_high_factor*nominal))
        basins,selection=select_frequency_basins(metric,signal,half_width_cycles=12,
            minimum_peak_fraction=.002,target_coverage=.9999,extra_frequencies=(nominal,))
        log=out/f'{name}_evaluations.jsonl'
        def progress(entry):
            with log.open('a') as stream: stream.write(json.dumps(entry,allow_nan=False)+'\n')
            if entry['unique_intrinsic_evaluation']%100==0:
                print(f"{name}: {entry['unique_intrinsic_evaluation']} intrinsic evaluations, basin {entry['basin_index']}, M={entry['overlap']:.9g}",flush=True)
        search=CircularFamilySearch(metric,prototype,response,basins=basins,
            chirp_scaled_bounds=(-16,16),evaluation_time_bounds_s=bounds,progress=progress)
        retained=[]
        if old:
            for stage in old['expanded_family_search']['basins_results']:
                retained.append(search.retain(EccentricBinaryParams(**stage['best']['source'])))
            retained.append(search.retain(EccentricBinaryParams(**old['final_refinement']['best']['source'])))
        original_candidates=[]
        gr=float(peters_mathews_derivatives(prototype.mean_motion,0.,prototype.total_mass_kg,prototype.symmetric_mass_ratio)[0]/np.pi)
        for factor in (0,.25,.5,1,2,4,8,16,32):
            source=replace(prototype,fdot=factor*gr)
            original_candidates.append(search.retain(source))
        baseline=max(item['profile']['overlap'] for item in original_candidates)
        if args.mode=='small-e':
            # Preserve <=1/T frequency spacing when expanding to +/-12/T.
            # A seven-point grid would be spaced 4/T apart and can seed an
            # offset circular injection in a secondary maximum.
            staged=search.run(sky_points=32,coarse_shape=(25,9),starts_per_basin=3,
                ftol=1e-12,maxiter=120,finite_difference_step=5e-6)
            refined=search.refine_best()
            if args.sky_partners:
                sky_starts=refine_sky_partners(search,r,v,acceleration,normal)
                refined=refine_with_waveform_jacobian(search)
                refined['additional_sky_starts']=sky_starts
            independent=[]
        else:
            # Recheck every selected basin independently; do not skip weak
            # basins because the old search gave them small objective values.
            independent=[]
            if args.restart_domain=='union':
                # A discrete basin coordinate samples the whole declared union.
                # Each basin gets two population starts at its spectrum-derived
                # centre; these use no injection or archived fit parameters.
                intrinsic_bounds=search._bounds(0,np.pi)
                union_bounds=intrinsic_bounds+[(0.,float(len(basins)))]
                for seed in args.seeds:
                    rng=np.random.default_rng(seed)
                    size=max(7*args.population,2*len(basins))
                    low,high=np.array(union_bounds).T
                    population=rng.uniform(low,high,size=(size,7))
                    for row in range(2*len(basins)):
                        population[row,0:2]=0.
                        population[row,6]=row%len(basins)+.5
                    visited={i:[] for i in range(len(basins))}
                    def objective_union(x):
                        index=min(len(basins)-1,int(x[6]))
                        item=search.evaluate(x[:6],index)
                        visited[index].append((item['profile']['overlap'],np.array(x[:6])))
                        return -item['profile']['overlap']
                    fit=differential_evolution(objective_union,union_bounds,
                        rng=rng,init=population,maxiter=args.iterations,tol=1e-7,atol=1e-9,polish=False)
                    polished=[]
                    # Refine every basin's best independently, using one common
                    # objective and retaining all points from this seed alone.
                    for index in range(len(basins)):
                        _,point=max(visited[index],key=lambda pair:pair[0])
                        def local_objective(x,index=index):
                            item=search.evaluate(x,index)
                            visited[index].append((item['profile']['overlap'],np.array(x)))
                            return -item['profile']['overlap']
                        # _local_fit calls search.evaluate directly; record its
                        # endpoint and compare with all DE evaluations.
                        polish,x,conditioning=search._local_fit(point,index,ftol=1e-12,maxiter=120,finite_difference_step=2e-6)
                        endpoint=-local_objective(x)
                        best=max(endpoint,max(pair[0] for pair in visited[index]))
                        independent.append({'basin_index':index,'seed':seed,'best_overlap':float(best),
                            'DE_success':bool(fit.success),'DE_iterations':int(fit.nit),
                            'DE_basin_evaluations':len(visited[index])-1,
                            'restart_domain':'all_declared_basins_union','population_size':size,
                            'polish_method':'SLSQP','polish_success':bool(polish.success),
                            'polish_evaluations':int(polish.nfev)})
                    write_json(out/f'{name}_independent_progress.json',independent)
            else:
                for index,basin in enumerate(basins):
                    physical_bounds=search._bounds(index,np.pi)
                    for seed in args.seeds:
                        rejected=[]
                        def objective(x):
                            # Reject Powell extrapolations outside the box;
                            # do not clip parameters or weaken the evaluator.
                            limits=np.array(physical_bounds)
                            if np.any(x<limits[:,0]) or np.any(x>limits[:,1]):
                                rejected.append(np.asarray(x).tolist());return np.inf
                            return -search.evaluate(x,index)['profile']['overlap']
                        fit=differential_evolution(objective,physical_bounds,
                            rng=np.random.default_rng(seed),popsize=args.population,
                            maxiter=args.iterations,tol=1e-7,atol=1e-9,polish=False)
                        # Powell supplies a derivative-free check distinct from the
                        # staged SLSQP method; each seed is refined independently.
                        if args.polish=='Powell':
                            polish=minimize(objective,fit.x,method='Powell',bounds=physical_bounds,
                                options={'xtol':1e-7,'ftol':1e-12,'maxiter':120,'maxfev':3000})
                        else:
                            polish,x,conditioning=search._local_fit(fit.x,index,ftol=1e-12,maxiter=120,finite_difference_step=2e-6)
                            polish.fun=objective(x)
                        best=max((-fit.fun,-polish.fun))
                        independent.append({'basin_index':index,'seed':seed,'best_overlap':float(best),
                            'DE_success':bool(fit.success),'DE_iterations':int(fit.nit),
                            'rejected_outside_domain_proposals':rejected,'polish_method':args.polish,'polish_success':bool(polish.success),'polish_evaluations':int(polish.nfev)})
                    write_json(out/f'{name}_independent_progress.json',independent)
            staged=None;refined=search.refine_best()
        best=refined['best'];recovered=EccentricBinaryParams(**best['source'])
        explicit=metric.compare(native(recovered))
        if abs(explicit['overlap']-best['profile']['overlap'])>1e-9: failures.append(f'{name}: profile/native difference')
        if best['profile']['overlap']<baseline-1e-10: failures.append(f'{name}: discarded restricted candidate')
        if truth.eccentricity==0 and explicit['overlap']<1-2e-7: failures.append(f'{name}: circular recovery')
        if args.mode=='restart':
            if not restart_coverage_complete(independent,len(basins),args.seeds):
                failures.append(f'{name}: incomplete or duplicate independent restart coverage')
        seed_best={str(seed):max(item['best_overlap'] for item in independent if item['seed']==seed) for seed in args.seeds if any(item['seed']==seed for item in independent)}
        if seed_best and best['profile']['overlap']-min(seed_best.values())>args.optimizer_error:
            failures.append(f'{name}: independent global seeds do not agree with retained maximum')
        shape=None
        if args.mode=='small-e':
            circular=native(replace(truth,eccentricity=0))
            a,b=metric.prepare(signal),metric.prepare(circular)
            shape=float(np.sqrt(metric.inner(a-b,a-b)/metric.reference_norm))
        record={'case':name,'injection':asdict(truth),'prototype':asdict(prototype),
            'evolution':evolution,'retained_duration_s':duration,'baseline_profile_overlap':baseline,
            'retained_archived_overlap':max((x['profile']['overlap'] for x in retained),default=None),
            'archived_source':str(old_path.relative_to(ROOT)) if old_path else None,
            'archived_source_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest() if old_path else None,
            'basin_selection':selection,'independent_results':independent,'independent_seed_best':seed_best,
            'staged_search':staged,'refined_search':refined,'explicit_native_metrics':explicit,
            'same_parameter_circular_shape_residual':shape,'wall_s':time.perf_counter()-started}
        records.append(record);write_json(out/'optimizer_validation.json',records)
        print(f"{name} complete: overlap={best['profile']['overlap']:.12g}, independent={seed_best}",flush=True)
    if args.mode=='small-e':
        residuals=[r['same_parameter_circular_shape_residual'] for r in records]
        if not all(a<b for a,b in zip(residuals,residuals[1:])): failures.append('small-e waveform residual does not converge to zero')
        deficits=[1-r['refined_search']['best']['profile']['overlap'] for r in records]
        if len(deficits)>2 and (deficits[1]>1e-4 or deficits[2]>1e-3): failures.append('small-e optimized recovery does not approach unity')
    summary={'status':'passed_declared_optimizer_audit' if not failures else 'discrepancy_requires_investigation',
        'failures':failures,'mode':args.mode,'cases':len(records),'production_certified':False,
        'limitation':'Native short-observation audit; one-year optimizer and production error budget remain separate'}
    write_json(out/'validation_summary.json',summary);assert_local();print(json.dumps(summary,indent=2),flush=True)
    return not failures


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['restart','small-e'],required=True)
    p.add_argument('--psd-run',type=Path,default=ROOT/'outputs/referee/task_2_3_reference_integrated_20261007')
    p.add_argument('--cases',nargs='+',default=['circular_unknown_sky_mass','eccentric_fixed','eccentric_pm','eccentric_high_e'])
    p.add_argument('--samples',type=int,default=8192);p.add_argument('--dt',type=float,default=10.)
    p.add_argument('--seeds',nargs='+',type=int,default=[20261008,20261009])
    p.add_argument('--population',type=int,default=4);p.add_argument('--iterations',type=int,default=35)
    p.add_argument('--optimizer-error',type=float,default=1e-4)
    p.add_argument('--band-low-factor',type=float,default=.2)
    p.add_argument('--band-high-factor',type=float,default=10.)
    p.add_argument('--restart-domain',choices=['per-basin','union'],default='per-basin')
    p.add_argument('--polish',choices=['Powell','SLSQP'],default='Powell')
    p.add_argument('--sky-partners',action='store_true')
    p.add_argument('--small-e-values',nargs='+',type=float,default=[0.,.001,.003,.01])
    args=p.parse_args();out=args.output.resolve();args.psd_run=args.psd_run.resolve()
    if not out.is_relative_to(ROOT) or not args.psd_run.is_relative_to(ROOT): raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);provenance(out,args)
    if not run(out,args): raise SystemExit(1)
