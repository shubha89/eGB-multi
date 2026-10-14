"""Task 2.6 native sky/mass/multiple-frequency-basin validation.

Clean injections use the original eGB/native factorized TDI path. The same
adopted finite reference PSD, physical band and norm weight all candidates.
Spectrum-informed seeds are appropriate for FF calculation, not detection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/referee/.cache'))
sys.path.insert(0,str(ROOT/'src'))

import numpy as np

from egb_multi.circular_family_search import CircularFamilySearch, select_frequency_basins
from egb_multi.circular_fit import CircularAmplitudeBasis, maximize_circular_orientation
from egb_multi.circular_search import CircularFrequencyChirpSearch
from egb_multi.comparison import AESeries, AEInnerProduct
from egb_multi.eccentric import EccentricBinaryParams, eccentric_harmonic_angular_frequency
from egb_multi.external import default_lisaorbits, link_delays_from_lisaorbits
from referee_task_2_frequency_chirp import load_psd, prepared_response, source_time_bounds
from referee_task_2_objective_orientation import assert_local, make_clean_response, write_json

CASES = ('circular_unknown_sky_mass','eccentric_fixed','eccentric_pm','eccentric_high_e')


def all_sky_source_time_bounds(geometry):
    # For any unit propagation direction k, |p dot k| <= ||p||. Include
    # emission LTT and all pre-crop samples; no injection sky is assumed.
    radius_s = np.linalg.norm(geometry['p_sender'],axis=-1)
    radius_r = np.linalg.norm(geometry['p_receiver'],axis=-1)
    sender = geometry['times'][None,:]-geometry['length']
    receiver = geometry['times'][None,:]
    return (float(min((sender-radius_s).min(),(receiver-radius_r).min())),
            float(max((sender+radius_s).max(),(receiver+radius_r).max())))


def weighted_shape_residual(metric, reference, candidate):
    a,b = metric.prepare(reference),metric.prepare(candidate)
    amplitude = metric.inner(a,b)/metric.inner(b,b)
    delta = a-amplitude*b
    return float(np.sqrt(metric.inner(delta,delta)/metric.inner(a,a)))


def validate(out,args):
    psd, convention = load_psd(args.psd_run)
    orbit = default_lisaorbits('esa-trailing')
    times = np.arange(args.samples,dtype=float)*args.dt
    delays = link_delays_from_lisaorbits(orbit,times)
    halo = int(np.ceil(8*max(v.max() for v in delays.values())/args.dt))+4*(convention.measurement_order+1)
    if times.size <= 2*halo+16:
        raise ValueError('Insufficient duration after interpolation halo')
    retained = times[halo:-halo]
    duration, epoch = retained.size*args.dt,float(.5*(retained[0]+retained[-1]))
    response, geometry = prepared_response(orbit,times,convention,halo)
    native_response,_ = make_clean_response(orbit,times,convention,halo)
    all_sky_bounds = all_sky_source_time_bounds(geometry)
    definitions = [
        ('circular_unknown_sky_mass',.003,0.,.313,-.717,.91,.77,.46,'fixed'),
        ('eccentric_fixed',.001,.1,0.,0.,1.2,.713,.83,'fixed'),
        ('eccentric_pm',.003,.3,0.,0.,.8,.213,.4,'peters_mathews'),
        ('eccentric_high_e',.001,.6,0.,0.,1.2,.713,.83,'fixed')]
    if args.cases:
        definitions = [case for case in definitions if case[0] in args.cases]
    results,failures = [],[]
    for name,f_center,e,alpha,gamma,inc,phase,psi,evolution in definitions:
        start = time.perf_counter()
        print(f'{name}: N={args.samples}, dt={args.dt:g}s, T_retained={duration:g}s',flush=True)
        truth = EccentricBinaryParams(mean_motion=np.pi*(f_center+alpha/duration),
            eccentricity=e,fdot=gamma/duration**2,t0=epoch,inclination=inc,phi0=phase,psi=psi,
            beta=.2,lambda_=-1.,m1_solar=.8,m2_solar=.6)
        if e == 0:
            truth = replace(truth,beta=.43,lambda_=6.08,m1_solar=1.12,m2_solar=.32)
        signal = native_response(truth,evolution)
        metric = AEInnerProduct(signal,psd,(.2*f_center,10*f_center))
        prototype = replace(truth,eccentricity=0.,mean_motion=np.pi*f_center,fdot=0.,
            inclination=.6,phi0=.1,psi=.2)
        if e == 0:
            prototype = replace(prototype,beta=-.6,lambda_=.8,m1_solar=.5,m2_solar=.5)
        direct = response(truth,evolution)
        native_error = weighted_shape_residual(metric,signal,direct)
        if native_error > 1e-9:
            failures.append(f'{name}: cached/native response disagreement')
        # Rerun the restricted fixed-sky/mass stage at this exact sampling and
        # weighting, rather than comparing unlike archived durations or PSDs.
        fixed = CircularFrequencyChirpSearch(metric,prototype,response,
            frequency_scaled_bounds=(-4.,4.),chirp_scaled_bounds=(-4.,4.),
            evaluation_time_bounds_s=source_time_bounds(geometry,prototype))
        fixed_result = fixed.run(coarse_shape=(5,3),starts=1)
        baseline_source = EccentricBinaryParams(**fixed_result['best']['source'])
        basins, selection = select_frequency_basins(metric,signal,half_width_cycles=4.,
            minimum_peak_fraction=.01,target_coverage=.99,extra_frequencies=(f_center,))
        progress_path = out/f'{name}_evaluations.jsonl'
        def progress(stage):
            def record(entry):
                with progress_path.open('a') as stream:
                    stream.write(json.dumps(dict(entry,stage=stage),allow_nan=False)+'\n')
                if entry['unique_intrinsic_evaluation']%100==0:
                    print(f"  {stage}: {entry['unique_intrinsic_evaluation']} evaluations, basin={entry['basin_index']}, overlap={entry['overlap']:.10g}",flush=True)
            return record
        family = CircularFamilySearch(metric,prototype,response,basins=basins,
            chirp_scaled_bounds=(-4.,4.),evaluation_time_bounds_s=all_sky_bounds,progress=progress('initial'))
        retained_baseline = family.retain(baseline_source)
        if abs(retained_baseline['profile']['overlap']-fixed_result['best']['profile']['overlap']) > 1e-9:
            failures.append(f'{name}: retained fixed-stage score changed under identical objective')
        print(f'  selected {len(basins)} initial basins; tapered seed coverage={selection["covered_seed_spectrum_power_fraction"]:.8g}',flush=True)
        initial = family.run(sky_points=16,coarse_shape=(5,3),starts_per_basin=2)
        refined = family.refine_best()
        # Expand carrier/chirp bounds, sky starts and peak coverage. Every
        # initial basin winner is re-evaluated and retained in the expanded run.
        expanded_basins,expanded_selection = select_frequency_basins(metric,signal,
            half_width_cycles=6.,minimum_peak_fraction=.005,target_coverage=.999,
            extra_frequencies=tuple(b.center_hz for b in basins))
        expanded_family = CircularFamilySearch(metric,prototype,response,basins=expanded_basins,
            chirp_scaled_bounds=(-8.,8.),evaluation_time_bounds_s=all_sky_bounds,progress=progress('expanded'))
        for basin_result in initial['basins_results']:
            expanded_family.retain(EccentricBinaryParams(**basin_result['best']['source']))
        expanded_family.retain(EccentricBinaryParams(**refined['best']['source']))
        print(f'  expanded to {len(expanded_basins)} basins and 32 sky points; seed coverage={expanded_selection["covered_seed_spectrum_power_fraction"]:.8g}',flush=True)
        expanded = expanded_family.run(sky_points=32,coarse_shape=(5,3),starts_per_basin=3)
        final = expanded_family.refine_best()
        recovered = EccentricBinaryParams(**final['best']['source'])
        explicit = metric.compare(native_response(recovered))
        global_orientation = maximize_circular_orientation(metric,CircularAmplitudeBasis.generate(recovered,response))
        if abs(explicit['overlap']-final['best']['profile']['overlap']) > 1e-9 or abs(global_orientation['overlap']-final['best']['profile']['overlap']) > 1e-8:
            failures.append(f'{name}: native/global orientation validation disagrees')
        baseline_score = fixed_result['best']['profile']['overlap']
        expansion_change = final['best']['profile']['overlap']-refined['best']['profile']['overlap']
        refinement_change = final['best']['profile']['overlap']-expanded['best']['profile']['overlap']
        if final['best']['profile']['overlap'] < baseline_score-1e-10 or expansion_change < -1e-10:
            failures.append(f'{name}: search expansion discarded a better candidate')
        if expansion_change > 1e-5 or refinement_change > 1e-7:
            failures.append(f'{name}: expanded bounds/starts or refinement are not stable')
        nearest = min(expanded['basins_results'],key=lambda result:abs(result['basin']['center_hz']-recovered.mean_motion/np.pi))
        if nearest['frequency_chirp_boundary_distance_scaled'] < .01:
            failures.append(f'{name}: winning basin frequency/chirp boundary is active')
        if e == 0 and explicit['overlap'] < 1.-2e-7:
            failures.append(f'{name}: unknown-sky/mass circular waveform recovery failed')
        mass_audit = []
        best_response = response(recovered)
        for m1,m2 in ((.1,.1),(1.4,.1),(1.4,1.4),(.8,.6)):
            changed = replace(recovered,m1_solar=m1,m2_solar=m2)
            mass_audit.append({'masses_solar':[m1,m2],
                'amplitude_fitted_weighted_shape_residual':weighted_shape_residual(metric,best_response,response(changed)),
                'injection_overlap_at_fixed_other_parameters':metric.compare(response(changed))['overlap']})
        labels = [{'harmonic_label':h,'frequency_hz':eccentric_harmonic_angular_frequency(truth,h)/(2*np.pi)}
            for h in range(1,21)]
        closest_label = min(labels,key=lambda item:abs(item['frequency_hz']-recovered.mean_motion/np.pi))
        entry = {'case':name,'injection':asdict(truth),'prototype':asdict(prototype),'evolution':evolution,
            'samples_before_crop':args.samples,'dt_s':args.dt,'halo_samples_each_side':halo,
            'all_sky_source_time_bounds_s':list(all_sky_bounds),'native_vs_cached_weighted_shape_residual':native_error,
            'fixed_sky_mass_baseline':fixed_result,'initial_basin_selection':selection,
            'initial_family_search':initial,'initial_refinement':refined,
            'expanded_basin_selection':expanded_selection,'expanded_family_search':expanded,'final_refinement':final,
            'expansion_overlap_change':expansion_change,'final_refinement_overlap_change':refinement_change,
            'explicit_native_metrics':explicit,'global_orientation_check':global_orientation,'mass_shape_audit_at_best':mass_audit,
            'winning_frequency_over_nominal_carrier':float(recovered.mean_motion/np.pi/f_center),
            'nearest_labelled_quadrupole_sideband':closest_label,
            'circular_recovery_sky_direction_error':float(np.linalg.norm(recovered.propagation_direction()-truth.propagation_direction())) if e==0 else None,
            'wall_s':time.perf_counter()-start}
        results.append(entry)
        write_json(out/'sky_mass_harmonic_validation.json',results)
        print(f"  finished: fixed={baseline_score:.10g}, family={final['best']['profile']['overlap']:.12g}, expansion_change={expansion_change:.3g}",flush=True)
    summary = {'status':'passed_Task_2_6_staged_validation' if not failures else 'discrepancy_requires_investigation',
        'failures':failures,'cases':len(results),'noise_model':'secondary_plus_laser',
        'max_native_vs_cached_weighted_shape_residual':max(r['native_vs_cached_weighted_shape_residual'] for r in results),
        'max_expansion_overlap_change':max(r['expansion_overlap_change'] for r in results),
        'max_final_refinement_overlap_change':max(r['final_refinement_overlap_change'] for r in results),
        'full_fitting_factor_certified':False,'production_convergence_certified':False,
        'limitation':'Full sky and ordered masses in finite spectrum-informed basins; short native validation, finite diagonal PSD; comprehensive optimizer and production certification remain 2.7-2.8.'}
    write_json(out/'validation_summary.json',summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--psd-run',type=Path,default=ROOT/'outputs/referee/task_2_3_reference_integrated_20261007')
    parser.add_argument('--samples',type=int,default=8192)
    parser.add_argument('--dt',type=float,default=10.)
    parser.add_argument('--cases',nargs='+',choices=CASES)
    args = parser.parse_args()
    out,args.psd_run = args.output.resolve(),args.psd_run.resolve()
    if not out.is_relative_to(ROOT) or not args.psd_run.is_relative_to(ROOT) or args.dt<=0:
        raise ValueError('Local output/PSD paths and positive cadence required')
    out.mkdir(parents=True,exist_ok=False)
    assert_local()
    files = list((ROOT/'src/egb_multi').glob('*.py')) + [Path(__file__).resolve(),
        ROOT/'benchmarks/referee_task_2_frequency_chirp.py',ROOT/'benchmarks/referee_task_2_objective_orientation.py',ROOT/'pyproject.toml',
        args.psd_run/'tdi2_psd.csv',args.psd_run/'psd_metadata.json']
    files += [ROOT/'tests'/name for name in ('test_circular_family_search.py','test_circular_fit.py','test_circular_search.py','test_comparison.py','test_tdi.py')]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = out/'snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'root':str(ROOT),'python':sys.executable,'configuration':dict(vars(args),output=str(out),psd_run=str(args.psd_run)),
        'versions':{name:version(name) for name in ('numpy','scipy','jax','pytdi','lisaorbits')},'sha256':hashes,
        'predetermined_gates':{'native_response_error':1e-9,'profile_explicit_score_difference':1e-9,
            'profile_global_score_difference':1e-8,'expansion_score_change':1e-5,
            'refinement_score_change':1e-7,'circular_overlap_deficit':2e-7,
            'winning_frequency_chirp_boundary_distance_scaled':.01}})
    summary = validate(out,args)
    assert_local()
    print(json.dumps(summary,indent=2),flush=True)
    if summary['failures']:
        raise SystemExit(1)
