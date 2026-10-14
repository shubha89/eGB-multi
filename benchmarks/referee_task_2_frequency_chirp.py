"""Task 2.5: bounded continuous frequency/chirp validation at fixed sky/masses.

Reuses the adopted laser-enabled PSD, without additive waveform noise. Each
candidate retains the exact physical six-response orientation/amplitude fit.
Short controlled injections validate optimization, not production convergence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
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

from egb_multi.circular_fit import CircularAmplitudeBasis,profile_circular_orientation,maximize_circular_orientation
from egb_multi.circular_search import CircularFrequencyChirpSearch
from egb_multi.comparison import AESeries,AEPSD,AEInnerProduct,TDIConvention
from egb_multi.eccentric import EccentricBinaryParams,peters_mathews_derivatives
from egb_multi.eccentric_jax import eccentric_links_jax,precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits,state_from_lisaorbits,link_delays_from_lisaorbits
from egb_multi.noise import aet_from_xyz
from egb_multi.tdi import PreparedGWTDIMichelson
from referee_task_2_objective_orientation import assert_local,make_clean_response,write_json


def prepared_response(orbit,times,convention,halo):
    state = state_from_lisaorbits(orbit,times,frame='ecliptic')
    geometry = precompute_jax_link_geometry(state,orbits=orbit,frame='ecliptic')
    delays = link_delays_from_lisaorbits(orbit,times)
    operator = PreparedGWTDIMichelson(state,delays,generation=convention.generation,
        measurement_order=convention.measurement_order,delay_order=convention.delay_order)
    def response(source,evolution='fixed'):
        links = eccentric_links_jax(source,geometry,physics_mode='1pn',evolution_mode=evolution,batch_size=1)
        ae = aet_from_xyz(operator(links))
        return AESeries(times[halo:-halo],{ch:ae[ch][halo:-halo] for ch in ('A','E')},convention)
    return response,geometry


def source_time_bounds(geometry,source):
    direction = source.propagation_direction()
    sender = geometry['times'][None,:]-geometry['length']-np.einsum('lti,i->lt',geometry['p_sender'],direction)
    receiver = geometry['times'][None,:]-np.einsum('lti,i->lt',geometry['p_receiver'],direction)
    return (float(min(sender.min(),receiver.min(),geometry['times'][0])),
            float(max(sender.max(),receiver.max(),geometry['times'][-1])))


def load_psd(folder):
    metadata = json.loads((folder/'psd_metadata.json').read_text())
    if metadata['noise_model']!='secondary_plus_laser' or metadata['enabled_noise_contributions']!=['laser','test-mass','oms']:
        raise ValueError('The adopted laser/test-mass/OMS reference is required')
    convention = TDIConvention(**metadata['convention'])
    if (convention.generation,convention.measurement_order,convention.delay_order)!=(2,45,5):
        raise ValueError('This validation uses the existing TDI2 order45/delay-order5 reference')
    data = np.genfromtxt(folder/'tdi2_psd.csv',delimiter=',',names=True)
    return AEPSD(data['frequency_hz'],{'A':data['PSD_A_fractional_per_hz'],
        'E':data['PSD_E_fractional_per_hz']},convention,metadata),convention


def validate(out,args):
    psd,convention = load_psd(args.psd_run)
    orbit = default_lisaorbits('esa-trailing')
    times = np.arange(args.samples,dtype=float)*args.dt
    delays = link_delays_from_lisaorbits(orbit,times)
    halo = int(np.ceil(8*max(v.max() for v in delays.values())/args.dt))+4*(convention.measurement_order+1)
    if times.size<=2*halo+16:
        raise ValueError('Insufficient duration after the interpolation halo')
    t = times[halo:-halo]
    duration,midpoint = t.size*args.dt,float(.5*(t[0]+t[-1]))
    response,geometry = prepared_response(orbit,times,convention,halo)
    native_response,_ = make_clean_response(orbit,times,convention,halo)
    cases = [
        ('circular_positive_offset',.001,0.,1.371,1.213,1.217,.923,.67,'fixed'),
        ('circular_negative_offset',.003,0.,-.917,-1.117,np.pi/2,.613,1.12,'fixed'),
        ('eccentric_fixed',.001,.1,0.,0.,1.2,.713,.83,'fixed'),
        ('eccentric_pm',.003,.3,0.,0.,.8,.213,.4,'peters_mathews')]
    if args.cases:
        cases = [case for case in cases if case[0] in args.cases]
    results,failures = [],[]
    for name,f_center,e,alpha,gamma,inc,phase,psi,evolution in cases:
        start = time.perf_counter()
        print(f'{name}: N={args.samples}, T_retained={duration:g}s, f_center={f_center:g}Hz',flush=True)
        truth = EccentricBinaryParams(mean_motion=np.pi*(f_center+alpha/duration),
            fdot=gamma/duration**2,eccentricity=e,t0=midpoint,inclination=inc,phi0=phase,psi=psi,
            beta=.2,lambda_=-1.,m1_solar=.8,m2_solar=.6)
        # Injection uses the original native factorized path, independently of
        # the cached-delay template evaluator introduced for this search.
        signal = native_response(truth,evolution)
        scale = 2.3 if e==0 else 1.
        signal = AESeries(signal.t,{ch:scale*v for ch,v in signal.channels.items()},convention)
        metric = AEInnerProduct(signal,psd,(.2*f_center,10*f_center))
        template = replace(truth,eccentricity=0.,mean_motion=np.pi*f_center,fdot=0.,
            inclination=.6,psi=.2,phi0=.1)
        direct = response(truth,evolution)
        delta = metric.prepare(signal)-scale*metric.prepare(direct)
        native_error = float(np.sqrt(metric.inner(delta,delta)/metric.reference_norm))
        if native_error>1e-9:
            failures.append(f'{name}: cached TDI disagrees with native injection path')
        log = out/f'{name}_evaluations.jsonl'
        def progress(entry):
            with log.open('a') as stream:
                stream.write(json.dumps(entry,allow_nan=False)+'\n')
            if entry['unique_intrinsic_evaluation']%25==0:
                print(f"  {entry['unique_intrinsic_evaluation']} intrinsic evaluations; latest overlap={entry['overlap']:.9g}",flush=True)
        search = CircularFrequencyChirpSearch(metric,template,response,
            frequency_scaled_bounds=(-3.5,3.5),chirp_scaled_bounds=(-4.,4.),
            evaluation_time_bounds_s=source_time_bounds(geometry,template),progress=progress)
        gr_fdot = peters_mathews_derivatives(template.mean_motion,0.,template.total_mass_kg,template.symmetric_mass_ratio)[0]/np.pi
        restricted = []
        for factor in (0.,.25,.5,1.,2.,4.,8.,16.,32.):
            point = (0.,factor*gr_fdot*duration**2)
            entry = search.evaluate(point)
            restricted.append({'factor':factor,'coordinates':list(point),'overlap':entry['profile']['overlap']})
        baseline = max(restricted,key=lambda item:item['overlap'])
        first = search.run(coarse_shape=(7,5),starts=2)
        final = search.refine(first)
        best = final['best']
        recovered = EccentricBinaryParams(**best['source'])
        explicit = metric.compare(native_response(recovered))
        basis = CircularAmplitudeBasis.generate(search.source_at(best['coordinates']),response)
        global_fit = maximize_circular_orientation(metric,basis)
        # Off-maximum and restricted candidates also check inner profiling.
        checks = []
        for point in (baseline['coordinates'],[-1.75,2.]):
            cached = search.evaluate(point)
            b = CircularAmplitudeBasis.generate(search.source_at(point),response)
            full = maximize_circular_orientation(metric,b)
            checks.append({'coordinates':point,'fast_overlap':cached['profile']['overlap'],
                'global_overlap':full['overlap'],'difference':cached['profile']['overlap']-full['overlap']})
        improvement = best['profile']['overlap']-first['best']['profile']['overlap']
        movements = np.asarray(best['coordinates'])-np.asarray(first['best']['coordinates'])
        refinement_neighbors = []
        for axis,step in ((0,.001),(1,.01)):
            for sign in (-1.,1.):
                point = np.asarray(best['coordinates'])+np.eye(2)[axis]*step*sign
                refinement_neighbors.append(search.evaluate(point)['profile']['overlap'])
        if abs(explicit['overlap']-best['profile']['overlap'])>1e-9 or abs(global_fit['overlap']-best['profile']['overlap'])>1e-8:
            failures.append(f'{name}: explicit/global orientation check disagrees with fast profile')
        if any(abs(check['difference'])>1e-8 for check in checks):
            failures.append(f'{name}: off-maximum global orientation check failed')
        if best['profile']['overlap']<baseline['overlap']-1e-10 or improvement>1e-7:
            failures.append(f'{name}: restricted-candidate retention or tighter-fit stability failed')
        if max(refinement_neighbors)>best['profile']['overlap']+1e-8:
            failures.append(f'{name}: local intrinsic variation improves claimed maximum')
        if final['minimum_scaled_distance_to_boundary']<.01:
            failures.append(f'{name}: maximum reached the declared intrinsic boundary')
        recovery_error = None
        if e==0:
            recovery_error = np.asarray(best['coordinates'])-search.coordinates_of(truth)
            if abs(recovery_error[0])>1e-3 or abs(recovery_error[1])>3e-3 or explicit['overlap']<1.-1e-8 or explicit['fitted_relative_residual']>2e-4:
                failures.append(f'{name}: frequency/chirp circular injection recovery failed')
        entry = {'case':name,'injection':asdict(truth),'evolution':evolution,'injection_amplitude_scale':scale,
            'samples_before_crop':args.samples,'dt_s':args.dt,'halo_samples_each_side':halo,
            'native_vs_cached_weighted_relative_error':native_error,'restricted_nine_point_comparison':restricted,
            'restricted_best_overlap':baseline['overlap'],'initial_search':first,'refined_search':final,
            'explicit_native_recovered_metrics':explicit,'global_orientation_at_best':global_fit,
            'global_orientation_off_maximum_checks':checks,'scaled_recovery_error':None if recovery_error is None else recovery_error.tolist(),
            'refinement_overlap_change':improvement,'refinement_coordinate_change':movements.tolist(),
            'nearby_intrinsic_profile_scores':refinement_neighbors,'wall_s':time.perf_counter()-start}
        results.append(entry)
        write_json(out/'frequency_chirp_validation.json',results)
        print(f"  finished: restricted={baseline['overlap']:.9g}, continuous={best['profile']['overlap']:.12g}, {search.result()['unique_intrinsic_evaluations']} evaluations",flush=True)
    summary = {'status':'passed_staged_validation' if not failures else 'discrepancy_requires_investigation',
        'failures':failures,'cases':len(results),'noise_model':'secondary_plus_laser',
        'max_native_vs_cached_weighted_relative_error':max(r['native_vs_cached_weighted_relative_error'] for r in results),
        'max_refinement_overlap_change':max(r['refinement_overlap_change'] for r in results),
        'limitation':'Bounded frequency/chirp and physical orientation fitting at fixed sky/masses; short controlled injections, finite diagonal PSD; full FF and production convergence pending.'}
    write_json(out/'validation_summary.json',summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--psd-run',type=Path,default=ROOT/'outputs/referee/task_2_3_reference_integrated_20261007')
    parser.add_argument('--samples',type=int,default=16384)
    parser.add_argument('--dt',type=float,default=5.)
    parser.add_argument('--cases',nargs='+',choices=('circular_positive_offset','circular_negative_offset','eccentric_fixed','eccentric_pm'))
    args = parser.parse_args()
    out,args.psd_run = args.output.resolve(),args.psd_run.resolve()
    if not out.is_relative_to(ROOT) or not args.psd_run.is_relative_to(ROOT) or args.dt<=0:
        raise ValueError('Local output/PSD paths and positive cadence required')
    out.mkdir(parents=True,exist_ok=False)
    assert_local()
    files = list((ROOT/'src/egb_multi').glob('*.py')) + [Path(__file__).resolve(),
        ROOT/'benchmarks/referee_task_2_objective_orientation.py',ROOT/'pyproject.toml',
        args.psd_run/'tdi2_psd.csv',args.psd_run/'psd_metadata.json']
    files += [ROOT/'tests'/name for name in ('test_circular_fit.py','test_circular_search.py','test_tdi.py','test_comparison.py')]
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
            'profile_global_score_difference':1e-8,'refinement_score_change':1e-7,
            'circular_overlap_deficit':1e-8,'circular_fitted_residual':2e-4,
            'frequency_recovery_scaled':1e-3,'chirp_recovery_scaled':3e-3}})
    result = validate(out,args)
    assert_local()
    print(json.dumps(result,indent=2),flush=True)
    if result['failures']:
        raise SystemExit(1)
