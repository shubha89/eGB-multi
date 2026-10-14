"""Controlled annual seed-resolution diagnosis, with unchanged physical fits.

Compare sky and chirp proposals at one common frequency basin, observation,
band, covariance and zero-phase epoch. Proposal scores remain approximate;
each selected start is evaluated and optimized with all six intrinsic
coordinates free. A frozen scalar implementation separately checks the
batched numerical calculation on identical physical inputs.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.circular_family_search import CircularFamilySearch, FrequencyBasin, equal_area_sky
from egb_multi.circular_seeds import rank_envelope_frequency_seeds
from egb_multi.circular_spectral import CircularSpectralTDI
from egb_multi.comparison import AESeries, AEPSD, AEInnerProduct, TDIConvention
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits
from referee_task_2_annual_refinement import validate_spectral_control
from referee_task_2_objective_orientation import assert_local, write_json


def run(args):
    assert_local()
    paths = [p.resolve() for p in (args.output, args.inputs, args.source_run, args.scalar_source,
                                   args.spectral_validation)]
    if not all(p.is_relative_to(ROOT) for p in paths):
        raise ValueError('Keep diagnosis in this checkout')
    out, inputs, source_run, scalar_source, control = paths
    validate_spectral_control(control)
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), *list((ROOT/'src/egb_multi').glob('*.py'))]
    hashes = {}
    for path in files:
        target = out/'snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    input_files = [scalar_source, inputs/(args.case+'_source.json'), inputs/(args.case+'_signal.npz'),
                   inputs/'annual_covariance.npz', source_run/(args.case+'_annual_search.json')]
    write_json(out/'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), sha256=hashes,
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files},
        scope='Controlled seed comparison; not a production convergence certificate'))
    old = json.loads(input_files[-1].read_text())
    meta = json.loads((inputs/(args.case+'_source.json')).read_text())
    duration, dt = meta['duration_s'], meta['dt_s']
    times = np.arange(round(duration/dt))*dt
    convention = TDIConvention(**meta['convention'])
    with np.load(inputs/(args.case+'_signal.npz')) as data:
        signal = AESeries(times, {c:data[c] for c in ('A','E')}, convention)
    with np.load(inputs/'annual_covariance.npz') as data:
        frequency, covariance = data['frequency_hz'], data['covariance']
    psd = AEPSD(frequency, {c:covariance[:,i,i].real for i,c in enumerate(('A','E'))}, convention,
        dict(sidedness='one_sided', scaling='density', noise_model='native_secondary_diagnostic',
             CSD_convention='S_AE=E[tilde_A*conj(tilde_E)]'), covariance[:,0,1])
    metric = AEInnerProduct(signal, psd, (2e-5,.04))
    prototype = replace(EccentricBinaryParams(**meta['source']), eccentricity=0., phi0=0., psi=0.,
        inclination=1., beta=-.6, lambda_=.8, m1_solar=.5, m2_solar=.4, fdot=0., t0=.5*times[-1])
    response = InterpolatedCircularTDI(default_lisaorbits('esa-trailing'), times, convention,
                                       envelope_intervals=8192)
    spectral = CircularSpectralTDI(response, metric)
    basin = FrequencyBasin(**old['selection']['basins'][args.basin])
    basin = replace(basin, half_width_cycles=2*basin.half_width_cycles)
    spec = importlib.util.spec_from_file_location('frozen_scalar_seeds', scalar_source)
    scalar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scalar)
    common = dict(sky=equal_area_sky(24)[:4], chirp_scaled_values=[0.], seed_samples=2048,
                  window_half_width_cycles=256)
    start = time.perf_counter()
    slow = scalar.rank_envelope_frequency_seeds(metric,response,prototype,basin,**common)
    scalar_s = time.perf_counter()-start
    start = time.perf_counter()
    fast = rank_envelope_frequency_seeds(metric,response,prototype,basin,**common)
    batch_s = time.perf_counter()-start
    key = lambda r:(r['sine_latitude'],r['longitude_rad'],r['chirp_scaled'])
    old_scores = {key(r):r for r in slow}
    differences = [abs(r['seed_score']-old_scores[key(r)]['seed_score']) for r in fast]
    frequency_equal = all(r['frequency_hz']==old_scores[key(r)]['frequency_hz'] for r in fast)
    write_json(out/'matched_scalar_batch_control.json', dict(scalar_wall_s=scalar_s,batch_wall_s=batch_s,
        maximum_seed_score_difference=max(differences),selected_frequency_identical=frequency_equal,
        same_inputs_and_outputs=True, proposal_only=True))
    if max(differences)>1e-12 or not frequency_equal:
        raise ValueError('Batched proposal calculation differs from the frozen scalar control')
    rows=[]
    for sky_points,chirp_points in ((24,5),(96,5),(24,25),(96,25)):
        start=time.perf_counter()
        seeds=rank_envelope_frequency_seeds(metric,response,prototype,basin,
            sky=equal_area_sky(sky_points),chirp_scaled_values=np.linspace(-48.,48.,chirp_points))
        label=f'sky{sky_points}_chirp{chirp_points}'
        write_json(out/(label+'_proposals.json'),[dict(r,source=asdict(r['source'])) for r in seeds])
        proposal_wall=time.perf_counter()-start
        search=CircularFamilySearch(metric,prototype,spectral,basins=[basin],
            chirp_scaled_bounds=(-48.,48.),evaluation_time_bounds_s=(-2180.,duration+2000.))
        exact=[]
        for seed in seeds[:32]:
            point=search.coordinates_of(seed['source'],0)
            exact.append((search.evaluate(point,0)['profile']['overlap'],point))
        selected=[];fits=[]
        for score,point in sorted(exact,key=lambda r:r[0],reverse=True):
            direction=np.array([np.sqrt(1-point[2]**2)*np.cos(point[3]),
                np.sqrt(1-point[2]**2)*np.sin(point[3]),point[2]])
            if selected and any(np.linalg.norm(direction-p[1])<=.05 for p in selected):continue
            selected.append((point,direction))
            fit,endpoint,conditioning=search._local_fit(point,0,ftol=1e-9,maxiter=60,finite_difference_step=1e-6)
            item=search.evaluate(endpoint,0)
            fits.append(dict(start=point.tolist(),coordinates=endpoint.tolist(),overlap=item['profile']['overlap'],
                success=bool(fit.success),message=str(fit.message)))
            if len(selected)==2:break
        rows.append(dict(sky_points=sky_points,chirp_points=chirp_points,chirp_range_scaled=[-48.,48.],
            seed_score=seeds[0]['seed_score'],proposal_wall_s=proposal_wall,wall_s=time.perf_counter()-start,
            local_runs=fits,best=search.result()['best'],all_six_intrinsic_coordinates_free=True))
        write_json(out/'resolution_controls.json',rows)
        print(label,'best',rows[-1]['best']['profile']['overlap'],'proposal_s',round(proposal_wall,2),flush=True)
    write_json(out/'validation_summary.json',dict(status='controlled_seed_resolution_diagnosis_completed',
        production_certified=False,scope='Same metric/domain/prototype; two local fits per proposal grid',
        global_maximum_proven=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--scalar-source',type=Path,required=True)
    p.add_argument('--spectral-validation',type=Path,required=True)
    p.add_argument('--case',default='high_pm');p.add_argument('--basin',type=int,default=1)
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        if args.output.exists() and not (args.output/'validation_summary.json').exists():
            write_json(args.output/'validation_summary.json',dict(status='failed_seed_resolution_diagnosis',
                exception=str(exc),production_certified=False))
        raise
