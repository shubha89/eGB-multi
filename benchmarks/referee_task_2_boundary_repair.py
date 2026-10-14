"""Replay the saved annual high-PM boundary failure and verify its repair.

Recover the failed starting point and its measured diagonal curvature from
the original evaluation log. No changed physics or relaxed parameter domain
is needed. Also exercise the repaired local optimizer with the actual annual
metric and verify its physical winner using native sampled TDI.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/referee/.cache'))
import numpy as np
from egb_multi.circular_family_search import CircularFamilySearch,FrequencyBasin,_unscale_optimizer_coordinates
from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.circular_spectral import CircularSpectralTDI
from egb_multi.comparison import AESeries,AEPSD,AEInnerProduct,TDIConvention
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits
from egb_multi.native_validation import ChunkedNativeResponse
from referee_task_2_objective_orientation import assert_local,write_json


def run(args):
    assert_local();out=args.output.resolve();prior=args.prior.resolve();inputs=args.inputs.resolve()
    if not all(p.is_relative_to(ROOT) for p in (out,prior,inputs)):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);hashes={}
    for path in [*list((ROOT/'src/egb_multi').glob('*.py')),Path(__file__),ROOT/'benchmarks/referee_task_2_objective_orientation.py']:
        relative=path.relative_to(ROOT);dest=out/'snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    input_files=[prior/'high_pm_evaluations.jsonl',prior/'high_pm_basin_selection.json',prior/'manifest.json',
        prior/'snapshot/src/egb_multi/circular_family_search.py',inputs/'high_pm_source.json',
        inputs/'high_pm_signal.npz',inputs/'annual_covariance.npz']
    for path in input_files:
        target=out/'inputs'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write_json(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sha256=hashes,
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files},
        scope='Specific boundary failure replay and actual annual local repair control',maxiter=args.maxiter,
        native_overlap_difference_allowance=2e-4,production_certified=False))
    entries=[json.loads(s) for s in (prior/'high_pm_evaluations.jsonl').read_text().splitlines()]
    lookup={tuple(row['coordinates']):row['overlap'] for row in entries}
    point=np.array(entries[-1]['coordinates']);point[-1]=0.
    # Last saved block is the lower-v endpoint's two one-sided curvature probes.
    if entries[-1]['coordinates'][-1]!=.04 or entries[-2]['coordinates'][-1]!=.02 or tuple(point) not in lookup:
        raise ValueError('Saved log does not contain the expected complete failed-start curvature probes')
    index=entries[-1]['basin_index'];selection=json.loads((prior/'high_pm_basin_selection.json').read_text())
    basins=[FrequencyBasin(**b) for b in selection['basins']]
    configuration=json.loads((prior/'manifest.json').read_text())['configuration']
    width=basins[index].half_width_cycles;gamma=configuration['chirp']
    bounds=np.array([[-width,width],[-gamma,gamma],[-1.,1.],[point[3]-np.pi,point[3]+np.pi],[0.,1.],[0.,1.]])
    curvature=[]
    for axis,(low,high) in enumerate(bounds):
        step=min(.02,.25*(high-low));offsets=(-step,0.,step)
        if point[axis]-step<low:offsets=(0.,step,2*step)
        elif point[axis]+step>high:offsets=(-2*step,-step,0.)
        values=[]
        for offset in offsets:
            trial=point.copy();trial[axis]+=offset
            values.append(-lookup[tuple(trial)])
        curvature.append((values[0]-2*values[1]+values[2])/step**2)
    scales=np.sqrt(np.maximum(abs(np.array(curvature)),1e-6));scaled_bounds=bounds*scales[:,None]
    raw=(point*scales)/scales
    repaired,record=_unscale_optimizer_coordinates(point*scales,scales,bounds,scaled_bounds)
    outside=(raw<bounds[:,0])|(raw>bounds[:,1])
    np.testing.assert_array_equal(repaired,point)
    if np.flatnonzero(outside).tolist()!=[1]:raise ValueError('Original one-ulp chirp failure was not reproduced')
    replay=dict(basin_index=index,original_start=point.tolist(),curvature=curvature,scales=scales.tolist(),
        original_roundtrip=raw.tolist(),original_chirp_hex=float(raw[1]).hex(),legal_chirp_hex=float(point[1]).hex(),
        original_roundtrip_drift=(raw-point).tolist(),outside_axes=np.flatnonzero(outside).tolist(),
        corrected_start=repaired.tolist(),correction_record=record,
        source_phase_change_scale_rad=float(np.pi*abs(raw[1]-point[1])/4),
        known_original_start_overlap=lookup[tuple(point)],original_failure_reproduced=True)
    write_json(out/'boundary_replay.json',replay);print(replay,flush=True)
    meta=json.loads((inputs/'high_pm_source.json').read_text());convention=TDIConvention(**meta['convention'])
    times=np.arange(round(meta['duration_s']/meta['dt_s']))*meta['dt_s']
    with np.load(inputs/'high_pm_signal.npz') as data:signal=AESeries(times,{ch:data[ch] for ch in ('A','E')},convention)
    with np.load(inputs/'annual_covariance.npz') as data:f=data['frequency_hz'];cov=data['covariance']
    psd=AEPSD(f,{ch:cov[:,i,i].real for i,ch in enumerate(('A','E'))},convention,
        dict(sidedness='one_sided',scaling='density',noise_model='native_annual_secondary_diagnostic',
             CSD_convention='S_AE=E[tilde_A*conj(tilde_E)]'),cov[:,0,1])
    metric=AEInnerProduct(signal,psd,tuple(configuration['band']));orbit=default_lisaorbits('esa-trailing')
    prototype=EccentricBinaryParams(mean_motion=meta['source']['mean_motion'],eccentricity=0.,t0=(times[0]+times[-1])/2,
        beta=-.6,lambda_=.8,m1_solar=.5,m2_solar=.5,inclination=.6,phi0=0.,psi=.2)
    response=InterpolatedCircularTDI(orbit,times,convention,envelope_intervals=configuration['envelope_intervals'])
    spectral=CircularSpectralTDI(response,metric)
    def progress(entry):
        with (out/'local_evaluations.jsonl').open('a') as stream:stream.write(json.dumps(entry)+'\n')
        if entry['unique_intrinsic_evaluation']%20==0:print('actual annual repair evaluation',entry['unique_intrinsic_evaluation'],flush=True)
    search=CircularFamilySearch(metric,prototype,spectral,basins=basins,chirp_scaled_bounds=(-gamma,gamma),
        evaluation_time_bounds_s=(times[0]-2180.,times[-1]+2000.),progress=progress)
    initial=search.evaluate(repaired,index)
    initial_delta=initial['profile']['overlap']-lookup[tuple(point)]
    if abs(initial_delta)>1e-8:raise ValueError('Repaired start changed the annual physical objective')
    fit,endpoint,conditioning=search._local_fit(repaired,index,ftol=configuration['ftol'],maxiter=args.maxiter,
        finite_difference_step=configuration['fd_step'])
    best=search.result()['best'];source=EccentricBinaryParams(**best['source'])
    native=ChunkedNativeResponse(orbit,times,convention,block_samples=65536,cached_blocks=2)
    explicit=metric.compare(native(source));delta=explicit['overlap']-best['profile']['overlap']
    write_json(out/'actual_annual_repair.json',dict(initial_overlap=initial['profile']['overlap'],initial_minus_saved=initial_delta,
        local_endpoint=endpoint.tolist(),local_termination=dict(success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit)),
        conditioning=conditioning,best=best,native_metrics=explicit,native_minus_spectral=delta,
        scope='Short local control from the previously failing start; full baseline is a separate continuation'))
    if abs(delta)>2e-4:raise ValueError('Native winner disagreement in the repaired local control')
    write_json(out/'validation_summary.json',dict(status='passed_boundary_repair_diagnostic',production_certified=False,
        original_failure_reproduced=True,original_failure_chirp_roundtrip_drift=float(raw[1]-point[1]),
        repaired_start_exactly_preserved=True,actual_annual_objective_minus_saved=initial_delta,
        repaired_local_optimizer_executed=True,native_minus_spectral=delta,
        limitation='Local numerical repair validated; does not complete the high-PM annual search or Task 2.8'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prior',type=Path,default=ROOT/'outputs/referee/task_2_8_20261008_annual_search_high_pm_spectral')
    p.add_argument('--inputs',type=Path,default=ROOT/'outputs/referee/task_2_8_20261008_annual_weighting_pilot')
    p.add_argument('--maxiter',type=int,default=5)
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        if args.output.exists():write_json(args.output/'validation_summary.json',dict(status='failed_boundary_repair_diagnostic',
            production_certified=False,exception_type=type(exc).__name__,exception=str(exc)))
        raise
