"""Audit waveform-Jacobian refinement of saved automatic native search results."""

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
import numpy as np
from egb_multi.circular_family_search import CircularFamilySearch,FrequencyBasin
from egb_multi.circular_refinement import refine_with_waveform_jacobian,refine_sky_partners
from egb_multi.comparison import AEInnerProduct
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits,link_delays_from_lisaorbits,state_from_lisaorbits
from referee_task_2_frequency_chirp import load_psd,prepared_response
from referee_task_2_objective_orientation import make_clean_response,write_json,assert_local
from referee_task_2_sky_mass_harmonics import all_sky_source_time_bounds


def run(args):
    out,source=args.output.resolve(),args.source.resolve()
    if not out.is_relative_to(ROOT) or not source.is_relative_to(ROOT): raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False)
    original=source/'optimizer_validation.json'
    input_bytes=original.read_bytes()
    entries=json.loads(input_bytes)
    (out/'input_optimizer_validation.json').write_bytes(input_bytes)
    entries=entries[args.case_start:]
    if args.case_count: entries=entries[:args.case_count]
    configuration=json.loads((source/'manifest.json').read_text())['configuration']
    psd,conv=load_psd(Path(configuration['psd_run']))
    orbit=default_lisaorbits('esa-trailing');times=np.arange(configuration['samples'])*configuration['dt']
    delays=link_delays_from_lisaorbits(orbit,times)
    halo=int(np.ceil(8*max(d.max() for d in delays.values())/configuration['dt']))+4*(conv.measurement_order+1)
    response,geometry=prepared_response(orbit,times,conv,halo)
    native,_=make_clean_response(orbit,times,conv,halo)
    files=list((ROOT/'src/egb_multi').glob('*.py'))+[Path(__file__),ROOT/'benchmarks/referee_task_2_frequency_chirp.py',
        ROOT/'benchmarks/referee_task_2_objective_orientation.py',ROOT/'benchmarks/referee_task_2_sky_mass_harmonics.py',
        Path(configuration['psd_run'])/'tdi2_psd.csv',Path(configuration['psd_run'])/'psd_metadata.json']
    hashes={}
    for path in files:
        rel=path.relative_to(ROOT);target=out/'snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes,
        'input_result':str(original.relative_to(ROOT)),'input_result_sha256':hashlib.sha256(input_bytes).hexdigest(),
        'immutable_input_copy':'input_optimizer_validation.json',
        'configuration':{'iterations':args.iterations,'case_start':args.case_start,'case_count':args.case_count,'sky_partners':args.sky_partners,'native_configuration':configuration},
        'predetermined_gates':{'circular_overlap_deficit':2e-7,'native_profile_difference':1e-9,'candidate_retention':1e-10}})
    records,failures=[],[]
    for old in entries:
        started=time.perf_counter();name=old['case']
        truth,prototype=[EccentricBinaryParams(**old[key]) for key in ('injection','prototype')]
        metric=AEInnerProduct(native(truth,old['evolution']),psd,(.2*prototype.mean_motion/np.pi,10*prototype.mean_motion/np.pi))
        basins=[FrequencyBasin(**b) for b in old['basin_selection']['basins']]
        search=CircularFamilySearch(metric,prototype,response,basins=basins,chirp_scaled_bounds=(-16,16),
            evaluation_time_bounds_s=all_sky_source_time_bounds(geometry))
        search.retain(EccentricBinaryParams(**old['refined_search']['best']['source']))
        additional=[]
        if args.sky_partners:
            epoch=search.epoch_s;step=3600.
            state=state_from_lisaorbits(orbit,np.array([epoch-step,epoch,epoch+step]),frame='ecliptic')
            r=state.barycenter[1];v=(state.barycenter[2]-state.barycenter[0])/(2*step)
            acceleration=(state.barycenter[2]-2*r+state.barycenter[0])/step**2
            plane=state.positions[1];normal=np.cross(plane[1]-plane[0],plane[2]-plane[0])
            additional=refine_sky_partners(search,r,v,acceleration,normal)
        result=refine_with_waveform_jacobian(search,iterations=args.iterations)
        explicit=metric.compare(native(EccentricBinaryParams(**result['best']['source'])))
        if result['best']['profile']['overlap']<old['refined_search']['best']['profile']['overlap']-1e-10: failures.append(f'{name}: candidate retention')
        if abs(explicit['overlap']-result['best']['profile']['overlap'])>1e-9: failures.append(f'{name}: native/profile difference')
        if truth.eccentricity==0 and explicit['overlap']<1-2e-7: failures.append(f'{name}: circular recovery')
        records.append({'case':name,'injection':asdict(truth),'initial_overlap':old['refined_search']['best']['profile']['overlap'],
            'additional_sky_starts':additional,'refinement':result,'explicit_native_metrics':explicit,'wall_s':time.perf_counter()-started})
        write_json(out/'jacobian_refinement.json',records)
        print(f"{name}: {records[-1]['initial_overlap']:.12g} -> {result['best']['profile']['overlap']:.12g}",flush=True)
    summary={'status':'passed_declared_refinement_checks' if not failures else 'discrepancy_requires_investigation',
        'failures':failures,'cases':len(records),'production_certified':False}
    write_json(out/'validation_summary.json',summary);assert_local();print(json.dumps(summary,indent=2),flush=True)
    if failures: raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case-count',type=int,default=0)
    p.add_argument('--case-start',type=int,default=0)
    p.add_argument('--iterations',type=int,default=20);p.add_argument('--sky-partners',action='store_true');run(p.parse_args())
