"""Common-candidate band sensitivity of the high-eccentricity audit case.

Reprofiles physical orientation under each common band. No intrinsic search is
redone; this is a diagnostic, not a fitting-factor band-convergence certificate.
"""

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
import numpy as np
from egb_multi.circular_fit import CircularAmplitudeBasis,profile_circular_orientation
from egb_multi.comparison import AEInnerProduct
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits,link_delays_from_lisaorbits
from referee_task_2_frequency_chirp import load_psd,prepared_response
from referee_task_2_objective_orientation import make_clean_response,write_json,assert_local
from referee_task_2_optimizer_certification import archived_cases


def run(args):
    out=args.output.resolve();source=args.source.resolve()
    if not out.is_relative_to(ROOT) or not source.is_relative_to(ROOT):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);assert_local()
    original=source/'eccentric_high_e_evaluations.jsonl'
    data=original.read_bytes();(out/'input_evaluations.jsonl').write_bytes(data)
    rows=[json.loads(line) for line in data.splitlines()]
    _,truth,prototype,evolution,old,path=archived_cases(['eccentric_high_e'])[0]
    psd_run=ROOT/'outputs/referee/task_2_3_reference_integrated_20261007'
    psd,conv=load_psd(psd_run);orbit=default_lisaorbits('esa-trailing');times=np.arange(8192)*10.
    delays=link_delays_from_lisaorbits(orbit,times)
    halo=int(np.ceil(8*max(v.max() for v in delays.values())/10.))+4*(conv.measurement_order+1)
    response,_=prepared_response(orbit,times,conv,halo)
    native,_=make_clean_response(orbit,times,conv,halo);signal=native(truth,evolution)
    candidates=[EccentricBinaryParams(**r['best']['source']) for r in old['expanded_family_search']['basins_results']]
    candidates.append(EccentricBinaryParams(**old['final_refinement']['best']['source']))
    for basin in sorted(set(r['basin_index'] for r in rows)):
        best=max((r for r in rows if r['basin_index']==basin),key=lambda r:r['overlap'])
        candidates.append(EccentricBinaryParams(**best['source']))
    hashes={}
    paths=[*list((ROOT/'src/egb_multi').glob('*.py')),Path(__file__),path,
        ROOT/'benchmarks/referee_task_2_frequency_chirp.py',ROOT/'benchmarks/referee_task_2_objective_orientation.py',
        ROOT/'benchmarks/referee_task_2_optimizer_certification.py',psd_run/'tdi2_psd.csv',psd_run/'psd_metadata.json']
    for file in paths:
        rel=file.relative_to(ROOT);dest=out/'snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
        hashes[str(rel)]=hashlib.sha256(file.read_bytes()).hexdigest()
    write_json(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes,
        'input_evaluations_sha256':hashlib.sha256(data).hexdigest(),'immutable_input_copy':'input_evaluations.jsonl',
        'scope':'Common-candidate orientation reprofiling only; no intrinsic band reoptimization'})
    bases=[CircularAmplitudeBasis.generate(s,response) for s in candidates]
    nominal=prototype.mean_motion/np.pi;records=[]
    for low,high in ((.2,10),(.1,20),(.05,40),(.02,40)):
        band=(low*nominal,high*nominal)
        metric=AEInnerProduct(signal,psd,band)
        scores=[profile_circular_orientation(metric,b)['overlap'] for b in bases]
        records.append({'band_hz':list(band),'candidates':len(bases),
            'best_common_candidate_overlap':max(scores),'winning_candidate_index':int(np.argmax(scores))})
        write_json(out/'band_sensitivity.json',records);print(records[-1],flush=True)
    write_json(out/'diagnostic_summary.json',{'status':'diagnostic_completed','production_certified':False,
        'limitation':'Common candidate pool and legacy provisional PSD; full annual search and final-weighting checks required'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'outputs/referee/task_2_7_20261008_independent_e06_expanded')
    run(p.parse_args())
