"""Quantify cutoff sensitivity before adopting a production regularization.

The candidate pool is held fixed, physical orientation is reprofiled, and full
annual frozen covariance is used. This is not a complete intrinsic search.
"""

import argparse
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/referee/.cache'))
import numpy as np
from egb_multi.comparison import AEInnerProduct,AEPSD
from egb_multi.circular_fit import CircularAmplitudeBasis,profile_circular_orientation
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits,link_delays_from_lisaorbits
from referee_task_2_covariance_grid import chunked_model
from referee_task_2_frozen_noise_audit import instrument
from referee_task_2_frequency_chirp import load_psd,prepared_response
from referee_task_2_objective_orientation import assert_local,make_clean_response,write_json
from referee_task_2_optimizer_certification import archived_cases


def run(args):
    out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);assert_local();hashes={}
    psd_run=ROOT/'outputs/referee/task_2_3_reference_integrated_20261007'
    cases=archived_cases(['eccentric_fixed','eccentric_pm','eccentric_high_e'])
    paths=[*list((ROOT/'src/egb_multi').glob('*.py')),Path(__file__),
        *[ROOT/f'benchmarks/{name}.py' for name in ('referee_task_2_covariance_grid',
            'referee_task_2_frozen_noise_audit','referee_task_2_frequency_chirp',
            'referee_task_2_objective_orientation','referee_task_2_optimizer_certification')],
        psd_run/'tdi2_psd.csv',psd_run/'psd_metadata.json',*[c[-1] for c in cases]]
    for path in paths:
        rel=path.relative_to(ROOT);dest=out/'snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes,
        'cutoff_days':[15,30,60,365.25],'noise_physics_dt_s':.03125,'annual_epochs':16,
        'covariance_grid_spacing_hz':1/(1048576*5.),'band_hz':[2e-5,.04],
        'common_candidate_score_range_allowance':4e-4,
        'scope':'Cutoff/diagonal sensitivity of a retained pool only; not production certification'})
    orbit=default_lisaorbits('esa-trailing');year=31557600.
    delays=link_delays_from_lisaorbits(orbit,(np.arange(16)+.5)*year/16)
    df=1/(1048576*5.);f=np.arange(1,int(np.ceil(.04/df))+2)*df
    _,conv=load_psd(psd_run);models={}
    for days in (15,30,60,365.25):
        print('Covariance cutoff days',days,flush=True)
        models[days]=chunked_model(f,instrument(1/(days*86400),.03125),delays)
    np.savez_compressed(out/'cutoff_covariance.npz',frequency_hz=f,**{f'days_{d:g}':c for d,c in models.items()})
    times=np.arange(8192)*10.;link_delays=link_delays_from_lisaorbits(orbit,times)
    halo=int(np.ceil(8*max(v.max() for v in link_delays.values())/10.))+4*(conv.measurement_order+1)
    response,_=prepared_response(orbit,times,conv,halo);native,_=make_clean_response(orbit,times,conv,halo)
    records=[];failures=[]
    for name,truth,prototype,evolution,old,path in cases:
        signal=native(truth,evolution)
        candidates=[EccentricBinaryParams(**s['best']['source']) for s in old['expanded_family_search']['basins_results']]
        candidates.append(EccentricBinaryParams(**old['final_refinement']['best']['source']))
        bases=[CircularAmplitudeBasis.generate(s,response) for s in candidates];scores={}
        for days,cov in models.items():
            for full in (False,True):
                psd=AEPSD(f,{'A':cov[:,0,0].real,'E':cov[:,1,1].real},conv,
                    {'sidedness':'one_sided','scaling':'density','noise_model':'diagnostic_native_annual_frozen_secondary',
                        'CSD_convention':'S_AE=E[tilde_A*conj(tilde_E)]','cutoff_days':days,'production_certified':False},
                    cov[:,0,1] if full else None)
                metric=AEInnerProduct(signal,psd,(2e-5,.04))
                profiled=[profile_circular_orientation(metric,b)['overlap'] for b in bases]
                scores[f'{days:g}_'+('full' if full else 'diagonal')]=max(profiled)
        values=[scores[f'{d:g}_full'] for d in (15,30,60)]
        record={'case':name,'candidate_count':len(bases),'scores':scores,
            '15_to_60day_score_range':max(values)-min(values),
            '30day_vs_year_score_difference':scores['30_full']-scores['365.25_full']}
        if record['15_to_60day_score_range']>4e-4:failures.append(f'{name}: cutoff score variation exceeds allowance')
        records.append(record);write_json(out/'cutoff_weighting.json',records);print(record,flush=True)
    write_json(out/'validation_summary.json',{'status':'passed_declared_pool_controls' if not failures else 'discrepancy_requires_investigation',
        'failures':failures,'production_certified':False,
        'limitations':['No intrinsic reoptimization','Short clean signals weighted with an annual stationary model',
            'Finite-window, annual optimized FF and physical laser residual remain separate']})
    if failures:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
