"""Native response sensitivity to tightened PM integration over one year."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/referee/.cache'))
import numpy as np
from egb_multi.comparison import AEInnerProduct
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits
from egb_multi.native_validation import ChunkedNativeResponse,weighted_relative_residual,normalized_overlap_change_bound
from referee_task_2_frequency_chirp import load_psd
from referee_task_2_objective_orientation import assert_local,write_json


def run(args):
    out=args.output.resolve()
    if not out.is_relative_to(ROOT): raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);assert_local()
    hashes={}
    for path in [*list((ROOT/'src/egb_multi').glob('*.py')),Path(__file__),args.psd_run/'tdi2_psd.csv',args.psd_run/'psd_metadata.json']:
        rel=path.relative_to(ROOT);target=out/'snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/'manifest.json',{'sha256':hashes,'duration_s':args.duration,'dt_s':10.,
        'default_tolerances':{'rtol':1e-10,'phase_atol':1e-8},'refined_tolerances':{'rtol':2e-13,'phase_atol':1e-10},
        'source_overlap_budget':args.budget,'scope':'PM ODE numerical precision, not fixed-vs-evolving model choice'})
    psd,conv=load_psd(args.psd_run);orbit=default_lisaorbits('esa-trailing')
    times=np.arange(int(round(args.duration/10)))*10.
    engine=ChunkedNativeResponse(orbit,times,conv)
    definitions={'low':(.0001,.1,.5,.5,.2,.2,-1.),'high':(.003,.5,.8,.6,.8,.2,-1.),'high_mass':(.003,.5,1.,.3,1.2,-.7,2.4)}
    records=[];failures=[]
    for name in args.cases:
        frequency,e,m1,m2,inc,beta,longitude=definitions[name]
        source=EccentricBinaryParams(mean_motion=np.pi*frequency,eccentricity=e,m1_solar=m1,m2_solar=m2,
            inclination=inc,beta=beta,lambda_=longitude,phi0=.2,psi=.4,t0=0.)
        print(name,'default response',flush=True);normal=engine(source,'peters_mathews')
        print(name,'tight response',flush=True);tight=engine(source,'peters_mathews',pm_ode_rtol=2e-13,pm_ode_phase_atol=1e-10)
        metric=AEInnerProduct(tight,psd,(.2*frequency,10*frequency))
        error=weighted_relative_residual(metric,tight,normal);bound=normalized_overlap_change_bound(error,0.)
        records.append({'case':name,'source':asdict(source),'weighted_relative_error':error,'overlap_change_bound':bound})
        if bound>args.budget: failures.append(f'{name}: source ODE error exceeds budget')
        write_json(out/'evolution_precision.json',records);print(name,'bound',bound,flush=True)
    write_json(out/'validation_summary.json',{'status':'passed_declared_source_precision_checks' if not failures else 'discrepancy_requires_investigation',
        'failures':failures,'production_duration_tested':args.duration>=31557600,'production_certified':False,
        'limitation':'Representative injection error, not uniform template-family or optimizer certification'})
    if failures: raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--psd-run',type=Path,default=ROOT/'outputs/referee/task_2_3_reference_integrated_20261007')
    p.add_argument('--cases',nargs='+',choices=['low','high','high_mass'],default=['low','high','high_mass'])
    p.add_argument('--duration',type=float,default=31557600.);p.add_argument('--budget',type=float,default=1e-4)
    args=p.parse_args();args.psd_run=args.psd_run.resolve();run(args)
