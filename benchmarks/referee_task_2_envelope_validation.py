"""Native annual error checks for the optional circular envelope evaluator."""

import argparse
from dataclasses import asdict,replace
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/referee/.cache'))
import numpy as np
from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.native_validation import ChunkedNativeResponse,weighted_relative_residual,normalized_overlap_change_bound
from egb_multi.comparison import AEInnerProduct
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits
from referee_task_2_frequency_chirp import load_psd
from referee_task_2_objective_orientation import write_json,assert_local


def run(args):
    out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);assert_local();hashes={}
    for path in [*list((ROOT/'src/egb_multi').glob('*.py')),Path(__file__),
                 ROOT/'benchmarks/referee_task_2_frequency_chirp.py',
                 ROOT/'benchmarks/referee_task_2_objective_orientation.py',
                 args.psd_run/'tdi2_psd.csv',args.psd_run/'psd_metadata.json']:
        rel=path.relative_to(ROOT);dest=out/'snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/'manifest.json',{'sha256':hashes,'duration_s':args.duration,'dt_s':args.dt,
        'envelope_intervals':args.intervals,'pointwise_overlap_allowance':args.budget,
        'measurement_order_override':args.tdi_order,'case_selection':args.case,
        'scope':'circular response numerical error, not optimizer or PSD certification'})
    psd,conv=load_psd(args.psd_run)
    if args.tdi_order:
        conv=replace(conv,measurement_order=args.tdi_order)
        psd=replace(psd,convention=conv,provenance=dict(psd.provenance,
            numerical_convention_override='Response interpolation convergence only; physical noise model unchanged'))
    orbit=default_lisaorbits('esa-trailing')
    times=np.arange(int(round(args.duration/args.dt)))*args.dt
    native=ChunkedNativeResponse(orbit,times,conv)
    approximate=[InterpolatedCircularTDI(orbit,times,conv,envelope_intervals=n) for n in args.intervals]
    definitions=[('low',.0001,.5,.5,.713,-.517),('carrier',.003,1.4,.1,.713,16.),('near_TDI_null',.03,1.4,1.4,.713,-16.)]
    if args.case:definitions=[d for d in definitions if d[0] in args.case]
    records=[];failures=[]
    for name,f,m1,m2,alpha,gamma in definitions:
        source=EccentricBinaryParams(mean_motion=np.pi*(f+alpha/args.duration),eccentricity=0,
            m1_solar=m1,m2_solar=m2,t0=(times[0]+times[-1])/2,fdot=gamma/args.duration**2,
            beta=.43,lambda_=6.08,inclination=.91,phi0=.77,psi=.46)
        print(name,'native reference',flush=True);start=time.perf_counter();truth=native(source);native_time=time.perf_counter()-start
        metric=AEInnerProduct(truth,psd,(2e-5,.04))
        comparisons=[]
        for n,model in zip(args.intervals,approximate):
            start=time.perf_counter();candidate=model(source);elapsed=time.perf_counter()-start
            error=weighted_relative_residual(metric,truth,candidate);bound=normalized_overlap_change_bound(0.,error)
            comparisons.append({'envelope_intervals':n,'weighted_relative_error':error,'pointwise_overlap_change_bound':bound,
                'evaluation_wall_s':elapsed,'approximation':'continuous nested delay polynomials plus cubic demodulated-envelope interpolation'})
            print(name,n,'bound',bound,'seconds',elapsed,flush=True)
        if comparisons[-1]['pointwise_overlap_change_bound']>args.budget:failures.append(f'{name}: native/envelope difference exceeds allowance')
        records.append({'case':name,'source':asdict(source),'native_wall_s':native_time,'comparisons':comparisons})
        write_json(out/'envelope_validation.json',records)
    write_json(out/'validation_summary.json',{'status':'passed_declared_envelope_checks' if not failures else 'discrepancy_requires_investigation',
        'failures':failures,'production_certified':False,'duration_s':args.duration,
        'limitation':'Pointwise selected circular templates; uniform family accuracy and optimized FF stability remain required'})
    if failures:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--psd-run',type=Path,default=ROOT/'outputs/referee/task_2_3_reference_integrated_20261007')
    p.add_argument('--duration',type=float,default=31557600.);p.add_argument('--dt',type=float,default=10.)
    p.add_argument('--intervals',type=int,nargs='+',default=[4096,8192,16384]);p.add_argument('--budget',type=float,default=2e-4)
    p.add_argument('--tdi-order',type=int,default=0)
    p.add_argument('--case',nargs='+',choices=['low','carrier','near_TDI_null'])
    run(p.parse_args())
