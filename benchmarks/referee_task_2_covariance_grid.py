"""Audit deterministic covariance interpolation on annual Fourier frequencies.

This checks numerical interpolation only. It does not adopt this covariance
model or certify its physical approximations or finite-window likelihood.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/referee/.cache'))

import numpy as np
from egb_multi.external import default_lisaorbits,link_delays_from_lisaorbits
from egb_multi.noise_covariance import frozen_native_secondary_covariance
from referee_task_2_frozen_noise_audit import instrument
from referee_task_2_moving_noise_symbol import relative_covariance_error
from referee_task_2_objective_orientation import write_json,assert_local


def chunked_model(f,definitions,delays):
    answer=np.empty((len(f),2,2),complex)
    epoch_count=len(next(iter(delays.values())))
    for start in range(0,len(f),8192):
        stop=min(len(f),start+8192);part=np.zeros((stop-start,2,2),complex)
        for epoch in range(epoch_count):
            part+=frozen_native_secondary_covariance(f[start:stop],
                {label:float(v[epoch]) for label,v in delays.items()},definitions)[0]/epoch_count
        answer[start:stop]=part
    return answer


def run(args):
    out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);assert_local();hashes={}
    paths=[*list((ROOT/'src/egb_multi').glob('*.py')),Path(__file__),
        ROOT/'benchmarks/referee_task_2_frozen_noise_audit.py',
        ROOT/'benchmarks/referee_task_2_moving_noise_symbol.py',
        ROOT/'benchmarks/referee_task_2_objective_orientation.py']
    for path in paths:
        rel=path.relative_to(ROOT);target=out/'snapshot'/rel
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    year=31557600.;epochs=(np.arange(args.epochs)+.5)*year/args.epochs
    delays=link_delays_from_lisaorbits(default_lisaorbits('esa-trailing'),epochs)
    definitions=instrument(1/year,.25)
    write_json(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'sha256':hashes,'annual_epochs':args.epochs,'noise_physics_dt_s':.25,
        'regularization_hz':1/year,'grid_segment_sizes':args.sizes,
        'maximum_relative_covariance_allowance':args.budget,
        'scope':'Numerical covariance-grid check only; not production certification'})

    def model(f):
        return chunked_model(f,definitions,delays)

    # Annual bins throughout the useful band, with every bin in a narrow
    # interval around the first TDI2 zero. No zero is removed or floored.
    first,last=int(np.ceil(2e-5*year)),int(np.floor(.04*year))
    uniform=np.arange(first,last+1,23)
    dense=np.arange(int(np.ceil(.029*year)),int(np.floor(.031*year))+1)
    indices=np.unique(np.r_[uniform,dense]);f=indices/year
    print('Direct covariance reference',len(f),'annual bins',flush=True)
    direct=model(f);records=[]
    for size in args.sizes:
        df=1/(size*5.)
        grid=np.arange(1,int(np.ceil(.04/df))+2)*df
        print('Covariance grid',size,len(grid),'nodes',flush=True)
        cov=model(grid);candidate=np.empty_like(direct)
        for a in range(2):
            for b in range(2):
                candidate[:,a,b]=np.interp(f,grid,cov[:,a,b].real)+1j*np.interp(f,grid,cov[:,a,b].imag)
        error=relative_covariance_error(direct,candidate)
        rows=[]
        for low,high in ((2e-5,.001),(.0002,.01),(.0006,.03),(.0003,.036),(2e-5,.04)):
            mask=(f>=low)&(f<=high)
            rows.append({'band_hz':[low,high],'tested_annual_bins':int(mask.sum()),
                'maximum_relative_covariance_error':float(error[mask].max()),
                'maximum_error_frequency_hz':float(f[mask][np.argmax(error[mask])]),
                'p95_relative_covariance_error':float(np.quantile(error[mask],.95))})
        records.append({'grid_segment_size':size,'grid_spacing_hz':df,'bands':rows})
        write_json(out/'grid_comparison.json',records)
        np.savez_compressed(out/f'covariance_grid_{size}.npz',frequency_hz=grid,covariance=cov)
        print(size,rows[-1],flush=True)
    maximum=records[-1]['bands'][-1]['maximum_relative_covariance_error']
    failures=[] if maximum<=args.budget else ['Finest covariance grid exceeds relative covariance allowance']
    np.savez_compressed(out/'direct_reference.npz',frequency_hz=f,covariance=direct)
    write_json(out/'validation_summary.json',{'status':'passed_declared_grid_checks' if not failures else 'discrepancy_requires_investigation',
        'failures':failures,'production_certified':False,
        'limitation':'Selected annual bins; finite-window and physical-model errors are separate'})
    if failures:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=16)
    p.add_argument('--sizes',type=int,nargs='+',default=[65536,262144,1048576])
    p.add_argument('--budget',type=float,default=1e-4)
    run(p.parse_args())
