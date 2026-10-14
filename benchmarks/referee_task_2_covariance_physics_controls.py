"""Native discrete-noise cadence, cutoff and annual-quadrature controls.

These validate numerical choices within the diagnostic frozen covariance.
They do not certify neglected moving-arm residuals or a finite-window model.
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
from egb_multi.external import default_lisaorbits,link_delays_from_lisaorbits
from referee_task_2_covariance_grid import chunked_model
from referee_task_2_frozen_noise_audit import instrument
from referee_task_2_moving_noise_symbol import relative_covariance_error
from referee_task_2_objective_orientation import write_json,assert_local


def run(args):
    out=args.output.resolve();reference=args.reference.resolve()
    if not out.is_relative_to(ROOT) or not reference.is_relative_to(ROOT):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);assert_local();hashes={}
    paths=[*list((ROOT/'src/egb_multi').glob('*.py')),Path(__file__),
        ROOT/'benchmarks/referee_task_2_covariance_grid.py',
        ROOT/'benchmarks/referee_task_2_frozen_noise_audit.py',
        ROOT/'benchmarks/referee_task_2_moving_noise_symbol.py',
        ROOT/'benchmarks/referee_task_2_objective_orientation.py',
        reference/'direct_reference.npz',reference/'manifest.json']
    for path in paths:
        rel=path.relative_to(ROOT);dest=out/'snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes,
        'reference':str(reference.relative_to(ROOT)),'relative_covariance_allowance':args.budget,
        'reference_noise_dt_s':args.reference_noise_dt,'refined_noise_dt_s':args.refined_noise_dt,
        'scope':'Numerical controls within a frozen covariance, not production certification'})
    data=np.load(reference/'direct_reference.npz');f=data['frequency_hz'];truth=data['covariance']
    orbit=default_lisaorbits('esa-trailing');year=31557600.;records=[];failures=[]
    base_dt=args.reference_noise_dt
    if base_dt!=.25:
        print('Recomputing reference at noise dt',base_dt,flush=True)
        delays=link_delays_from_lisaorbits(orbit,(np.arange(16)+.5)*year/16)
        truth=chunked_model(f,instrument(1/year,base_dt),delays)
    np.savez_compressed(out/'reference_covariance.npz',frequency_hz=f,covariance=truth)
    for name,dt,epochs,cutoff_factor in [('noise_dt_refined',args.refined_noise_dt,16,1.),('annual_quadrature_refined',base_dt,32,1.),
        ('cutoff_halved',base_dt,16,.5),('cutoff_doubled',base_dt,16,2.)]:
        print(name,flush=True)
        delays=link_delays_from_lisaorbits(orbit,(np.arange(epochs)+.5)*year/epochs)
        candidate=chunked_model(f,instrument(cutoff_factor/year,dt),delays)
        error=relative_covariance_error(truth,candidate)
        maximum=float(error.max())
        records.append({'control':name,'noise_dt_s':dt,'epochs':epochs,'cutoff_hz':cutoff_factor/year,
            'tested_annual_bins':len(f),'maximum_relative_covariance_error':maximum,
            'maximum_error_frequency_hz':float(f[np.argmax(error)]),
            'p95_relative_covariance_error':float(np.quantile(error,.95))})
        write_json(out/'physics_controls.json',records);print(records[-1],flush=True)
        if maximum>args.budget:failures.append(f'{name}: exceeds relative covariance allowance')
    write_json(out/'validation_summary.json',{'status':'passed_declared_model_controls' if not failures else 'discrepancy_requires_investigation',
        'failures':failures,'production_certified':False,
        'limitation':'Selected annual bins; finite-window and physical-model errors remain separate'})
    if failures:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=ROOT/'outputs/referee/task_2_8_20261008_covariance_grid')
    p.add_argument('--reference-noise-dt',type=float,default=.25)
    p.add_argument('--refined-noise-dt',type=float,default=.125)
    p.add_argument('--budget',type=float,default=1e-4);run(p.parse_args())
