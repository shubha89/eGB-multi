"""Paired laser residual controls at four annual epochs and a fixed cutoff.

Finite simulations are a residual diagnostic, not population PSD or annual
finite-window covariance certification. Secondary draws and MPRS are paired.
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
from referee_laser_on_off import pair
from referee_task_2_objective_orientation import make_common_orbit_file,write_json,assert_local


def run(args):
    out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);assert_local();hashes={}
    paths=[*list((ROOT/'src/egb_multi').glob('*.py')),Path(__file__),
        ROOT/'benchmarks/referee_laser_on_off.py',ROOT/'benchmarks/referee_task_2_objective_orientation.py']
    for path in paths:
        rel=path.relative_to(ROOT);dest=out/'snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    year=31557600.;cutoff=args.cutoff
    make_common_orbit_file(out/'native_annual_orbits.h5',year,model='retarded')
    write_json(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'sha256':hashes,'samples':args.samples,'dt_s':5.,'nperseg':65536,
        'cutoff_hz':cutoff,'epoch_fractions':args.epochs,'seeds':args.seeds,
        'maximum_increment_ratio_allowance':args.budget,
        'scope':'Paired finite laser residual diagnostic at annual epochs, not annual covariance'})
    records=[];failures=[]
    for fraction in args.epochs:
        for seed in args.seeds:
            label=f'epoch{fraction:g}_secondary'
            comparison,_=pair(out,out/'native_annual_orbits.h5',model='secondary',
                seed=seed,samples=args.samples,dt=5.,nperseg=65536,label=label,
                t0=year*fraction,noise_cutoff=cutoff)
            values=np.genfromtxt(out/f'{label}_seed{seed}'/'paired_psd.csv',delimiter=',',names=True)
            f=values['frequency_hz'];mask=(f>=2e-5)&(f<=.04)
            ratios={ch:values[f'PSD_{ch}_increment'][mask]/values[f'PSD_{ch}_off'][mask] for ch in ('A','E')}
            maximum=max(float(v.max()) for v in ratios.values())
            records.append({'epoch_fraction':fraction,'seed':seed,'band_hz':[2e-5,.04],
                'maximum_increment_over_secondary':maximum,
                'p95_increment_over_secondary':{ch:float(np.quantile(v,.95)) for ch,v in ratios.items()},
                'pairing':comparison['pairing']})
            write_json(out/'annual_residual_checks.json',records)
            if maximum>args.budget:failures.append(f'epoch {fraction}, seed {seed}: laser increment exceeds allowance')
    write_json(out/'validation_summary.json',{'status':'passed_declared_residual_checks' if not failures else 'discrepancy_requires_investigation',
        'failures':failures,'production_certified':False,'pairs':len(records),
        'limitations':['Finite paired periodograms do not bound population residual without statistical calibration',
            '5 s unfiltered synthetic noise sampling; filter/cadence refinements remain separate',
            'Four annual epochs do not constitute an annual realization']})
    if failures:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=262144)
    p.add_argument('--epochs',type=float,nargs='+',default=[0.,.25,.5,.75])
    p.add_argument('--seeds',type=int,nargs='+',default=[20261008])
    p.add_argument('--cutoff',type=float,default=1/31557600.)
    p.add_argument('--budget',type=float,default=1e-4);run(p.parse_args())
