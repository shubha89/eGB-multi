"""Compare frozen and native nested-delay secondary covariance symbols.

This quantifies one approximation only; finite-window and laser-locking
residuals remain separate checks before a production covariance is adopted.
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
from lisainstrument import Instrument
from egb_multi.noise_covariance import moving_native_secondary_symbol,frozen_native_secondary_covariance
from egb_multi.external import default_lisaorbits,link_delays_from_lisaorbits
from referee_task_2_objective_orientation import write_json,assert_local


def relative_covariance_error(reference,candidate):
    root=np.linalg.cholesky(reference);inverse=np.linalg.inv(root)
    difference=np.einsum('fab,fbc,fdc->fad',inverse,candidate-reference,inverse.conj())
    return np.max(abs(np.linalg.eigvalsh(difference)),axis=1)


def run(args):
    out=args.output.resolve()
    if not out.is_relative_to(ROOT): raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);assert_local()
    hashes={}
    for path in [*list((ROOT/'src/egb_multi').glob('*.py')),Path(__file__)]:
        rel=path.relative_to(ROOT);target=out/'snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes,
        'geometry_grid':args.geometry_grid,'epochs':args.epochs,'noise_physics_dt_s':args.noise_dt,'cutoff_hz':1/31557600.,
        'maximum_frequency_hz':args.maximum_frequency,'band_hz':args.band,'dense_nulls':args.dense_nulls,
        'scope':'local adiabatic secondary covariance symbol, diagnostic only'})
    year=31557600.;dt=year/args.geometry_grid;times=np.arange(-64,args.geometry_grid+65)*dt
    indices=(np.arange(args.epochs)+.5)*args.geometry_grid/args.epochs+64
    if np.any(indices!=indices.astype(int)): raise ValueError('Epochs must exactly align with geometry grid')
    indices=indices.astype(int);orbit=default_lisaorbits('esa-trailing')
    delays=link_delays_from_lisaorbits(orbit,times)
    f=np.r_[np.geomspace(1e-5,1e-4,64),np.linspace(1.01e-4,args.maximum_frequency,args.frequency_points)]
    if args.dense_nulls:
        for center in (.03,.06,.09):
            if center < args.maximum_frequency:
                f=np.r_[f,np.linspace(center-.001,center+.001,1025)]
        f=np.unique(f)
    instrument=Instrument(size=524288,dt=args.noise_dt,physics_upsampling=1,aafilter=None,
        telemetry_downsampling=524288,noises_f_min_hz=1/year,interpolation=('lagrange',45))
    print('Building native moving symbol',flush=True)
    dynamic=moving_native_secondary_symbol(f,times,delays,instrument,indices)
    frozen=np.array([frozen_native_secondary_covariance(f,{key:float(value[i]) for key,value in delays.items()},instrument)[0] for i in indices])
    mean_dynamic,mean_frozen=dynamic.mean(axis=0),frozen.mean(axis=0)
    error=relative_covariance_error(mean_dynamic,mean_frozen)
    np.savez_compressed(out/'covariance_symbol.npz',frequency_hz=f,native_moving_mean=mean_dynamic,
        frozen_mean=mean_frozen,relative_covariance_error=error)
    records=[]
    for low,high in ((2e-5,.001),(.0002,.01),(.0006,.03),(.0003,.036),tuple(args.band)):
        mask=(f>=low)&(f<=high);values=error[mask]
        records.append({'band_hz':[low,high],'maximum_relative_covariance_error':float(values.max()),
            'p95_relative_covariance_error':float(np.quantile(values,.95)),
            'maximum_error_frequency_hz':float(f[mask][np.argmax(values)])})
    write_json(out/'symbol_comparison.json',{'status':'diagnostic_completed','production_certified':False,'bands':records,
        'limitations':['An adiabatic local covariance symbol is not an exact finite-observation covariance',
            'No native noise-driven laser-locking residual or spectral-window convolution',
            'Geometry/epoch/frequency resolution refinement required separately',
            'Expanded pyTDI delay polynomial; factorized numerical interpolation must also be checked']})
    write_json(out/'validation_summary.json',dict(status='moving_secondary_symbol_diagnostic_completed',
        production_certified=False,maximum_relative_covariance_error=records[-1]['maximum_relative_covariance_error'],
        scope='Selected adiabatic frequencies; no exact moving finite-observation likelihood'))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--geometry-grid',type=int,default=4096);p.add_argument('--epochs',type=int,default=32)
    p.add_argument('--frequency-points',type=int,default=8192)
    p.add_argument('--noise-dt',type=float,default=.25)
    p.add_argument('--maximum-frequency',type=float,default=.05)
    p.add_argument('--band',nargs=2,type=float,default=[2e-5,.04])
    p.add_argument('--dense-nulls',action='store_true')
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        if args.output.exists() and not (args.output/'validation_summary.json').exists():
            write_json(args.output/'validation_summary.json',dict(status='failed_moving_symbol_control',
                production_certified=False,exception_type=type(exc).__name__,exception=str(exc)))
        raise
