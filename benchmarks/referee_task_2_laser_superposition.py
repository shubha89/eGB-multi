"""Distinguish laser-only TDI residual from loss of numerical superposition.

Use the exact seeds, epochs and fixed cutoff from a completed paired run.
Laser noise is still enabled in the reference; this is a diagnostic only.
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
from referee_laser_on_off import generate,spectra
from referee_task_2_objective_orientation import write_json,assert_local


def run(args):
    out=args.output.resolve();parent=args.parent.resolve()
    if not out.is_relative_to(ROOT) or not parent.is_relative_to(ROOT):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);hashes={};assert_local()
    paired=parent/'epoch0_secondary_seed20261008'
    paths=[*list((ROOT/'src/egb_multi').glob('*.py')),Path(__file__),
        ROOT/'benchmarks/referee_laser_on_off.py',ROOT/'benchmarks/referee_task_2_objective_orientation.py',
        paired/'paired_noise.npz',paired/'on_metadata.json']
    for path in paths:
        rel=path.relative_to(ROOT);dest=out/'snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes,
        'scope':'Finite numerical superposition diagnostic; not production certification'})
    only=generate(parent/'native_annual_orbits.h5',model='laser_only',laser_on=True,
        seed=20261008,samples=262144,dt=5.,noise_cutoff=1/31557600.)
    pair=np.load(paired/'paired_noise.npz');times=pair['t']
    first=np.searchsorted(only['t'],times[0]);last=first+len(times)
    np.testing.assert_array_equal(only['t'][first:last],times)
    laser={ch:only['channels'][ch][first:last] for ch in ('A','E')}
    error={ch:pair[f'increment_{ch}_fractional']-laser[ch] for ch in ('A','E')}
    off={ch:pair[f'off_{ch}_fractional'] for ch in ('A','E')}
    f,s_off,_,_=spectra(off,5.,65536)
    _,s_laser,_,_=spectra(laser,5.,65536)
    _,s_error,_,_=spectra(error,5.,65536)
    mask=(f>=2e-5)&(f<=.04)
    record={'status':'diagnostic_completed','production_certified':False,
        'laser_only_over_secondary':{ch:float(np.max(s_laser[ch][mask]/s_off[ch][mask])) for ch in ('A','E')},
        'superposition_error_over_secondary':{ch:float(np.max(s_error[ch][mask]/s_off[ch][mask])) for ch in ('A','E')},
        'limitation':'This identifies numerical/nonlinear sensitivity; instrument-level precision and common raw laser inputs require further audit'}
    np.savez_compressed(out/'superposition_spectra.npz',frequency_hz=f,
        **{f'{name}_{ch}':values[ch] for name,values in [('secondary',s_off),('laser_only',s_laser),('superposition_error',s_error)] for ch in ('A','E')})
    write_json(out/'diagnostic_summary.json',record);print(record,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parent',type=Path,default=ROOT/'outputs/referee/task_2_8_20261008_annual_laser_residual')
    run(p.parse_args())
