"""Locate loss of laser/secondary superposition in the native float64 pipeline.

No noise model is changed. Shared noise inputs and delays are verified before
splitting the arithmetic diagnostic into instrument-stream and TDI parts.
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
from pytdi import Data
from egb_multi.noise import xyz_from_pytdi_data,aet_from_xyz
from referee_laser_on_off import generate,spectra
from referee_task_2_objective_orientation import write_json,assert_local


def run(args):
    out=args.output.resolve();parent=args.parent.resolve()
    if not out.is_relative_to(ROOT) or not parent.is_relative_to(ROOT):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);assert_local();hashes={}
    paths=[*list((ROOT/'src/egb_multi').glob('*.py')),Path(__file__),
        ROOT/'benchmarks/referee_laser_on_off.py',ROOT/'benchmarks/referee_task_2_objective_orientation.py',
        parent/'native_annual_orbits.h5']
    for path in paths:
        rel=path.relative_to(ROOT);dest=out/'snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes,
        'seed':20261008,'samples':262144,'dt_s':5.,'cutoff_hz':1/31557600.,
        'scope':'Float64 superposition diagnostic, not physical laser residual certification'})
    runs={}
    for name,model,on in [('on','secondary',True),('off','secondary',False),('laser','laser_only',True)]:
        print('Generating',name,flush=True)
        runs[name]=generate(parent/'native_annual_orbits.h5',model=model,laser_on=on,
            seed=20261008,samples=262144,dt=5.,noise_cutoff=1/31557600.,return_tdi_data=True)
        write_json(out/f'{name}_metadata.json',runs[name]['metadata'])
    if not runs['on']['metadata']['laser_noise_hashes'] or runs['on']['metadata']['laser_noise_hashes']!=runs['laser']['metadata']['laser_noise_hashes']:
        raise ValueError('Laser draws not matched; superposition diagnosis invalid')
    if runs['on']['metadata']['nonlaser_noise_hashes']!=runs['off']['metadata']['nonlaser_noise_hashes']:
        raise ValueError('Secondary draws not matched; superposition diagnosis invalid')
    on,off,laser=[runs[k]['tdi_data'] for k in ('on','off','laser')]
    for name in on.delays:
        np.testing.assert_array_equal(on.delays[name],off.delays[name])
        np.testing.assert_array_equal(on.delays[name],laser.delays[name])
    # Subtract large matched laser streams first, then the small secondary term.
    error={key:(on.measurements[key]-laser.measurements[key])-off.measurements[key] for key in on.measurements}
    mapped=Data(error,on.delays,on.fs);mapped.compute_delay_derivatives()
    projected=aet_from_xyz(xyz_from_pytdi_data(mapped,generation=2,order=45,delay_order=5))
    halo=198;selected=slice(halo,-halo);carrier=runs['on']['metadata']['central_frequency_hz']
    secondary={ch:runs['off']['channels'][ch][selected] for ch in ('A','E')}
    total={ch:runs['on']['channels'][ch][selected]-runs['laser']['channels'][ch][selected]-secondary[ch] for ch in ('A','E')}
    frontend={ch:projected[ch][selected]/carrier for ch in ('A','E')}
    backend={ch:total[ch]-frontend[ch] for ch in ('A','E')}
    f,s_off,_,_=spectra(secondary,5.,65536);mask=(f>=2e-5)&(f<=.04)
    rows={};arrays={'frequency_hz':f}
    for name,values in [('total_superposition',total),('instrument_stream_error',frontend),('TDI_evaluation_remainder',backend)]:
        _,psd,_,_=spectra(values,5.,65536)
        rows[name]={ch:{'maximum_PSD_over_secondary':float(np.max(psd[ch][mask]/s_off[ch][mask])),
            'p95_PSD_over_secondary':float(np.quantile(psd[ch][mask]/s_off[ch][mask],.95))} for ch in ('A','E')}
        arrays.update({f'{name}_{ch}':psd[ch] for ch in ('A','E')})
    np.savez_compressed(out/'arithmetic_spectra.npz',**arrays)
    maximum_raw=max(float(np.max(abs(v))) for v in on.measurements.values())
    record={'status':'diagnostic_completed','production_certified':False,
        'matched_laser_arrays':len(runs['on']['metadata']['laser_noise_hashes']),
        'matched_nonlaser_arrays':len(runs['on']['metadata']['nonlaser_noise_hashes']),
        'maximum_raw_measurement_abs_Hz':maximum_raw,'float64_epsilon':float(np.finfo(float).eps),
        'longdouble_epsilon':float(np.finfo(np.longdouble).eps),'components':rows,
        'limitations':['Loss of numerical superposition is separated; true laser-only residual still requires a precision/physical audit',
            'All projections use native float64 interpolation; component errors may be correlated']}
    write_json(out/'diagnostic_summary.json',record);print(record,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parent',type=Path,default=ROOT/'outputs/referee/task_2_8_20261008_annual_laser_residual')
    run(p.parse_args())
