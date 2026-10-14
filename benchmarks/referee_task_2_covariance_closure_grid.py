"""Extend the reference covariance and test its final numerical resolution.

Uses this checkout's native spectra and native nested moving pyTDI delays.
Nested full grids permit all-bin comparisons of the implemented metrics;
independent direct selected bins test interpolation, filter and annual-epoch
convergence. This does not establish an exact time-dependent likelihood.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/referee/.cache'))
import numpy as np
from egb_multi.external import default_lisaorbits,link_delays_from_lisaorbits
from egb_multi.metric_perturbation import relative_covariance_eigenvalues,normalized_overlap_change_bound
from egb_multi.noise_covariance import moving_native_secondary_symbol
from referee_task_2_covariance_grid import chunked_model
from referee_task_2_frozen_noise_audit import instrument
from referee_task_2_objective_orientation import assert_local,write_json


def interpolate(f,grid,cov):
    if f[0]<grid[0] or f[-1]>grid[-1]:raise ValueError('No covariance extrapolation')
    return np.stack([np.interp(f,grid,cov[:,a,b]) for a in range(2) for b in range(2)],axis=-1).reshape(-1,2,2)


def moving_model(f,definitions,orbit,*,geometry_grid,epochs):
    year=31557600.;dt=year/geometry_grid
    times=np.arange(-64,geometry_grid+65)*dt
    locations=(np.arange(epochs)+.5)*geometry_grid/epochs+64
    if np.any(locations!=locations.astype(int)):raise ValueError('Epochs must align exactly with the geometry grid')
    delays=link_delays_from_lisaorbits(orbit,times)
    answer=np.empty((len(f),2,2),complex)
    for start in range(0,len(f),8192):
        stop=min(len(f),start+8192)
        answer[start:stop]=moving_native_secondary_symbol(f[start:stop],times,delays,definitions,locations.astype(int)).mean(axis=0)
        if start%(32*8192)==0:print('Native moving covariance',stop,'/',len(f),'epochs',epochs,'geometry',geometry_grid,flush=True)
    return answer


def run(args):
    assert_local();out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);hashes={}
    files=[*list((ROOT/'src/egb_multi').glob('*.py')),Path(__file__),
        ROOT/'benchmarks/referee_task_2_covariance_grid.py',ROOT/'benchmarks/referee_task_2_frozen_noise_audit.py',
        ROOT/'benchmarks/referee_task_2_objective_orientation.py']
    for path in files:
        target=out/'snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    year=31557600.;orbit=default_lisaorbits('esa-trailing');definitions=instrument(1/year,.03125)
    delays=link_delays_from_lisaorbits(orbit,(np.arange(16)+.5)*year/16)
    write_json(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sha256=hashes,
        configuration=vars(args)|{'output':str(out)},cutoff_hz=1/year,noise_filter_dt_s=.03125,
        annual_epochs=16,model='native_nested_moving_delay_adiabatic_annual_secondary_PSD',
        environment={name:version(name) for name in ('numpy','scipy','pytdi','lisainstrument','lisaorbits')},
        scope='Reference covariance numerical convergence; stationary annual average of local moving symbols',production_certified=False))
    grids=[]
    for size in (1048576,2097152):
        df=1/(size*5.)
        uniform=np.arange(round(args.maximum_frequency/df)+1)*df
        grid=np.unique(np.r_[uniform,np.geomspace(1/(100*year),100*df,1024)])
        cov=np.zeros((len(grid),2,2),complex)
        print('Reference covariance nodes',len(grid),'df',df,flush=True)
        cov[1:]=moving_model(grid[1:],definitions,orbit,geometry_grid=args.geometry_grid,epochs=16)
        filename='annual_covariance.npz' if size==2097152 else 'coarse_covariance.npz'
        np.savez_compressed(out/filename,frequency_hz=grid,covariance=cov)
        grids.append((grid,cov))
    write_json(out/'covariance_ready.json',dict(status='annual_covariance_cache_ready',production_certified=False,
        reference_file='annual_covariance.npz',sha256=hashlib.sha256((out/'annual_covariance.npz').read_bytes()).hexdigest()))
    first,last=int(np.ceil(args.band[0]*year)),int(np.floor(args.band[1]*year))
    f=np.arange(first,last+1)/year
    coarse=interpolate(f,*grids[0]);fine=interpolate(f,*grids[1])
    epsilon=float(np.max(abs(relative_covariance_eigenvalues(fine,coarse)-1.)))
    rows=[dict(control='nested_frequency_grid_all_annual_bins',tested_bins=len(f),relative_covariance_epsilon=epsilon,
        uniform_same_family_overlap_change_bound=normalized_overlap_change_bound(epsilon))]
    write_json(out/'covariance_controls.json',rows)
    del f,coarse,fine
    indices=np.arange(first,last+1,31)
    for low,high in ((.029,.031),(.059,.061),(.089,.091)):
        if low<args.band[1]:indices=np.r_[indices,np.arange(max(first,int(np.ceil(low*year))),min(last,int(np.floor(high*year)))+1)]
    selected=np.unique(indices)/year
    baseline=moving_model(selected,definitions,orbit,geometry_grid=args.geometry_grid,epochs=16)
    interpolated=interpolate(selected,*grids[-1])
    controls=[('direct_interpolation_selected_bins',interpolated)]
    for name,dt,epochs,geometry in (('noise_filter_dt_refined',.015625,16,args.geometry_grid),
        ('annual_epochs_refined',.03125,32,args.geometry_grid),('moving_geometry_refined',.03125,16,2*args.geometry_grid)):
        print(name,'direct frequencies',len(selected),flush=True)
        candidate=moving_model(selected,instrument(1/year,dt),orbit,geometry_grid=geometry,epochs=epochs)
        controls.append((name,candidate))
        np.savez_compressed(out/(name+'.npz'),frequency_hz=selected,covariance=candidate)
    for name,candidate in controls:
        error=np.max(abs(relative_covariance_eigenvalues(baseline,candidate)-1.),axis=1)
        rows.append(dict(control=name,tested_bins=len(selected),relative_covariance_epsilon=float(error.max()),
            maximum_error_frequency_hz=float(selected[np.argmax(error)]),
            scope='Selected direct physical frequencies; not a uniform physical-model bound'))
        write_json(out/'covariance_controls.json',rows);print(rows[-1],flush=True)
    # Keep the old approximation as a diagnostic, not the expanded-band
    # reference. Its second-null correction is larger than the noise budget.
    frozen=chunked_model(selected,definitions,delays)
    for band in ((1e-5,.04),tuple(args.band)):
        keep=(selected>=band[0])&(selected<=band[1])
        correction=float(np.max(abs(relative_covariance_eigenvalues(frozen[keep],baseline[keep])-1)))
        rows.append(dict(control='native_moving_vs_frozen_physical_model',band_hz=list(band),
            tested_bins=int(keep.sum()),relative_covariance_epsilon=correction,
            scope='Physical approximation comparison at selected bins; not a numerical-convergence failure or uniform bound'))
    write_json(out/'covariance_controls.json',rows)
    failures=[row['control'] for row in rows if row['control']!='native_moving_vs_frozen_physical_model' and row['relative_covariance_epsilon']>args.covariance_allowance]
    write_json(out/'validation_summary.json',dict(status='passed_reference_covariance_numerical_controls' if not failures else 'requires_investigation',
        failures=failures,production_certified=False,stationary_model_only=True,
        limitations=['Annual stationary/adiabatic approximation and finite-window covariance require separate evidence',
            'Native secondary symbol is not a complete sampled instrument or noise-driven-locking simulation',
            'Sub-band infrared cache values are not certified as an exact moving finite-window noise covariance',
            'All-bin bound controls nested implemented pointwise metrics; direct physical controls are sampled']))
    if failures:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--maximum-frequency',type=float,default=.1)
    p.add_argument('--band',nargs=2,type=float,default=[1e-5,.08])
    p.add_argument('--covariance-allowance',type=float,default=1e-4)
    p.add_argument('--geometry-grid',type=int,default=8192)
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        if args.output.exists() and not (args.output/'validation_summary.json').exists():
            write_json(args.output/'validation_summary.json',dict(status='failed_covariance_closure_run',
                exception_type=type(exc).__name__,exception=str(exc),production_certified=False))
        raise
