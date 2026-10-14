#!/usr/bin/env python
"""Controlled laser-noise and convention audit using the local eGB checkout.

Coherence of a laser-only residual need not vanish: suppression relative to
secondary noise, rather than coherence alone, is the cancellation criterion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'outputs/referee/.cache'))
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
from scipy.signal import csd, welch
from lisainstrument import Instrument, SchedulerConfigSerial
from pytdi import Data
from egb_multi.noise import pytdi_data_from_lisainstrument_result, xyz_from_pytdi_data, aet_from_xyz
from referee_task_2_objective_orientation import make_common_orbit_file, assert_local


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))+'\n')


def run_case(out, name, *, dt=10., duration=327680., instrument_order=3,
             tdi_order=7, moving=True, clocks=False, keep=('laser',), seed=20261007,
             upsampling=1, aafilter=None, geometry_model='instantaneous'):
    folder = out/name
    folder.mkdir()
    size = round(duration/dt)
    orbit_path = out/f'{geometry_model}_orbits.h5'
    if not orbit_path.exists():
        make_common_orbit_file(orbit_path,duration,model=geometry_model)
    kwargs = dict(size=size, dt=dt, t0=0., seed=seed, physics_upsampling=upsampling,
                  aafilter=aafilter, telemetry_downsampling=size,
                  interpolation=('lagrange', instrument_order), noises_f_min_hz=None,
                  orbits=str(orbit_path) if moving else 'static')
    if moving:
        kwargs['orbit_dataset'] = 'tcb/ltt'
    instrument = Instrument(**kwargs)
    instrument.disable_all_noises(excluding=list(keep)+(['clock'] if clocks else []))
    started = time.perf_counter()
    result = instrument.export_numpy_full(cfgscheduler=SchedulerConfigSerial(chunk_size=size))
    data = pytdi_data_from_lisainstrument_result(result, include_local_metrology=True)
    native = Data.from_instrument(result, signals='fluctuations')
    # Native adapter also retains sidebands; carrier TDI uses SCI/TMI/REF only.
    carrier_errors = {key: float(np.max(np.abs(data.measurements[key]-native.measurements[key])))
                      for key in data.measurements if not '_sb_' in key}
    delay_errors = {key: float(np.max(np.abs(data.delays[key]-native.delays[key]))) for key in data.delays}
    xyz = xyz_from_pytdi_data(data, generation=2, order=tdi_order, delay_order=5)
    native_xyz = xyz_from_pytdi_data(native, generation=2, order=tdi_order, delay_order=5)
    channels = aet_from_xyz(xyz)
    halo = int(np.ceil(8*max(np.max(v) for v in data.delays.values())/dt))+4*(tdi_order+1)
    nperseg = round(81920/dt)
    trimmed = {key: value[halo:-halo] for key, value in channels.items()}
    f, pa = welch(trimmed['A'], fs=1/dt, nperseg=nperseg)
    _, pe = welch(trimmed['E'], fs=1/dt, nperseg=nperseg)
    _, cross = csd(trimmed['A'], trimmed['E'], fs=1/dt, nperseg=nperseg)
    coh = np.divide(np.abs(cross)**2, pa*pe, out=np.zeros_like(pa), where=pa*pe>0)
    np.savetxt(folder/'psd.csv', np.column_stack((f, pa/instrument.central_freq**2,
        pe/instrument.central_freq**2, coh, cross.real/instrument.central_freq**2,
        cross.imag/instrument.central_freq**2)), delimiter=',',
        header='frequency_hz,PSD_A_fractional_per_hz,PSD_E_fractional_per_hz,coherence_squared,CSD_AconjE_real_fractional_per_hz,CSD_AconjE_imag_fractional_per_hz', comments='')
    metrics = dict(name=name, kwargs=kwargs, geometry_model=geometry_model,
        enabled_noises=list(keep), default_clock_errors=clocks or 'clock' in keep,
        tdi_generation=2, tdi_order=tdi_order, delay_order=5, unit='frequency',
        carrier_adapter_max_absolute_errors_hz=carrier_errors,
        delay_adapter_max_absolute_errors_s=delay_errors,
        native_vs_local_xyz_max_absolute_error_hz=max(float(np.max(np.abs(xyz[k]-native_xyz[k]))) for k in xyz),
        halo_samples=halo, nperseg=nperseg, wall_s=time.perf_counter()-started,
        central_frequency_hz=instrument.central_freq, sample_metrics=[])
    for target in (.001, .003, .01, .03):
        j=int(np.argmin(np.abs(f-target)))
        metrics['sample_metrics'].append(dict(frequency_hz=f[j], PSD_A=pa[j]/instrument.central_freq**2,
            PSD_E=pe[j]/instrument.central_freq**2, coherence_squared=coh[j]))
    dump(folder/'metrics.json', metrics)
    print(json.dumps(dict(name=name, wall_s=metrics['wall_s'], samples=metrics['sample_metrics'],
        adapter_xyz_error=metrics['native_vs_local_xyz_max_absolute_error_hz'])), flush=True)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('initial','refine','confirm'), default='initial')
    args=parser.parse_args()
    out=args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('Results must remain in this checkout')
    out.mkdir(parents=True, exist_ok=False)
    sources=sorted((ROOT/'src/egb_multi').glob('*.py'))+[
        Path(__file__),ROOT/'benchmarks/referee_task_2_objective_orientation.py',
        ROOT/'tests/test_lisa_conventions.py',ROOT/'tests/test_noise.py']
    hashes={}
    for path in sources:
        relative=path.relative_to(ROOT)
        target=out/'snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    dump(out/'manifest.json',dict(project_root=str(ROOT),stage=args.stage,python=sys.executable,
        versions={name:version(name) for name in ('numpy','scipy','jax','lisainstrument','lisaorbits','lisagwresponse','pytdi')},
        source_hashes=hashes))
    initial=[
        ('static_fast',dict(moving=False)),
        ('moving_fast',dict()),
        ('moving_default_clocks',dict(clocks=True)),
        ('static_order45',dict(moving=False,instrument_order=45,tdi_order=45)),
        ('moving_order45',dict(instrument_order=45,tdi_order=45)),
        ('moving_secondary',dict(keep=('test-mass','oms'),instrument_order=45,tdi_order=45)),
    ]
    refine=[
        ('instrument3_tdi45',dict(instrument_order=3,tdi_order=45)),
        ('instrument45_tdi7',dict(instrument_order=45,tdi_order=7)),
        ('moving_order15',dict(instrument_order=15,tdi_order=15)),
        ('moving_order31',dict(instrument_order=31,tdi_order=31)),
        ('dt5_order45',dict(dt=5.,instrument_order=45,tdi_order=45)),
        ('dt2p5_order45',dict(dt=2.5,instrument_order=45,tdi_order=45)),
        ('dt2p5_order45_secondary',dict(dt=2.5,instrument_order=45,tdi_order=45,keep=('test-mass','oms'))),
        ('dt2p5_order45_all',dict(dt=2.5,instrument_order=45,tdi_order=45,
            keep=('laser','modulation','clock','test-mass','backlink','reciprocal_backlink','oms','ranging',
                  'angular-jitters','dws','moc-time-correlation','longitudinal-jitters'))),
    ]
    confirm=[
        ('native_dt5_order45_laser',dict(dt=5.,duration=1310720.,geometry_model='retarded',instrument_order=45,tdi_order=45)),
        ('native_dt2p5_order45_laser',dict(dt=2.5,duration=1310720.,geometry_model='retarded',instrument_order=45,tdi_order=45)),
        ('native_dt5_order63_laser',dict(dt=5.,duration=1310720.,geometry_model='retarded',instrument_order=63,tdi_order=63)),
        ('native_dt5_order45_secondary',dict(dt=5.,duration=1310720.,geometry_model='retarded',instrument_order=45,tdi_order=45,keep=('test-mass','oms'))),
        ('native_dt5_order45_all',dict(dt=5.,duration=1310720.,geometry_model='retarded',instrument_order=45,tdi_order=45,
            keep=('laser','modulation','clock','test-mass','backlink','reciprocal_backlink','oms','ranging',
                  'angular-jitters','dws','moc-time-correlation','longitudinal-jitters'))),
    ]
    cases={'initial':initial,'refine':refine,'confirm':confirm}[args.stage]
    results=[run_case(out,name,**kwargs) for name,kwargs in cases]
    assert_local()
    dump(out/'summary.json',results)


if __name__=='__main__':
    main()
