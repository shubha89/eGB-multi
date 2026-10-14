"""Paired laser-on/off validation with bit-identical nonlaser noise inputs.

Run from the local eGB checkout, with a fresh --output directory. This tests
whether retaining laser noise changes PSD weighting in the synthetic corrected
pipeline; it does not validate mission clock/filter calibration or full FF.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/referee/.cache'))
sys.path.insert(0,str(ROOT/'src'))

import numpy as np
from scipy.signal import csd,welch
from lisainstrument import Instrument,SchedulerConfigSerial
from egb_multi.noise import pytdi_data_from_lisainstrument_result,xyz_from_pytdi_data,aet_from_xyz
from egb_multi.comparison import AEPSD,AESeries,AEInnerProduct,TDIConvention
from egb_multi.circular_fit import CircularAmplitudeBasis,maximize_circular_orientation
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import precompute_jax_link_geometry,eccentric_links_jax
from egb_multi.external import state_from_lisaorbits
from egb_multi.tdi import pytdi_data_from_links,LINK_LABELS
from referee_task_2_objective_orientation import make_common_orbit_file,assert_local


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')


def digest(array):
    a=np.asarray(array)
    h=hashlib.sha256(str((a.shape,a.dtype.str)).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def generate(orbit_path, *, model,laser_on,seed,samples,dt,disabled_noises=(),t0=0.,noise_cutoff=None,return_tdi_data=False):
    kwargs=dict(size=samples,dt=dt,t0=t0,seed=seed,orbits=str(orbit_path),orbit_dataset='tcb/ltt',
        physics_upsampling=1,aafilter=None,telemetry_downsampling=samples,
        interpolation=('lagrange',45),noises_f_min_hz=noise_cutoff)
    instrument=Instrument(**kwargs)
    if model=='secondary':
        instrument.disable_all_noises(excluding=['laser','test-mass','oms'])
    elif model=='laser_only':
        instrument.disable_all_noises(excluding=['laser'])
    for contribution in disabled_noises:
        {'clock':instrument.disable_clock_noise,'ranging':instrument.disable_ranging_noises}[contribution]()
    if not laser_on:
        instrument.disable_laser_noise()
    started=time.perf_counter()
    raw=instrument.export_numpy_full(cfgscheduler=SchedulerConfigSerial(chunk_size=samples))
    # Check every exposed named nonlaser noise array, not only equal seeds.
    noise_hashes={};laser_hashes={};laser_statistics={}
    for group in ('mosa_data','sat_data'):
        data=getattr(raw,group)
        for field in fields(data):
            if 'noise' in field.name and 'laser' in field.name:
                for label,array in getattr(data,field.name).items():
                    key=f'{group}.{field.name}.{label}'
                    laser_hashes[key]=digest(array)
                    laser_statistics[key]={'mean_Hz':float(np.mean(array)),
                        'standard_deviation_Hz':float(np.std(array)),
                        'maximum_absolute_Hz':float(np.max(abs(array)))}
            if 'noise' in field.name and 'laser' not in field.name:
                values=getattr(data,field.name)
                for label,array in values.items():
                    noise_hashes[f'{group}.{field.name}.{label}']=digest(array)
    data=pytdi_data_from_lisainstrument_result(raw)
    xyz=xyz_from_pytdi_data(data,generation=2,order=45,delay_order=5)
    ae=aet_from_xyz(xyz)
    carrier=float(instrument.central_freq)
    result=dict(t=np.array(raw.t),delays={label:np.array(data.delays[f'd_{label}']) for label in LINK_LABELS},
        channels={ch:np.asarray(ae[ch])/carrier for ch in ('A','E')},
        metadata=dict(model=model,laser_on=laser_on,kwargs=kwargs,disabled_noises=disabled_noises,
            laser_asds=instrument.laser_asds,central_frequency_hz=carrier,
            nonlaser_noise_hashes=noise_hashes,laser_noise_hashes=laser_hashes,
            laser_noise_statistics=laser_statistics,wall_s=time.perf_counter()-started))
    if return_tdi_data:result['tdi_data']=data
    del raw,data,xyz,ae
    gc.collect()
    return result


def spectra(channels,dt,nperseg):
    f,pa=welch(channels['A'],fs=1/dt,nperseg=nperseg)
    _,pe=welch(channels['E'],fs=1/dt,nperseg=nperseg)
    _,cross=csd(channels['A'],channels['E'],fs=1/dt,nperseg=nperseg)
    coherence=np.abs(cross)**2/(pa*pe)
    return f,{'A':pa,'E':pe},cross.conjugate(),coherence


def pair(out,orbit_path, *, model,seed,samples,dt,nperseg,disabled_noises=(),label=None,t0=0.,noise_cutoff=None):
    folder=out/f'{label or model}_seed{seed}'
    folder.mkdir()
    print(f'Paired test {model}, seed={seed}: laser OFF',flush=True)
    off=generate(orbit_path,model=model,laser_on=False,seed=seed,samples=samples,dt=dt,disabled_noises=disabled_noises,t0=t0,noise_cutoff=noise_cutoff)
    print(f'Paired test {model}, seed={seed}: laser ON',flush=True)
    on=generate(orbit_path,model=model,laser_on=True,seed=seed,samples=samples,dt=dt,disabled_noises=disabled_noises,t0=t0,noise_cutoff=noise_cutoff)
    input_hashes_off=off['metadata']['nonlaser_noise_hashes']
    input_hashes_on=on['metadata']['nonlaser_noise_hashes']
    mismatches=[key for key in set(input_hashes_off)|set(input_hashes_on) if input_hashes_off.get(key)!=input_hashes_on.get(key)]
    if mismatches:
        raise RuntimeError(f'Unpaired nonlaser draws: {mismatches}')
    np.testing.assert_array_equal(on['t'],off['t'])
    for link_label in LINK_LABELS:
        np.testing.assert_array_equal(on['delays'][link_label],off['delays'][link_label])
    halo=int(np.ceil(8*max(np.max(d) for d in on['delays'].values())/dt))+4*46
    selected=slice(halo,-halo)
    trimmed={name:{ch:run['channels'][ch][selected] for ch in ('A','E')} for name,run in (('on',on),('off',off))}
    delta={ch:trimmed['on'][ch]-trimmed['off'][ch] for ch in ('A','E')}
    f,psd_off,cross_off,coh_off=spectra(trimmed['off'],dt,nperseg)
    f_on,psd_on,cross_on,coh_on=spectra(trimmed['on'],dt,nperseg)
    f_delta,psd_delta,_,_=spectra(delta,dt,nperseg)
    np.testing.assert_array_equal(f,f_on);np.testing.assert_array_equal(f,f_delta)
    band=(f>=.0001)&(f<=.03)
    columns=[f,psd_off['A'],psd_off['E'],psd_on['A'],psd_on['E'],psd_delta['A'],psd_delta['E'],
        coh_off,coh_on,cross_off.real,cross_off.imag,cross_on.real,cross_on.imag]
    np.savetxt(folder/'paired_psd.csv',np.column_stack(columns),delimiter=',',comments='',
        header='frequency_hz,PSD_A_off,PSD_E_off,PSD_A_on,PSD_E_on,PSD_A_increment,PSD_E_increment,coherence_off,coherence_on,CSD_AE_off_real,CSD_AE_off_imag,CSD_AE_on_real,CSD_AE_on_imag')
    np.savez_compressed(folder/'paired_noise.npz',t=on['t'][selected],
        **{f'{name}_{ch}_fractional':values[ch] for name,values in (*trimmed.items(),('increment',delta)) for ch in ('A','E')})
    for name,run in (('on',on),('off',off)):
        dump(folder/f'{name}_metadata.json',run['metadata'])
    quantiles=lambda x:dict(zip(('median','p95','maximum'),np.quantile(x,[.5,.95,1.])))
    result=dict(model=label or model,disabled_noises=disabled_noises,seed=seed,samples=samples,dt_s=dt,nperseg=nperseg,halo_samples=halo,
        pairing=dict(nonlaser_arrays_bit_identical=len(input_hashes_off),unmatched_arrays=mismatches,
            timestamps_bit_identical=True,MPRS_bit_identical=True),
        band_hz=[.0001,.03],psd_unit='fractional_frequency_squared_per_hz',
        PSD_increment_to_off={ch:quantiles(psd_delta[ch][band]/psd_off[ch][band]) for ch in ('A','E')},
        absolute_fractional_PSD_change={ch:quantiles(np.abs(psd_on[ch][band]/psd_off[ch][band]-1)) for ch in ('A','E')},
        coherence=dict(off_median=float(np.median(coh_off[band])),on_median=float(np.median(coh_on[band])),
            absolute_change=quantiles(np.abs(coh_on[band]-coh_off[band]))))
    dump(folder/'comparison.json',result)
    print(json.dumps(result),flush=True)
    return result,dict(folder=folder,t=on['t'][selected],delays=on['delays'],full_t=on['t'],
        psd_on=psd_on,psd_off=psd_off,frequency=f,halo=halo)


def waveform_checks(out,orbit,reference,pairs):
    state=state_from_lisaorbits(orbit,reference['full_t'])
    geometry=precompute_jax_link_geometry(state,orbits=orbit)
    convention=TDIConvention(2,45,5,'analytic_esa_trailing_BME_J2000','shared_MPRS_for_laser_on_off_pair')
    halo=reference['halo']
    def response(source):
        links=eccentric_links_jax(source,geometry,batch_size=1,physics_mode='1pn')
        data=pytdi_data_from_links(state,links,link_delays=reference['delays'])
        ae=aet_from_xyz(xyz_from_pytdi_data(data,generation=2,order=45,delay_order=5))
        return AESeries(reference['t'],{ch:np.asarray(ae[ch].real)[halo:-halo] for ch in ('A','E')},convention)
    results=[]
    for frequency,e in ((.001,.1),(.003,.3)):
        print(f'Checking waveform PSD weighting: f={frequency}, e={e}',flush=True)
        source=EccentricBinaryParams(mean_motion=np.pi*frequency,eccentricity=e,m1_solar=.8,m2_solar=.6,
            inclination=1.21,beta=.2,lambda_=-1.,psi=.71,phi0=.617)
        template=replace(source,eccentricity=0.)
        signal=response(source);candidate=response(template)
        basis=CircularAmplitudeBasis.generate(template,response)
        for comparison,arrays in pairs:
            metrics={}
            for name in ('off','on'):
                psd=AEPSD(arrays['frequency'],arrays[f'psd_{name}'],convention,
                    dict(sidedness='one_sided',scaling='density',noise_model=f"{comparison['model']}_laser_{name}"))
                metric=AEInnerProduct(signal,psd,(.0001,.03))
                fixed=metric.compare(candidate)
                maximum=maximize_circular_orientation(metric,basis)
                metrics[name]=dict(snr=np.sqrt(metric.reference_norm),fixed_overlap=fixed['overlap'],
                    physical_extrinsic_maximum=maximum['overlap'],maximum=maximum)
            result=dict(model=comparison['model'],seed=comparison['seed'],frequency_hz=frequency,eccentricity=e,
                results=metrics,relative_SNR_change=metrics['on']['snr']/metrics['off']['snr']-1,
                fixed_overlap_change=metrics['on']['fixed_overlap']-metrics['off']['fixed_overlap'],
                extrinsic_maximum_change=metrics['on']['physical_extrinsic_maximum']-metrics['off']['physical_extrinsic_maximum'],
                limitation='One fixed signal and response, same on/off delays; diagonal PSD reweighting diagnostic, not SCET-aware injection recovery or full FF.')
            results.append(result)
            dump(out/'waveform_comparison.json',results)
            print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=262144)
    parser.add_argument('--dt',type=float,default=5.)
    parser.add_argument('--nperseg',type=int,default=65536)
    parser.add_argument('--seeds',type=int,nargs='+',default=[20261007,20261008,20261009])
    args=parser.parse_args()
    out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Results must stay inside this checkout')
    if args.samples<4*args.nperseg or args.dt<=0:raise ValueError('Invalid sampling/PSD configuration')
    out.mkdir(parents=True,exist_ok=False)
    files=sorted((ROOT/'src/egb_multi').glob('*.py'))+[Path(__file__),ROOT/'benchmarks/referee_task_2_objective_orientation.py']
    hashes={}
    for path in files:
        rel=path.relative_to(ROOT);target=out/'snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    tolerances=dict(maximum_increment_PSD_ratio=1e-3,p95_absolute_fractional_PSD_change=.01,
        maximum_coherence_change=.01,relative_SNR_change=1e-3,extrinsic_maximum_change=1e-4)
    dump(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),root=str(ROOT),python=sys.executable,
        configuration=vars(args),versions={name:version(name) for name in ('numpy','scipy','jax','pytdi','lisainstrument','lisaorbits')},
        source_hashes=hashes,predetermined_acceptance_tolerances=tolerances,
        limitations=['Synthetic unfiltered sampling','Raw full-noise clocks are not calibrated','15-day duration; production convergence separate']))
    orbit_path=out/'native_orbits.h5'
    orbit=make_common_orbit_file(orbit_path,args.samples*args.dt,model='retarded')
    summaries=[];retained=[];started=time.perf_counter()
    for model in ('all_defaults','secondary'):
        for seed in args.seeds:
            summary,arrays=pair(out,orbit_path,model=model,seed=seed,samples=args.samples,dt=args.dt,nperseg=args.nperseg)
            summaries.append(summary)
            # Check PSD reweighting for both backgrounds at the first seed.
            if seed==args.seeds[0]:retained.append((summary,arrays))
            dump(out/'noise_comparison.json',summaries)
    waveform=waveform_checks(out,orbit,retained[0][1],retained)
    failures=[]
    for result in summaries:
        for ch in ('A','E'):
            if result['PSD_increment_to_off'][ch]['maximum']>tolerances['maximum_increment_PSD_ratio']:
                failures.append(f"{result['model']}/{result['seed']}/{ch}: laser increment excessive")
            if result['absolute_fractional_PSD_change'][ch]['p95']>tolerances['p95_absolute_fractional_PSD_change']:
                failures.append(f"{result['model']}/{result['seed']}/{ch}: PSD change excessive")
        if result['coherence']['absolute_change']['maximum']>tolerances['maximum_coherence_change']:
            failures.append(f"{result['model']}/{result['seed']}: coherence change excessive")
    for result in waveform:
        if abs(result['relative_SNR_change'])>tolerances['relative_SNR_change'] or abs(result['extrinsic_maximum_change'])>tolerances['extrinsic_maximum_change']:
            failures.append(f"{result['model']}/{result['frequency_hz']}: weighted waveform comparison changes")
    assert_local()
    final=dict(status='passed' if not failures else 'requires_investigation',failures=failures,
        noise_pairs=len(summaries),waveform_comparisons=len(waveform),wall_s=time.perf_counter()-started,
        conclusion_scope='Retaining laser noise under the stated settings/backgrounds; no certification of a full mission noise model.')
    dump(out/'validation_summary.json',final);print(json.dumps(final,indent=2),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
