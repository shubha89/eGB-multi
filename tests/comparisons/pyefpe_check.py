"""Bounded source-only pyEFPE release comparison, with no LISA timing conflation.

Match SI masses/distance and Newtonian+leading-PM physics first. The native
default-PN comparison is separately labelled as model differences. All annual
arrays are evaluated in bounded chunks; only scalar diagnostics are retained.
"""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,gc,warnings,shutil
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from egb_multi.constants import C_M_PER_S,G_M3_KG_S2,KILOPARSEC_M
from egb_multi.eccentric import EccentricBinaryParams,eccentric_polarizations,peters_mathews_derivatives
import pyEFPE
EFPE=importlib.import_module('pyEFPE.waveform.EFPE')
from comparison_utils import atomic, pair_stats
from pyEFPE.utils.utils import sorted_vals_in_intervals

def release_time_domain(model,times):
    """Explicit adapter: 2 h0 Re sum A_mp(t) exp(-i Phi_mp(t)).

    This uses the release's own mode selection, amplitude evaluators and
    phase polynomials. It is NOT a native v1.0.0 time-domain API or an SPA
    Fourier transform. No source dynamics are replaced or fitted.
    """
    times=np.asarray(times)
    if np.any(times<model.sol.all_ts[0]) or np.any(times>model.sol.all_ts[-1]):
        raise ValueError('No pyEFPE extrapolation permitted')
    tids,idx=sorted_vals_in_intervals(times,model.sol.all_ts[model.mode_interp_idx],model.sol.all_ts[model.mode_interp_idx+1])
    query=times[tids]
    amps=model.compute_Amplitudes(query,model.necessary_modes[idx,0],idx)
    segment=model.mode_interp_idx[idx]
    x=(query-model.sol.ts[segment])/model.sol.hs[segment]
    phase=np.zeros_like(x)
    for col in reversed(range(model.mode_phases_Qs[0].shape[1])):
        phase=x*(model.mode_phases_Qs[0][idx,col]+phase)
    phase+=model.mode_phases_y0[0][idx]
    values=2*(amps.real*np.cos(phase)[:,None]+amps.imag*np.sin(phase)[:,None])
    result=np.array([np.bincount(tids,weights=values[:,j],minlength=times.size) for j in (0,1)])*model.h0_pref
    if not np.all(np.isfinite(result)):raise ValueError('Nonfinite pyEFPE reconstructed polarizations')
    return result

def parameters(source,duration,pn=0,tol=1e-10):
    # Equal physical SI masses, not numerically equal incompatible Msun units.
    factor=G_M3_KG_S2*source.total_mass_kg/C_M_PER_S**3/(source.m1_solar+source.m2_solar)/EFPE.t_sun_s
    mass1,mass2=source.m1_solar*factor,source.m2_solar*factor
    f=source.mean_motion/np.pi;e=source.eccentricity
    M=(mass1+mass2)*EFPE.t_sun_s
    y0=(np.pi*M*f)**(1/3)/np.sqrt(1-e*e)
    nd,ed=peters_mathews_derivatives(source.mean_motion,e,source.total_mass_kg,source.symmetric_mass_ratio)
    dy_over_y=nd/source.mean_motion/3+e*ed/(1-e*e)
    # The release's stop parameter is converted to yf assuming circular final
    # motion. Use that documented y proxy to stop after a short covered span,
    # instead of integrating a mHz binary all the way to ISCO.
    yend=y0*(1+max(2*float(dy_over_y)*duration,1e-10))
    f_end=yend**3/(np.pi*M)
    return dict(mass1=mass1,mass2=mass2,e_start=e,
        inclination=source.inclination,f22_start=f,f22_end=f_end,
        # The nonprecessing J-frame spherical-harmonic m=2 phase origin is
        # offset from eGB's orbital-plane x axis by pi/2. This flips m=2
        # while preserving m=0; a global polarization sign is insufficient.
        phi_start=source.phi0+np.pi/2,mean_anomaly_start=0.,
        distance=source.distance_m/(EFPE.Mpc_s*C_M_PER_S),
        pn_phase_order=pn,pn_spin_order=0,Amplitude_tol=tol,
        Amplitude_pmax=200,Interpolate_Amplitudes=True,
        RR_sol_rtol=[1e-12]*8,RR_sol_atol=[1e-14,1e-14,1e-10,1e-10,1e-14,1e-10,1e-10,1e-10])

def model_for(source,duration,pn,tol):
    p=parameters(source,duration,pn,tol)
    tick=time.perf_counter();model=pyEFPE.pyEFPE(p);setup=time.perf_counter()-tick
    start=float(model.sol.all_ts[0]);span=float(model.sol.all_ts[-1]-start)
    if span<duration:raise ValueError(f'Upstream integration span {span} < requested {duration}; do not extrapolate')
    return model,p,setup,start,span


def pilot(out,f,e,duration,dt,m1=.8,m2=.6):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    s=EccentricBinaryParams(mean_motion=np.pi*f,eccentricity=e,m1_solar=m1,m2_solar=m2,
        distance_m=KILOPARSEC_M,inclination=.8,phi0=.2)
    model,p,setup,start,span=model_for(s,duration,0,1e-12)
    tau=np.arange(0,duration,dt)
    tick=time.perf_counter();a=release_time_domain(model,start+tau);pywall=time.perf_counter()-tick
    tick=time.perf_counter();b=np.array(eccentric_polarizations(s,tau,physics_mode='newtonian',evolution_mode='peters_mathews',pm_ode_rtol=1e-12,pm_ode_phase_atol=1e-10));egbwall=time.perf_counter()-tick
    result=dict(status='complete_pilot',source=asdict(s),pyefpe_parameters=p,
        duration_s=duration,dt_s=dt,pyefpe_setup_s=setup,pyefpe_td_s=pywall,egb_numpy_td_s=egbwall,
        metrics=pair_stats(a,b),pyefpe_start_time_s=start,integration_span_s=span,
        upstream_absolute_time_ulp_s=float(abs(np.spacing(start))),
        pyefpe_import_path=str(pyEFPE.__file__),pyefpe_release='v1.0.0',
        pyefpe_release_commit='5b6964c0002893083c69f3ccdd687eb8b5d60e76',
        upstream_distribution_version=importlib.metadata.version('pyEFPE'),
        required_frequency_scope='source polarizations only; no LISA response',
        pyefpe_td_interface='explicit harmonic reconstruction adapter; release v1.0.0 exposes FD only',
        waveform_arrays_saved=False)
    atomic(out/'result.json',result);print(json.dumps(result,indent=2))

def annual(out,f,e,duration,dt,m1=.8,m2=.6):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Use a new result folder')
    shutil.copy2(__file__,out/'pyefpe_check_source.py')
    source=EccentricBinaryParams(mean_motion=np.pi*f,eccentricity=e,m1_solar=m1,m2_solar=m2,
        distance_m=KILOPARSEC_M,inclination=.8,phi0=.2)
    tau=np.arange(0,duration,dt);models={};rows={};fd_outputs={};strong=None
    frequencies=None
    for key,pn,tol in [('shared_0pn',0,1e-12),('higher_pn_strict',6,1e-12),('higher_pn_loose_modes',6,1e-3)]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always');model,p,setup,start,span=model_for(source,duration,pn,tol)
        arrays=np.empty((2,len(tau)));ticks=[];chunk=65536
        for j,first in enumerate(range(0,len(tau),chunk)):
            last=min(len(tau),first+chunk);tick=time.perf_counter()
            arrays[:,first:last]=release_time_domain(model,start+tau[first:last]);ticks.append(time.perf_counter()-tick)
            if j%16==0 or last==len(tau):
                atomic(out/'progress.json',dict(status='running',stage=key,block=j+1,blocks=int(np.ceil(len(tau)/chunk))))
                print(json.dumps(dict(stage=key,block=j+1,total=int(np.ceil(len(tau)/chunk)))),flush=True)
        # Repeat selected chunks after setup to expose dispersion without
        # retaining an annual waveform catalogue or claiming a native TD API.
        repeat=[]
        for first in np.linspace(0,max(0,len(tau)-chunk),5).astype(int):
            tick=time.perf_counter();again=release_time_domain(model,start+tau[first:first+chunk]);repeat.append(time.perf_counter()-tick)
            if not np.array_equal(again,arrays[:,first:first+chunk]):raise ValueError('Repeated harmonic reconstruction changed')
        if key!='higher_pn_loose_modes':
            physics='newtonian' if pn==0 else '1pn'
            tick=time.perf_counter();egb=np.array(eccentric_polarizations(source,tau,physics_mode=physics,
                evolution_mode='peters_mathews',pm_ode_rtol=1e-12,pm_ode_phase_atol=1e-10));egb_wall=time.perf_counter()-tick
            comparison=pair_stats(arrays,egb)
            if pn==0 and comparison['raw_relative_l2']>1e-3:
                atomic(out/'failed_common_physics_gate.json',comparison)
                raise ValueError('Shared Newtonian+PM error exceeds the predeclared 1e-3 diagnostic gate')
            egb_repeated=[]
            for _ in range(3):
                tick=time.perf_counter();again=np.array(eccentric_polarizations(source,tau,physics_mode=physics,
                    evolution_mode='peters_mathews',pm_ode_rtol=1e-12,pm_ode_phase_atol=1e-10))
                egb_repeated.append(time.perf_counter()-tick)
                if not np.array_equal(again,egb):raise ValueError('Repeated evolving NumPy source changed')
                del again
            del egb
        else:egb_wall=None;egb_repeated=[];comparison=pair_stats(strong,arrays)
        if key=='higher_pn_strict':strong=arrays
        if key=='shared_0pn':del arrays
        if key=='higher_pn_loose_modes':del arrays,strong
        if frequencies is None or key=='higher_pn_strict':
            ff=[]
            for a,b in zip(model.ts_interp_w0,model.ts_interp_wf):
                if b>a and a>0:ff.append(np.linspace(a+.1*(b-a),b-.1*(b-a),16)/(2*np.pi))
            frequencies=np.unique(np.concatenate(ff))
        # Native public frequency-domain API. Grid resolves active mode chirp
        # intervals, not the 1/year DFT of a finite rectangular observation.
        fd_warm=[]
        fd_warnings=[]
        for _ in range(3):
            with warnings.catch_warnings(record=True) as wf:
                warnings.simplefilter('always');tick=time.perf_counter()
                fd=np.asarray(model.generate_waveform(frequencies));fd_warm.append(time.perf_counter()-tick)
            fd_warnings+=list(str(x.message) for x in wf)
        bad=~np.all(np.isfinite(fd),axis=0)
        fd_outputs[key]=(fd,start,frequencies.copy())
        rows[key]=dict(pn_order_twice=pn,amplitude_power_tolerance=tol,parameters=p,
            model_setup_s=setup,full_duration_harmonic_adapter_s=sum(ticks),
            selected_repeated_chunk_s=repeat,chunk_samples=chunk,
            egb_numpy_full_duration_polarizations_s=egb_wall,egb_numpy_repeated_full_duration_s=egb_repeated,source_comparison=comparison,
            native_fd_s=fd_warm,native_fd_frequency_count=len(frequencies),native_fd_nonzero_values=int(np.count_nonzero(fd)),
            native_fd_status='nonfinite_output_requires_review' if np.any(bad) else 'finite_output',
            native_fd_nonfinite_frequency_count=int(np.count_nonzero(bad)),
            native_fd_nonfinite_frequencies_hz=frequencies[bad].tolist(),
            native_fd_warnings=sorted(set(fd_warnings)),
            initial_absolute_time_s=start,initial_absolute_time_ulp_s=float(abs(np.spacing(start))),
            integrated_span_s=span,warnings=sorted(set(str(x.message) for x in w)),
            mode_interval_count=len(model.necessary_modes))
        atomic(out/'partial.json',rows);models[key]=model;gc.collect()
    strict,st,grid=fd_outputs['higher_pn_strict'];loose,lt,lgrid=fd_outputs['higher_pn_loose_modes']
    if not np.array_equal(grid,lgrid) or st!=lt:raise ValueError('FD convergence control has inconsistent grid or epoch')
    mask=np.all(np.isfinite(strict)&np.isfinite(loose),axis=0)
    fd_error=float(np.linalg.norm(strict[:,mask]-loose[:,mask])/np.linalg.norm(strict[:,mask])) if np.any(mask) else None
    result=dict(status='complete',source=asdict(source),duration_s=duration,dt_s=dt,samples=len(tau),rows=rows,
        native_fd_common_finite_subset_loose_vs_strict_relative_error=fd_error,
        native_fd_common_finite_subset_count=int(np.count_nonzero(mask)),
        native_fd_full_grid_finite=bool(np.all(mask)),
        pyefpe_release='v1.0.0',pyefpe_commit='5b6964c0002893083c69f3ccdd687eb8b5d60e76',
        distribution_metadata_version=importlib.metadata.version('pyEFPE'),
        contracts=dict(shared_physics='Newtonian amplitudes/dynamics with leading PM; zero spins, physical SI masses/distance and mapped periastron phase',
            higher_pn='pyEFPE 3PN phasing vs eGB conservative 1PN plus leading PM; model difference, not pure numerical error',
            native_FD='upstream SPA/SUA output on active instantaneous harmonic intervals; not the finite-year eGB FFT',
            common_TD='release harmonic reconstruction adapter vs existing eGB NumPy source; no native-release TD API claim',
            source_only='no LISA response, TDI or noise weighting in this benchmark',
            cutoff_control='1e-3 versus 1e-12 mode-power selection at identical higher-PN dynamics and epoch'),
        waveform_arrays_saved=False)
    atomic(out/'result.json',result);atomic(out/'progress.json',dict(status='complete'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--f0',type=float,default=.003);p.add_argument('--e',type=float,default=.3)
    p.add_argument('--duration',type=float,default=86400.);p.add_argument('--dt',type=float,default=2.5)
    p.add_argument('--m1',type=float,default=.8);p.add_argument('--m2',type=float,default=.6)
    p.add_argument('--annual',action='store_true')
    a=p.parse_args();(annual if a.annual else pilot)(a.output,a.f0,a.e,a.duration,a.dt,a.m1,a.m2)
