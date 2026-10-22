"""Circular response checks at declared shared source/coordinate conventions.

No per-channel calibration is applied to the physical comparison. Static-arm
controls precede moving-orbit tests; JAXGB's TDI2 transform remains its native
equal-arm approximation. Only compact diagnostics are saved.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.interpolate import CubicSpline
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
from jaxgb.jaxgb import JaxGB,JaxGBaccurate
from egb_multi.constants import C_M_PER_S,G_M3_KG_S2,KILOPARSEC_M,SOLAR_MASS_KG
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_links_jax,precompute_jax_link_geometry
from egb_multi.external import (cartesian_frame_rotation,default_lisaorbits,
    state_from_lisaorbits,link_delays_from_lisaorbits)
from egb_multi.tdi import PreparedGWTDIMichelson
from egb_multi.noise import aet_from_xyz

from comparison_utils import atomic

class StaticOrbits:
    """A stationary equilateral triangle, returned in native ICRS meters."""
    L=2.5e9
    def __init__(self):
        a=np.arange(3)*2*np.pi/3
        positions=np.column_stack((np.cos(a),np.sin(a),np.zeros(3)))*self.L/np.sqrt(3)
        self.positions=positions@cartesian_frame_rotation('ecliptic','icrs').T
    def compute_position(self,t,sc):
        ids=np.atleast_1d(sc).astype(int)-1
        return np.broadcast_to(self.positions[ids],(np.size(t),len(ids),3)).copy()
    def compute_ltt(self,t,links=None):
        if links is None:links=[12,23,31,13,32,21]
        return np.full((np.size(t),np.size(links)),self.L/C_M_PER_S)

class EclipticOrbits:
    """Coordinate-only adapter; identical native delays and physical events."""
    def __init__(self,raw):self.raw=raw
    def __getattr__(self,name):return getattr(self.raw,name)
    def compute_position(self,*args,**kwargs):
        return np.asarray(self.raw.compute_position(*args,**kwargs))@cartesian_frame_rotation('icrs','ecliptic').T

def amplitude(source):
    return 2*source.symmetric_mass_ratio*(G_M3_KG_S2*source.total_mass_kg)**(5/3)*source.mean_motion**(2/3)/(C_M_PER_S**4*source.distance_m)

def parameters(source):
    # eGB's phase is orbital, JAXGB's is a sign-flipped GW phase.
    # Inclination reflection accounts for the source cross-polarization sign.
    return jnp.array([[source.mean_motion/np.pi,source.fdot,amplitude(source),
        source.lambda_,source.beta,source.psi,np.pi-source.inclination,
        np.pi-2*source.phi0]],dtype=jnp.float64)

def reconstruct(model,source,bins,t):
    """Real physical series from one-sided integrated FD coefficients.

    Inverse-transform the slow spectrum on its actual extended grid, then
    interpolate only that envelope. The physical carrier is kept analytic.
    The factor two restores the conjugate negative-frequency contribution.
    """
    T=float(getattr(model,'t_ext',model.t_obs));ext=float(getattr(model,'extension',0))
    n=bins.shape[-1];q=int(np.rint(source.mean_motion/np.pi*T))
    slow=np.fft.ifft(np.fft.ifftshift(bins,axes=-1),axis=-1)*n/T
    slow=np.concatenate((slow,slow[...,:1]),axis=-1)
    spline=CubicSpline(np.arange(n+1),slow,axis=-1,bc_type='periodic')
    u=(t+ext*model.t_obs)/T
    return 2*np.real(spline(u*n)*np.exp(2j*np.pi*q*u))

def stats(reference,candidate):
    a=np.stack([reference[c] for c in ('X','Y','Z')]);b=np.stack([candidate[c] for c in ('X','Y','Z')])
    aa=float(np.sum(a*a));bb=float(np.sum(b*b));ab=float(np.sum(a*b))
    scales=[float(a[i]@b[i]/(b[i]@b[i])) for i in range(3)]
    return dict(raw_relative_xyz_l2=float(np.linalg.norm(a-b)/np.sqrt(aa)),
        coherent_real_xyz_overlap=ab/np.sqrt(aa*bb),
        candidate_to_reference_norm=np.sqrt(bb/aa),
        diagnostic_common_real_scale=ab/bb,
        diagnostic_per_channel_real_scales=scales,
        raw_relative_channel_l2={c:float(np.linalg.norm(a[i]-b[i])/np.linalg.norm(a[i])) for i,c in enumerate(('X','Y','Z'))})

def receiver_labelled_equalarm_xyz(model,params,link_fd,generation,sign=-1):
    """Explicit TDI of receiver-labelled links for a constant equal-arm model.

    X1=(D^2-1)[y12+D*y21-y13-D*y31], cyclic for Y/Z;
    X2=(1-D^4)X1. This is a diagnostic convention bridge, not a replacement
    for the package's native get_tdi or unequal-arm moving-delay TDI.
    """
    frequency=np.asarray(model.get_frequency_grid(model.get_kmin(params[:,0])))[0]
    D=np.exp(-2j*np.pi*frequency*float(np.mean(np.asarray(model.arm_length)))/C_M_PER_S)
    y=dict(zip(('12','23','31','13','32','21'),link_fd*sign))
    result=np.array([(D*D-1)*(y[a]+D*y[b]-y[c]-D*y[d])
        for a,b,c,d in [('12','21','13','31'),('23','32','21','12'),('31','13','32','23')]])
    return result*(1-D**4) if generation==2 else result

def run(out,duration,dt,frequencies,orbit_kind,nvalues,generation,repeats,jaxgb_sign=1):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Preserve the existing result; choose a fresh output')
    shutil.copy2(__file__,out/'jaxgb_check_source.py')
    orbit=StaticOrbits() if orbit_kind=='static' else default_lisaorbits('esa-trailing')
    t=np.arange(round(duration/dt))*dt
    halo=int(np.ceil(8*12/dt))+4*102
    padded=np.arange(-halo,t.size+halo)*dt
    tick=time.perf_counter();state=state_from_lisaorbits(orbit,padded)
    geometry=precompute_jax_link_geometry(state,orbits=orbit)
    operator=PreparedGWTDIMichelson(state,link_delays_from_lisaorbits(orbit,padded),generation=generation,measurement_order=101,delay_order=7)
    prepared_s=time.perf_counter()-tick
    results=[]
    for f in frequencies:
        source=EccentricBinaryParams(mean_motion=np.pi*f,eccentricity=0.,
            m1_solar=.5,m2_solar=.5,distance_m=KILOPARSEC_M,
            inclination=1.,beta=.3,lambda_=-.7,psi=.4,phi0=.2)
        def native(physics='newtonian'):
            links=eccentric_links_jax(source,geometry,batch_size=1,physics_mode=physics,evolution_mode='fixed')
            return {c:np.asarray(x)[halo:-halo].copy() for c,x in operator(links).items()}
        tick=time.perf_counter();reference=native();cold=time.perf_counter()-tick
        native_times=[]
        for _ in range(repeats):
            tick=time.perf_counter();again=native();native_times.append(time.perf_counter()-tick)
            if not all(np.array_equal(again[c],reference[c]) for c in reference):raise ValueError('Native repeated output changed')
        pn=native('1pn');pn_stats=stats(reference,pn);del pn,again
        native_links=eccentric_links_jax(source,geometry,batch_size=1,physics_mode='newtonian',evolution_mode='fixed')
        link_labels=('12','23','31','13','32','21')
        native_link_array=np.stack([native_links[c][halo:-halo].real for c in link_labels])
        for kind,cls in [('JaxGB',JaxGB),('JaxGBaccurate',JaxGBaccurate)]:
            previous=None
            for n in nvalues:
                tick=time.perf_counter();model=cls(orbits=EclipticOrbits(orbit),t_obs=duration,n=n,t0=0.)
                model_setup=time.perf_counter()-tick;p=parameters(source)
                def evaluate():
                    result=model.get_tdi(p,tdi_generation=1.5 if generation==1 else 2,tdi_combination='XYZ')
                    jax.block_until_ready(result);return result
                tick=time.perf_counter();evaluate();jax_cold=time.perf_counter()-tick
                warm=[]
                for _ in range(repeats):
                    tick=time.perf_counter();fd=evaluate();warm.append(time.perf_counter()-tick)
                arrays=np.stack([np.asarray(a[0]) for a in fd])
                tick=time.perf_counter();dense=reconstruct(model,source,arrays,t);conversion=time.perf_counter()-tick
                raw_metrics=stats(reference,dict(zip(('X','Y','Z'),dense)))
                dense=dense*jaxgb_sign
                candidate=dict(zip(('X','Y','Z'),dense))
                metrics=stats(reference,candidate)
                link_fd=np.asarray(model.get_link_responses(p)[0]);link_dense=reconstruct(model,source,link_fd,t)
                link_norm=np.linalg.norm(native_link_array)
                link_metrics=dict(raw_relative_l2=float(np.linalg.norm(link_dense-native_link_array)/link_norm),
                    sign_reversed_relative_l2=float(np.linalg.norm(-link_dense-native_link_array)/link_norm),
                    same_link_real_correlations=[float(a@b/np.linalg.norm(a)/np.linalg.norm(b)) for a,b in zip(native_link_array,link_dense)])
                # Reuse the exact SAME native prepared TDI on JAXGB's links.
                # This isolates link agreement from JAXGB's native TDI formula.
                padded_jax_links=reconstruct(model,source,link_fd,padded)*jaxgb_sign
                shared_tdi={c:np.asarray(x)[halo:-halo] for c,x in operator(dict(zip(link_labels,padded_jax_links))).items()}
                shared_metrics=stats(reference,shared_tdi)
                receiver_metrics=None
                if kind=='JaxGBaccurate':
                    receiver_fd=receiver_labelled_equalarm_xyz(model,p,link_fd,generation,jaxgb_sign)
                    receiver_td=reconstruct(model,source,receiver_fd,t)
                    receiver_metrics=stats(reference,dict(zip(('X','Y','Z'),receiver_td)))
                if previous is not None:metrics['successive_n_relative_xyz_l2']=float(np.linalg.norm(dense-previous)/np.linalg.norm(dense))
                previous=dense
                row=dict(f0_hz=f,source=asdict(source),model=kind,n=n,metrics=metrics,
                    raw_native_jaxgb_metrics=raw_metrics,jaxgb_to_egb_response_sign=jaxgb_sign,
                    six_link_diagnostics=link_metrics,jaxgb_links_through_shared_native_tdi=shared_metrics,
                    explicit_receiver_labelled_equalarm_tdi=receiver_metrics,
                    timing=dict(egb_shared_geometry_tdi_preparation_s=prepared_s,
                        egb_first_evaluation_s=cold,egb_warm_evaluation_s=native_times,
                        jaxgb_setup_s=model_setup,jaxgb_first_call_s=jax_cold,
                        jaxgb_warm_sparse_fd_s=warm,jaxgb_fd_to_dense_real_td_s=conversion),
                    egb_1pn_vs_newtonian=pn_stats)
                results.append(row);atomic(out/'progress.json',dict(status='running',completed=len(results),latest=row))
                print(json.dumps(dict(f0=f,model=kind,n=n,**metrics)),flush=True)
    result=dict(status='complete',duration_s=duration,dt_s=dt,orbit=orbit_kind,
        generation=generation,measurement_order=101,delay_order=7,results=results,
        jaxgb_to_egb_response_sign=jaxgb_sign,
        versions={p:importlib.metadata.version(p) for p in ['jaxgb','jax','jaxlib','numpy','scipy','pytdi','lisaorbits']},
        executable=sys.executable,platform=platform.platform(),jax_devices=[str(x) for x in jax.devices()],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sorted((ROOT/'src/egb_multi').glob('*.py'))]},
        contracts=dict(frame='BME J2000; coordinate-only orbit wrapper for JAXGB',
            source='fixed Newtonian eccentric source at e=0; shared physical amplitude and mapped orbital/GW phase',
            representation='physical real XYZ; JAXGB integrated one-sided spectrum reconstructed on its actual grid',
            calibration='none; fitted real scales are diagnostic only',
            timing='native sparse and dense-conversion costs separate; no raw speed ratio treated as matched accuracy',
            limits='JAXGB equal-arm TDI transform and moving-response approximations retained, not silently repaired'),
        waveform_arrays_saved=False)
    atomic(out/'result.json',result);atomic(out/'progress.json',dict(status='complete',completed=len(results)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--duration',type=float,default=604800.);p.add_argument('--dt',type=float,default=2.5)
    p.add_argument('--frequencies',type=float,nargs='+',default=[.001,.003,.01])
    p.add_argument('--orbit',choices=['static','moving'],default='static')
    p.add_argument('--n',type=int,nargs='+',default=[256,512,1024]);p.add_argument('--generation',type=int,choices=[1,2],default=1)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--jaxgb-sign',type=int,choices=[-1,1],default=1);a=p.parse_args()
    run(a.output,a.duration,a.dt,a.frequencies,a.orbit,a.n,a.generation,a.repeats,a.jaxgb_sign)
