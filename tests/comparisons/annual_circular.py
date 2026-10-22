"""Annual circular A/E benchmark, separately labelled from dense native XYZ.

Use eGB's existing circular-envelope evaluator and certify selected windows
against its native JAX links/TDI. JAXGB's sparse FD generation and dense-output
conversion are timed separately. No fitted channel scales or source arrays saved.
"""
from pathlib import Path
from dataclasses import asdict
import argparse,gc,json,sys,time,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import jax
from jaxgb.jaxgb import JaxGBaccurate
from jaxgb_check import atomic,parameters,reconstruct,EclipticOrbits,amplitude,receiver_labelled_equalarm_xyz
from egb_multi.eccentric import EccentricBinaryParams,eccentric_dynamics
from egb_multi.constants import G_M3_KG_S2,C_M_PER_S
from egb_multi.external import default_lisaorbits
from egb_multi.comparison import AESeries,AEInnerProduct,TDIConvention
from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.native_validation import ChunkedNativeResponse
from egb_multi.noise import aet_from_xyz
from comparison_utils import make_psd

def metrics(a,b,metric):
    aa=metric.prepare(a);bb=metric.prepare(b)
    n=metric.inner(aa,aa);m=metric.inner(bb,bb);z=metric.inner(aa,bb)
    diff=metric.inner(aa-bb,aa-bb)
    raw=sum(np.sum((a.channels[c]-b.channels[c])**2) for c in ('A','E'))
    norm=sum(np.sum(a.channels[c]**2) for c in ('A','E'))
    return dict(full_band_covariance_overlap=float(z/np.sqrt(n*m)),
        covariance_relative_residual=float(np.sqrt(max(0,diff/n))),
        unweighted_relative_residual=float(np.sqrt(raw/norm)),norm_ratio=float(np.sqrt(m/n)))

def run(output,frequencies,repeats=3):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Preserve previous evidence')
    shutil.copy2(__file__,out/'annual_circular_source.py')
    cfg=json.loads((ROOT/'tests/paper/reference_data/outputs/referee/task_2_9_20261012_production/manifest.json').read_text())['frozen_configuration']
    conv=TDIConvention(**cfg['tdi_convention']);dt=cfg['waveform_dt_s'];T=cfg['duration_s']
    times=np.arange(round(T/dt))*dt;orbit=default_lisaorbits('esa-trailing')
    tick=time.perf_counter();evaluator=InterpolatedCircularTDI(orbit,times,conv,envelope_intervals=32768)
    geometry_s=time.perf_counter()-tick
    psd=make_psd(cfg);rows=[]
    for f in frequencies:
        s=EccentricBinaryParams(mean_motion=np.pi*f,eccentricity=0.,m1_solar=.5,m2_solar=.5,
            inclination=1.,beta=.3,lambda_=-.7,psi=.4,phi0=.2)
        def evaluate_egb():
            # Geometry is reused; force intrinsic waveform/basis evaluation.
            evaluator.cache_key=evaluator.cache_basis=None
            return evaluator(s)
        tick=time.perf_counter();reference=evaluate_egb();cold=time.perf_counter()-tick
        warm=[]
        for _ in range(repeats):
            tick=time.perf_counter();again=evaluate_egb();warm.append(time.perf_counter()-tick)
            if not all(np.array_equal(again.channels[c],reference.channels[c]) for c in ('A','E')):raise ValueError('Repeated envelope output changed')
        del again
        # Verify absolute-time windows throughout the annual response. This is
        # selected empirical certification, not a full-year uniform bound.
        controls=[]
        for start in np.linspace(0,len(times)-4096,5).astype(int):
            window=times[start:start+4096]
            native=ChunkedNativeResponse(orbit,window,conv,block_samples=4096,cached_blocks=1)(s,'fixed')
            x=np.stack([native.channels[c] for c in ('A','E')]);y=np.stack([reference.channels[c][start:start+4096] for c in ('A','E')])
            controls.append(dict(first_time_s=float(window[0]),samples=len(window),relative_ae_l2=float(np.linalg.norm(x-y)/np.linalg.norm(x))))
        metric=AEInnerProduct(reference,psd,cfg['analysis_band_hz'])
        if max(x['relative_ae_l2'] for x in controls)>2e-4:
            atomic(out/'failed_native_window_controls.json',controls)
            raise ValueError('Circular envelope native-window error exceeds 2e-4')
        d=eccentric_dynamics(s,np.array([0.]),physics_mode='1pn',evolution_mode='fixed')
        r=float(d['r'][0]);w=float(d['phidot'][0]);gm=G_M3_KG_S2*s.total_mass_kg
        amp1=gm*s.symmetric_mass_ratio/(C_M_PER_S**4*s.distance_m)*(gm/r+r*r*w*w)
        ratio=amp1/amplitude(s)
        for n in (512,1024):
            tick=time.perf_counter();model=JaxGBaccurate(orbits=EclipticOrbits(orbit),t_obs=T,t0=0,n=n)
            setup=time.perf_counter()-tick;p=parameters(s)
            def evaluate_jax():
                value=model.get_tdi(p,tdi_generation=2,tdi_combination='XYZ');jax.block_until_ready(value);return value
            tick=time.perf_counter();evaluate_jax();jcold=time.perf_counter()-tick
            jwarm=[]
            for _ in range(repeats):
                tick=time.perf_counter();fd=evaluate_jax();jwarm.append(time.perf_counter()-tick)
            tick=time.perf_counter();xyz=-reconstruct(model,s,np.stack([np.asarray(x[0]) for x in fd]),times)
            ae=aet_from_xyz(dict(zip(('X','Y','Z'),xyz)))
            candidate=AESeries(times,{c:ae[c] for c in ('A','E')},conv)
            dense_s=time.perf_counter()-tick
            newtonian_metrics=metrics(reference,candidate,metric)
            candidate1=AESeries(times,{c:candidate.channels[c]*ratio for c in ('A','E')},conv)
            matched_metrics=metrics(reference,candidate1,metric)
            tick=time.perf_counter();link_fd=np.asarray(model.get_link_responses(p)[0]);jax.block_until_ready(link_fd)
            link_s=time.perf_counter()-tick
            receiver_fd=receiver_labelled_equalarm_xyz(model,p,link_fd,2,-1)
            tick=time.perf_counter();receiver_xyz=reconstruct(model,s,receiver_fd,times)*ratio
            receiver_ae=aet_from_xyz(dict(zip(('X','Y','Z'),receiver_xyz)))
            receiver_series=AESeries(times,{c:receiver_ae[c] for c in ('A','E')},conv)
            receiver_dense_s=time.perf_counter()-tick
            receiver_metrics=metrics(reference,receiver_series,metric)
            row=dict(f0_hz=f,source=asdict(s),jaxgb_n=n,
                native_window_controls=controls,
                egb_1pn_vs_jaxgb_newtonian=newtonian_metrics,
                equal_circular_polarization_amplitude=matched_metrics,
                receiver_labelled_equalarm_tdi_bridge=receiver_metrics,
                analytic_1pn_to_newtonian_amplitude_ratio=ratio,
                timings=dict(egb_shared_geometry_s=geometry_s,egb_first_evaluation_s=cold,
                    egb_warm_recomputed_intrinsic_basis_and_dense_ae_s=warm,
                    jaxgb_setup_s=setup,jaxgb_first_call_s=jcold,
                    jaxgb_warm_sparse_xyz_s=jwarm,jaxgb_sparse_to_dense_ae_s=dense_s,
                    jaxgb_six_link_fd_call_s=link_s,receiver_bridge_sparse_to_dense_ae_s=receiver_dense_s))
            rows.append(row);atomic(out/'progress.json',dict(status='running',completed=len(rows),latest=row))
            print(json.dumps(dict(f0_hz=f,n=n,**matched_metrics)),flush=True)
            del candidate,candidate1,xyz,ae,receiver_xyz,receiver_ae,receiver_series;gc.collect()
        del metric,reference;gc.collect()
    atomic(out/'result.json',dict(status='complete',duration_s=T,dt_s=dt,rows=rows,
        contracts=dict(egb_path='existing 1PN circular-envelope A/E evaluator, 32768 intervals; intrinsic cache invalidated for timed calls',
            jaxgb_path='native JaxGBaccurate TDI2 sparse XYZ, explicit common Doppler sign, dense real A/E conversion',
            amplitude_mapping='computed circular 1PN coefficient, never fitted; raw Newtonian amplitude comparison also retained',
            accuracy='full frozen annual covariance/band; five native windows per frequency, not a uniform full-year bound',
            timing='serial CPU float64; setup, compilation, warmed sparse output and dense conversion reported separately'),
        waveform_arrays_saved=False))
    atomic(out/'progress.json',dict(status='complete',completed=len(rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frequencies',type=float,nargs='+',default=[.001,.003,.01]);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();run(a.output,a.frequencies,a.repeats)
