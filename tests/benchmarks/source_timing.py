"""Prepared fixed-source JAX timing with a separate NumPy physics reference.

No Peters--Mathews evolution: this complements, rather than replaces, the
evolving NumPy source used in the pyEFPE common-physics checks.
"""
from pathlib import Path
import argparse,gc,json,sys,time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests/comparisons')]
from comparison_utils import atomic, pair_stats
from egb_multi.eccentric import EccentricBinaryParams, eccentric_polarizations
from egb_multi.eccentric_jax import eccentric_complex_strain_batch_jax

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Preserve prior evidence')
    t=np.arange(12623040)*2.5;rows=[]
    for f,e in [(.001,0.),(.001,.3),(.003,.3),(.01,.6)]:
        s=EccentricBinaryParams(mean_motion=np.pi*f,eccentricity=e,
            m1_solar=.8,m2_solar=.6,inclination=.8,phi0=.2)
        def call():return eccentric_complex_strain_batch_jax(s,t,physics_mode='1pn')[0]
        tick=time.perf_counter();h=call();cold=time.perf_counter()-tick
        warm=[]
        for _ in range(3):
            tick=time.perf_counter();h2=call();warm.append(time.perf_counter()-tick)
            if not np.array_equal(h,h2):raise ValueError('Repeated JAX output changed')
            del h2
        tick=time.perf_counter();ref=np.array(eccentric_polarizations(s,t,physics_mode='1pn',evolution_mode='fixed'))
        numpy_s=time.perf_counter()-tick
        stats=pair_stats(ref,np.array([h.real,-h.imag]))
        if stats['raw_relative_l2']>1e-7:raise ValueError('JAX/NumPy fixed-source control failed')
        rows.append(dict(f0_hz=f,eccentricity=e,jax_first_call_s=cold,jax_warmed_s=warm,
            numpy_reference_s=numpy_s,comparison=stats))
        atomic(out/'progress.json',dict(status='running',completed=len(rows),latest=rows[-1]))
        del ref,h;gc.collect()
    atomic(out/'result.json',dict(status='complete',duration_s=31557600.,dt_s=2.5,rows=rows,
        contract='1PN fixed intrinsic source, hplus and hcross, no LISA/TDI or PM evolution; JAX timings include device-to-host conversion',
        waveform_arrays_saved=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
