"""Native public pyEFPE FD control over a complete inspiral, not a short stop.

This is a functionality/timing control; it is not a finite-observation eGB
Fourier-accuracy comparison. Keep nonfinite values visible in scalar evidence.
"""
from pathlib import Path
import argparse,json,time,warnings
import numpy as np
from pyefpe_check import parameters,pyEFPE,EccentricBinaryParams,atomic

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Preserve prior evidence')
    source=EccentricBinaryParams(mean_motion=np.pi*.003,eccentricity=.3,
        m1_solar=.8,m2_solar=.6,inclination=.8,phi0=.2)
    p=parameters(source,31557600.,6,1e-12);p['f22_end']=None
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');tick=time.perf_counter();model=pyEFPE.pyEFPE(p)
        setup=time.perf_counter()-tick
        # Fixed grid, declared independently of numerical interpolation edges.
        frequencies=np.geomspace(.0031,.1,4096)
        timings=[]
        for _ in range(4):
            tick=time.perf_counter();h=np.asarray(model.generate_waveform(frequencies));timings.append(time.perf_counter()-tick)
    bad=~np.all(np.isfinite(h),axis=0)
    atomic(out/'result.json',dict(status='complete',parameters=p,setup_s=setup,
        native_fd_first_s=timings[0],native_fd_warmed_s=timings[1:],
        frequency_grid=dict(kind='geomspace',minimum_hz=.0031,maximum_hz=.1,count=4096),
        nonfinite_frequency_count=int(bad.sum()),nonfinite_frequencies_hz=frequencies[bad].tolist(),
        nonzero_finite_values=int(np.count_nonzero(h[:,~bad])),
        warnings=sorted(set(str(w.message) for w in ws)),
        integration_span_s=float(model.sol.all_ts[-1]-model.sol.all_ts[0]),
        contract='Native full-inspiral FD functionality only; not a finite-year Fourier comparison',
        waveform_arrays_saved=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
