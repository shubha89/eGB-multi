"""Annual native injections and finite-window weighting diagnostics for 2.8.

Creates reusable clean annual signals and deterministic secondary covariance.
Physical orientation is maximized for a declared harmonic candidate pool;
these scores are not full intrinsic fitting factors. The rectangular-window
periodogram diagonal is a sensitivity control, not an adopted likelihood:
it omits finite-observation frequency-bin covariance.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"outputs/referee/.mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT/"outputs/referee/.cache"))

import numpy as np
from egb_multi.comparison import AEPSD, AESeries, AEInnerProduct, TDIConvention
from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.circular_fit import CircularAmplitudeBasis, profile_circular_orientation
from egb_multi.eccentric import EccentricBinaryParams, peters_mathews_derivatives
from egb_multi.external import default_lisaorbits, link_delays_from_lisaorbits
from egb_multi.native_validation import ChunkedNativeResponse
from egb_multi.spectral_window import expected_windowed_covariance
from referee_task_2_covariance_grid import chunked_model
from referee_task_2_frozen_noise_audit import instrument
from referee_task_2_objective_orientation import assert_local, write_json


def covariance(out, args, orbit):
    if args.covariance_inputs:
        inputs=args.covariance_inputs.resolve()
        if not inputs.is_relative_to(ROOT):raise ValueError("Use local covariance cache")
        manifest=json.loads((inputs/"manifest.json").read_text())
        config=manifest["configuration"]
        if any(config[key]!=getattr(args,key) for key in ("duration","dt","cutoff_days")):
            raise ValueError("Cached covariance observation/noise configuration differs")
        for name in ("annual_covariance.npz","annual_rectangular_diagonal.npz","annual_window_ratios.json"):
            shutil.copy2(inputs/name,out/name)
        with np.load(out/"annual_covariance.npz") as data:f=data["frequency_hz"];values=data["covariance"]
        with np.load(out/"annual_rectangular_diagonal.npz") as data:annual_f=data["frequency_hz"];diagonal=[data["PSD_A"],data["PSD_E"]]
        return f,values,annual_f,diagonal
    duration, fs = args.duration, 1/args.dt
    length = int(round(duration/args.dt))
    size = 1048576
    uniform = np.fft.rfftfreq(size, args.dt)
    low = np.geomspace(1/(100*duration), 100*uniform[1], 1024)
    f = np.unique(np.r_[uniform, low])
    epochs = (np.arange(16)+.5)*duration/16
    delays = link_delays_from_lisaorbits(orbit, epochs)
    print("Annual secondary covariance", len(f), "frequencies", flush=True)
    values = np.zeros((len(f),2,2), complex)
    values[1:] = chunked_model(f[1:], instrument(1/(args.cutoff_days*86400), .03125), delays)
    np.savez_compressed(out/"annual_covariance.npz", frequency_hz=f, covariance=values)
    integral_size = 1 << int(np.ceil(np.log2(2*length)))
    refined_frequency = np.fft.rfftfreq(integral_size, args.dt)
    diagonal = []
    for channel in range(2):
        spectrum = np.interp(refined_frequency, f, values[:,channel,channel].real)
        spectrum[-1] *= .5
        _, expectation = expected_windowed_covariance(spectrum[:,None,None], fs, length, window="boxcar", detrend=False)
        diagonal.append(expectation[:,0,0].real)
        del spectrum, expectation
        gc.collect()
    year_frequency = np.fft.rfftfreq(length, args.dt)
    np.savez_compressed(out/"annual_rectangular_diagonal.npz", frequency_hz=year_frequency, PSD_A=diagonal[0], PSD_E=diagonal[1])
    mask=(year_frequency>=2e-5)&(year_frequency<=.04)
    rows=[]
    for channel, name in enumerate(("A","E")):
        point = np.interp(year_frequency[mask], f, values[:,channel,channel].real)
        ratio = diagonal[channel][mask]/point
        rows.append(dict(channel=name, maximum_ratio=float(np.max(ratio)), minimum_ratio=float(np.min(ratio)),
            maximum_ratio_frequency_hz=float(year_frequency[mask][np.argmax(ratio)])))
    write_json(out/"annual_window_ratios.json", dict(integral_grid_samples=integral_size, ratios=rows,
        limitations=["Integral-grid and PSD-grid refinement remain required", "Periodogram diagonal does not include frequency-bin covariance"]))
    return f, values, year_frequency, diagonal


def run(args):
    assert_local()
    out=args.output.resolve()
    if not out.is_relative_to(ROOT): raise ValueError("Use this checkout")
    out.mkdir(parents=True, exist_ok=False)
    hashes={}
    sources=[*list((ROOT/"src/egb_multi").glob("*.py")),Path(__file__),ROOT/"benchmarks/referee_task_2_covariance_grid.py",
        ROOT/"benchmarks/referee_task_2_frozen_noise_audit.py",ROOT/"benchmarks/referee_task_2_objective_orientation.py"]
    for path in sources:
        rel=path.relative_to(ROOT);target=out/"snapshot"/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    input_hashes={}
    if args.case_definitions:
        path=args.case_definitions.resolve()
        if not path.is_relative_to(ROOT):raise ValueError("Use local case definitions")
        input_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    if args.covariance_inputs:
        for name in ("manifest.json","annual_covariance.npz","annual_rectangular_diagonal.npz","annual_window_ratios.json"):
            path=args.covariance_inputs.resolve()/name;input_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/"manifest.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),sha256=hashes,inputs=input_hashes,
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scope="Annual weighting/candidate-pool diagnostic, not optimized FF certification"))
    convention=TDIConvention(2,args.order,7,"analytic_esa_trailing_BME_J2000","native_analytic_retarded_delays")
    orbit=default_lisaorbits("esa-trailing")
    f, values, annual_f, windowed=covariance(out,args,orbit)
    provenance=dict(sidedness="one_sided",scaling="density",noise_model="native_annual_secondary_diagnostic",
        CSD_convention="S_AE=E[tilde_A*conj(tilde_E)]", production_certified=False)
    point=AEPSD(f,{ch:values[:,i,i].real for i,ch in enumerate(("A","E"))},convention,provenance,values[:,0,1])
    rectangular=AEPSD(annual_f,dict(zip(("A","E"),windowed)),convention,
        dict(provenance,noise_model="native_annual_rectangular_periodogram_diagonal_diagnostic"))
    length=int(round(args.duration/args.dt));times=np.arange(length)*args.dt
    if length*args.dt!=args.duration:raise ValueError("Preserve the exact physical duration")
    definitions={"circular":(.001,0.,1.12,.32,.91,.43,6.08,"fixed"),
        "near_contour":(.001,.1,.8,.6,1.21,.2,-1.,"peters_mathews"),
        "contour_e008":(.001,.08,.8,.6,1.21,.2,-1.,"peters_mathews"),
        "contour_e0088":(.001,.088,.8,.6,1.21,.2,-1.,"peters_mathews"),
        "contour_e009":(.001,.09,.8,.6,1.21,.2,-1.,"peters_mathews"),
        "high_e":(.001,.6,.8,.6,1.21,.2,-1.,"peters_mathews"),
        "high_pm":(.003,.5,.8,.6,.8,.2,-1.,"peters_mathews")}
    if args.case_definitions:
        supplied=json.loads(args.case_definitions.read_text())
        fields=("frequency_hz","eccentricity","m1_solar","m2_solar","inclination_rad",
                "beta_rad","longitude_rad","evolution")
        for name,definition in supplied.items():
            if set(definition)!=set(fields):raise ValueError("Explicit case definition fields required: "+name)
            definitions[name]=tuple(definition[key] for key in fields)
    envelope=InterpolatedCircularTDI(orbit,times,convention,envelope_intervals=args.envelope_intervals)
    native=ChunkedNativeResponse(orbit,times,convention,block_samples=65536,cached_blocks=2)
    records=[]
    for name in args.cases:
        started=time.perf_counter()
        frequency,e,m1,m2,inc,beta,longitude,evolution=definitions[name]
        source=EccentricBinaryParams(mean_motion=np.pi*frequency,eccentricity=e,m1_solar=m1,m2_solar=m2,
            inclination=inc,beta=beta,lambda_=longitude,phi0=.617,psi=.71,t0=0.)
        if name=="circular":source=replace(source,mean_motion=source.mean_motion+np.pi*.713/args.duration,fdot=-.517/args.duration**2)
        print("Annual native injection",name,len(times),flush=True)
        signal=native(source,evolution)
        np.savez_compressed(out/f"{name}_signal.npz",A=signal.channels["A"],E=signal.channels["E"])
        write_json(out/f"{name}_source.json",dict(source=asdict(source),evolution=evolution,convention=asdict(convention),
            duration_s=args.duration,dt_s=args.dt,signal_sha256=hashlib.sha256((out/f"{name}_signal.npz").read_bytes()).hexdigest()))
        metrics={"pointwise_stationary":AEInnerProduct(signal,point,(2e-5,.04)),
            "rectangular_diagonal":AEInnerProduct(signal,rectangular,(2e-5,.04))}
        candidates=[]
        for harmonic in ([2] if name=="circular" else args.harmonics):
            template=replace(source,eccentricity=0.,mean_motion=source.mean_motion*harmonic/2,t0=args.duration/2)
            if name=="circular":
                template=replace(template,mean_motion=source.mean_motion+np.pi*source.fdot*template.t0,
                    phi0=source.phi0+source.mean_motion*template.t0+.5*np.pi*source.fdot*template.t0**2)
            else:
                fdot=float(peters_mathews_derivatives(source.mean_motion,source.eccentricity,source.total_mass_kg,source.symmetric_mass_ratio)[0]/np.pi)*harmonic/2
                template=replace(template,mean_motion=template.mean_motion+np.pi*fdot*template.t0,fdot=fdot)
            basis=CircularAmplitudeBasis.generate(template,envelope)
            scores={label:profile_circular_orientation(metric,basis)["overlap"] for label,metric in metrics.items()}
            candidates.append(dict(harmonic=harmonic,source=asdict(template),scores=scores))
            print(name,"harmonic",harmonic,scores,flush=True)
        maxima={label:max(c["scores"][label] for c in candidates) for label in metrics}
        records.append(dict(case=name,source=asdict(source),evolution=evolution,candidates=candidates,
            maxima=maxima,window_diagonal_minus_pointwise=maxima["rectangular_diagonal"]-maxima["pointwise_stationary"],wall_s=time.perf_counter()-started))
        write_json(out/"annual_weighting_comparison.json",records)
        del signal, metrics, basis
        gc.collect()
    write_json(out/"validation_summary.json",dict(status="annual_candidate_pool_diagnostic_completed",production_certified=False,
        cases=len(records),duration_s=args.duration,
        limitations=["No intrinsic reoptimization; harmonic candidate pool only", "True moving/locking laser residual is not in this covariance",
            "Rectangular periodogram diagonal omits cross-frequency covariance", "Final covariance/response refinement gates remain required"]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--duration",type=float,default=31557600.)
    p.add_argument("--dt",type=float,default=10.)
    p.add_argument("--order",type=int,default=81)
    p.add_argument("--cutoff-days",type=float,default=365.25)
    p.add_argument("--envelope-intervals",type=int,default=8192)
    p.add_argument("--cases",nargs="+",default=["circular","near_contour","high_e","high_pm"])
    p.add_argument("--harmonics",type=int,nargs="+",default=[1,2,3,5,7,10])
    p.add_argument("--covariance-inputs",type=Path)
    p.add_argument("--case-definitions",type=Path,help="Additional explicit annual stress cases; defaults are unchanged")
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        if args.output.exists() and not (args.output/"validation_summary.json").exists():
            write_json(args.output/"validation_summary.json",dict(status="failed_annual_cache_generation",
                production_certified=False,exception_type=type(exc).__name__,exception=str(exc)))
        raise
