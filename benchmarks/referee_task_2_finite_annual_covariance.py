"""Exact finite stationary covariance versus annual pointwise PSD weighting.

Uses the cached annual native injection and one retained physical candidate.
This quantifies finite covariance at fixed intrinsic parameters; it is not an
intrinsic fitting-factor certification or a moving-noise likelihood.
"""

import argparse
from dataclasses import replace
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/referee/.mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/"outputs/referee/.cache"))

import numpy as np
from egb_multi.comparison import AESeries,AEPSD,AEInnerProduct,TDIConvention
from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.circular_fit import CircularAmplitudeBasis,profile_circular_orientation
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits
from egb_multi.finite_covariance import FiniteWindowAEInnerProduct
from referee_task_2_objective_orientation import assert_local,write_json


def run(args):
    assert_local();out=args.output.resolve();inputs=args.inputs.resolve()
    if not out.is_relative_to(ROOT) or not inputs.is_relative_to(ROOT):raise ValueError("Use this checkout")
    if args.laser_covariance:
        args.laser_covariance=args.laser_covariance.resolve()
        if not args.laser_covariance.is_relative_to(ROOT):raise ValueError("Use local laser covariance")
    if args.candidate_run:
        args.candidate_run=args.candidate_run.resolve()
        if not args.candidate_run.is_relative_to(ROOT):raise ValueError("Use a local completed candidate run")
    out.mkdir(parents=True,exist_ok=False);hashes={}
    for path in [*list((ROOT/"src/egb_multi").glob("*.py")),Path(__file__),ROOT/"benchmarks/referee_task_2_objective_orientation.py"]:
        rel=path.relative_to(ROOT);target=out/"snapshot"/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    input_files=[inputs/"annual_covariance.npz",inputs/"annual_weighting_comparison.json",inputs/f"{args.case}_source.json",inputs/f"{args.case}_signal.npz"]
    if args.laser_covariance:input_files.append(args.laser_covariance)
    if args.candidate_run:input_files.append(args.candidate_run/f"{args.case}_annual_search.json")
    write_json(out/"manifest.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),sha256=hashes,
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files},
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scope="Finite stationary annual covariance, one retained intrinsic candidate"))
    meta=json.loads((inputs/f"{args.case}_source.json").read_text());dt=meta["dt_s"];duration=meta["duration_s"]
    convention=TDIConvention(**meta["convention"]);times=np.arange(round(duration/dt))*dt
    with np.load(inputs/f"{args.case}_signal.npz") as data:
        signal=AESeries(times,{ch:data[ch] for ch in ("A","E")},convention)
    record=next(r for r in json.loads((inputs/"annual_weighting_comparison.json").read_text()) if r["case"]==args.case)
    if args.candidate_run:
        completed=json.loads((args.candidate_run/f"{args.case}_annual_search.json").read_text())
        selected=completed["refinement"]["best"]
    else:selected=max(record["candidates"],key=lambda c:c["scores"]["pointwise_stationary"])
    template=replace(EccentricBinaryParams(**selected["source"]),phi0=0.)
    with np.load(inputs/"annual_covariance.npz") as data:
        frequency=data["frequency_hz"];covariance=data["covariance"]
    laser_frequency=laser_covariance=None
    if args.laser_covariance:
        with np.load(args.laser_covariance) as data:
            laser_frequency=data["frequency_hz"];laser_covariance=data["covariance"]
        if laser_frequency[-1]<.5/dt-8*np.spacing(.5/dt):raise ValueError("Laser covariance must cover the complete sampled-noise Nyquist interval")
        covariance=covariance.copy()
        for a in range(2):
            for b in range(2):covariance[:,a,b]+=np.interp(frequency,laser_frequency,laser_covariance[:,a,b])
        # A real-process DC endpoint uses the half-density limiting real value.
        # The lowest laser-symbol frequency is 1/(100 years); refine this
        # infrared endpoint independently before production adoption.
        covariance[0]=covariance[0].real*.5
    provenance=dict(sidedness="one_sided",scaling="density",noise_model="native_annual_secondary_diagnostic",
        CSD_convention="S_AE=E[tilde_A*conj(tilde_E)]",production_certified=False)
    psd=AEPSD(frequency,{ch:covariance[:,i,i].real for i,ch in enumerate(("A","E"))},convention,provenance,covariance[:,0,1])
    point=AEInnerProduct(signal,psd,tuple(args.band))
    response=InterpolatedCircularTDI(default_lisaorbits("esa-trailing"),times,convention,envelope_intervals=args.envelope_intervals)
    basis=CircularAmplitudeBasis.generate(template,response)
    profile=profile_circular_orientation(point,basis)
    physical=basis.reconstruct(inclination=profile["inclination_rad"],psi=profile["polarization_rad"],phi0=profile["orbital_phase_rad"])
    write_json(out/"pointwise_candidate.json",dict(source=selected["source"],profile=profile))
    del basis,response;gc.collect()
    size=args.integral_samples or 1<<int(np.ceil(np.log2(2*len(times))))
    if size<2*len(times) or size%2:raise ValueError("Even spectral integral grid >= twice observation required")
    f=np.fft.rfftfreq(size,dt);spectral=np.empty((len(f),2,2),complex)
    for a in range(2):
        for b in range(2):spectral[:,a,b]=np.interp(f,frequency,covariance[:,a,b])
    spectral[-1]=spectral[-1].real*.5
    started=time.perf_counter()
    def progress(solve,iteration):
        if iteration%10==0:print("solve",solve,"iteration",iteration,"elapsed_s",round(time.perf_counter()-started,1),flush=True)
    try:
        metric=FiniteWindowAEInnerProduct(signal,psd,tuple(args.band),spectral,rtol=args.rtol,maxiter=args.maxiter,progress=progress)
        del spectral;gc.collect()
        comparison=metric.compare(physical)
        print("Finite physical candidate",comparison["overlap"],"point",profile["overlap"],flush=True)
        write_json(out/"fixed_candidate_comparison.json",dict(pointwise_overlap=profile["overlap"],finite_covariance=comparison,
            delta_overlap=comparison["overlap"]-profile["overlap"],solve_records=metric.solve_records,wall_s=time.perf_counter()-started))
        if args.profile_orientation:
            response=InterpolatedCircularTDI(default_lisaorbits("esa-trailing"),times,convention,envelope_intervals=args.envelope_intervals)
            basis=CircularAmplitudeBasis.generate(template,response)
            finite_profile=profile_circular_orientation(metric,basis)
            write_json(out/"finite_orientation_profile.json",dict(profile=finite_profile,delta_overlap=finite_profile["overlap"]-profile["overlap"],solve_records=metric.solve_records))
        write_json(out/"validation_summary.json",dict(status="finite_covariance_candidate_diagnostic_completed",production_certified=False,
            integral_grid_samples=size,rtol=args.rtol,wall_s=time.perf_counter()-started,
            limitations=["Stationary annual-averaged covariance, not moving noise", "No intrinsic reoptimization",
                "Input spectrum and integral-grid refinements remain required"]))
    except Exception as exc:
        write_json(out/"validation_summary.json",dict(status="failed_finite_covariance_control",production_certified=False,
            exception_type=type(exc).__name__,exception=str(exc),integral_grid_samples=size,rtol=args.rtol,wall_s=time.perf_counter()-started))
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inputs",type=Path,default=ROOT/"outputs/referee/task_2_8_20261008_annual_weighting_pilot")
    p.add_argument("--case",default="near_contour")
    p.add_argument("--integral-samples",type=int,default=0)
    p.add_argument("--band",type=float,nargs=2,default=[2e-5,.04])
    p.add_argument("--envelope-intervals",type=int,default=8192)
    p.add_argument("--rtol",type=float,default=1e-7)
    p.add_argument("--maxiter",type=int,default=200)
    p.add_argument("--profile-orientation",action="store_true")
    p.add_argument("--laser-covariance",type=Path)
    p.add_argument("--candidate-run",type=Path,help="Use the physical winner of a completed annual baseline instead of the initial pool")
    args=p.parse_args()
    try:
        run(args)
    except Exception as exc:
        path=args.output/"validation_summary.json"
        if args.output.exists() and not path.exists():
            write_json(path,dict(status="failed_before_covariance_solve",production_certified=False,
                exception_type=type(exc).__name__,exception=str(exc)))
        raise
