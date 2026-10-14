"""Validate full-band spline-Fourier preparation against sampled annual FFTs."""

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/referee/.mplconfig"));os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/"outputs/referee/.cache"))
import numpy as np
from egb_multi.comparison import AESeries,AEPSD,AEInnerProduct,TDIConvention
from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.circular_spectral import CircularSpectralTDI
from egb_multi.circular_fit import CircularAmplitudeBasis,profile_circular_orientation
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits
from referee_task_2_objective_orientation import assert_local,write_json


def run(args):
    assert_local();out=args.output.resolve();inputs=args.inputs.resolve()
    if not out.is_relative_to(ROOT) or not inputs.is_relative_to(ROOT):raise ValueError("Use local files")
    out.mkdir(exist_ok=False,parents=True);hashes={}
    for path in [*list((ROOT/"src/egb_multi").glob("*.py")),Path(__file__),ROOT/"benchmarks/referee_task_2_objective_orientation.py"]:
        rel=path.relative_to(ROOT);target=out/"snapshot"/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    input_files=[inputs/name for name in ("circular_source.json","circular_signal.npz","annual_covariance.npz")]
    write_json(out/"manifest.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),sha256=hashes,
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files},
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        weighted_basis_error_allowance=1e-7,
        absolute_profile_difference_allowance=1e-8,production_certified=False,scope="Selected annual spectral-preparation control"))
    meta=json.loads((inputs/"circular_source.json").read_text());convention=TDIConvention(**meta["convention"])
    duration=meta["duration_s"];dt=meta["dt_s"];times=np.arange(round(duration/dt))*dt
    with np.load(inputs/"circular_signal.npz") as data:signal=AESeries(times,{ch:data[ch] for ch in ("A","E")},convention)
    with np.load(inputs/"annual_covariance.npz") as data:f=data["frequency_hz"];cov=data["covariance"]
    psd=AEPSD(f,{ch:cov[:,i,i].real for i,ch in enumerate(("A","E"))},convention,
        dict(sidedness="one_sided",scaling="density",noise_model="native_annual_secondary_diagnostic",CSD_convention="S_AE=E[tilde_A*conj(tilde_E)]"),cov[:,0,1])
    metric=AEInnerProduct(signal,psd,(2e-5,.04));orbit=default_lisaorbits("esa-trailing")
    response=InterpolatedCircularTDI(orbit,times,convention,envelope_intervals=8192);spectral=CircularSpectralTDI(response,metric)
    records=[];failures=[]
    definitions=[("low",.0001,.5,.5,.713,-.517),("carrier",.003,1.4,.1,.713,16.),("near_TDI_null",.03,1.4,1.4,.713,-16.)]
    for name,frequency,m1,m2,alpha,gamma in definitions:
        source=EccentricBinaryParams(mean_motion=np.pi*(frequency+alpha/duration),eccentricity=0.,m1_solar=m1,m2_solar=m2,
            t0=(times[0]+times[-1])/2,fdot=gamma/duration**2,beta=.43,lambda_=6.08,inclination=.91,phi0=.77,psi=.46)
        start=time.perf_counter();basis=CircularAmplitudeBasis.generate(source,response)
        standard=np.array([metric.prepare(v) for v in basis.responses]);sampled_time=time.perf_counter()-start
        sampled_profile=profile_circular_orientation(metric,basis)
        del basis;response.cache_key=response.cache_basis=None;gc.collect()
        start=time.perf_counter();prepared=spectral.prepared_basis(source);spectral_time=time.perf_counter()-start
        approximate_profile=profile_circular_orientation(metric,prepared)
        scale=max(np.linalg.norm(row) for row in standard[:4])
        error=max(np.linalg.norm(a-b)/scale for a,b in zip(standard,prepared.prepared))
        difference=approximate_profile["overlap"]-sampled_profile["overlap"]
        row=dict(case=name,source=asdict(source),maximum_weighted_basis_error_relative_to_oscillatory_scale=float(error),
            absolute_profile_difference=float(abs(difference)),sampled_prepare_wall_s=sampled_time,spectral_prepare_wall_s=spectral_time,
            sampled_overlap=sampled_profile["overlap"],spectral_overlap=approximate_profile["overlap"])
        records.append(row);write_json(out/"spectral_validation.json",records);print(row,flush=True)
        if error>1e-7 or abs(difference)>1e-8:failures.append(name+": spectral objective disagreement")
        del standard,prepared;gc.collect()
    write_json(out/"validation_summary.json",dict(status="passed_selected_annual_spectral_objective" if not failures else "failed_spectral_objective_control",
        failures=failures,production_certified=False,duration_s=duration,
        limitations=["Selected sources only; no uniform family accuracy inferred", "Envelope/native error is a separate control",
            "Internal cubic derivative-jump sampling correction remains approximated", "Winning candidates require native verification"]))
    if failures:raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inputs",type=Path,default=ROOT/"outputs/referee/task_2_8_20261008_annual_weighting_pilot");run(p.parse_args())
