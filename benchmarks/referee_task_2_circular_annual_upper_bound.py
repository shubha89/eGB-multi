"""Certify recovered annual circular candidates against overlap <= 1.

This control needs no optimizer convergence flag or exhaustive sky coverage:
an independently proposed physical candidate with native overlap >=1-epsilon
places the circular-family FF in [1-epsilon,1] by Cauchy-Schwarz. It does not
prove parameter recovery, eccentric FF convergence or production certification.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/referee/.mplconfig"));os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/"outputs/referee/.cache"))
import numpy as np
from egb_multi.comparison import AESeries,AEPSD,AEInnerProduct,TDIConvention
from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.circular_fit import CircularAmplitudeBasis,profile_circular_orientation
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits
from egb_multi.native_validation import ChunkedNativeResponse
from referee_task_2_objective_orientation import assert_local,write_json


def run(args):
    assert_local();out=args.output.resolve();inputs=args.inputs.resolve()
    if not all(p.resolve().is_relative_to(ROOT) for p in [out,inputs,*args.search_runs]):raise ValueError("Use local files")
    out.mkdir(exist_ok=False,parents=True);hashes={}
    for path in [*list((ROOT/"src/egb_multi").glob("*.py")),Path(__file__),ROOT/"benchmarks/referee_task_2_objective_orientation.py"]:
        rel=path.relative_to(ROOT);target=out/"snapshot"/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    sources=[];input_hashes={}
    for index,folder in enumerate(args.search_runs):
        path=folder.resolve()/"circular_evaluations.jsonl";data=path.read_bytes()
        if not data.endswith(b"\n"):data=data[:data.rfind(b"\n")+1]
        (out/f"input_log_{index}.jsonl").write_bytes(data);input_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(data).hexdigest()
        records=[json.loads(line) for line in data.decode().splitlines()];best=max(records,key=lambda r:r["overlap"])
        manifest=json.loads((folder/"manifest.json").read_text())
        if manifest["configuration"]["cases"]!=["circular"]:raise ValueError("Dedicated independent circular controls required")
        sources.append(dict(run=str(folder.resolve().relative_to(ROOT)),best=best,evaluations=len(records)))
    write_json(out/"manifest.json",dict(sha256=hashes,input_log_prefix_sha256=input_hashes,
        circular_overlap_deficit_allowance=2e-7,scope="Native physical candidate plus mathematical upper bound",production_certified=False))
    meta=json.loads((inputs/"circular_source.json").read_text());convention=TDIConvention(**meta["convention"])
    times=np.arange(round(meta["duration_s"]/meta["dt_s"]))*meta["dt_s"]
    with np.load(inputs/"circular_signal.npz") as data:signal=AESeries(times,{ch:data[ch] for ch in ("A","E")},convention)
    with np.load(inputs/"annual_covariance.npz") as data:f=data["frequency_hz"];cov=data["covariance"]
    psd=AEPSD(f,{ch:cov[:,i,i].real for i,ch in enumerate(("A","E"))},convention,
        dict(sidedness="one_sided",scaling="density",noise_model="native_annual_secondary_diagnostic",CSD_convention="S_AE=E[tilde_A*conj(tilde_E)]"),cov[:,0,1])
    metric=AEInnerProduct(signal,psd,(2e-5,.04));orbit=default_lisaorbits("esa-trailing")
    native=ChunkedNativeResponse(orbit,times,convention,block_samples=65536,cached_blocks=2)
    response=InterpolatedCircularTDI(orbit,times,convention,envelope_intervals=8192)
    results=[];failures=[]
    for record in sources:
        source=EccentricBinaryParams(**record["best"]["source"])
        profile=profile_circular_orientation(metric,CircularAmplitudeBasis.generate(source,response))
        recovered=replace(source,inclination=profile["inclination_rad"],psi=profile["polarization_rad"],phi0=profile["orbital_phase_rad"])
        result=metric.compare(native(recovered));deficit=max(0.,1-result["overlap"])
        results.append(dict(record,native_metrics=result,profile=profile,FF_bracket=[result["overlap"],1.],
            native_overlap_deficit=deficit,parameter_recovery_claimed=False))
        if deficit>2e-7:failures.append(record["run"]+": circular native recovery")
        print(record["run"],"native overlap",result["overlap"],flush=True)
        write_json(out/"circular_annual_controls.json",results)
    write_json(out/"validation_summary.json",dict(status="passed_native_annual_circular_upper_bound" if not failures else "failed_native_circular_control",
        failures=failures,production_certified=False,controls=len(results),duration_s=meta["duration_s"],
        largest_native_overlap_deficit=max(r["native_overlap_deficit"] for r in results),
        limitation="Circular recovery only; eccentric optimizer, numerical/band/noise and contour gates remain"))
    if failures:raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inputs",type=Path,default=ROOT/"outputs/referee/task_2_8_20261008_annual_weighting_pilot")
    p.add_argument("--search-runs",type=Path,nargs="+",required=True);run(p.parse_args())
