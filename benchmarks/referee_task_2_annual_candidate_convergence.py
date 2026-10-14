"""Native annual numerical screening of a retained intrinsic candidate.

Freeze an exact input log prefix or completed winner, keep physical epochs,
noise spectra and band fixed, then vary cadence/interpolation/envelope/ODE
controls. Physical orientation is reprofiled. Intrinsic reoptimization under
each setting is a separate required convergence gate, not implied here.
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/referee/.mplconfig"));os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/"outputs/referee/.cache"))
import numpy as np
from egb_multi.comparison import AESeries,AEPSD,AEInnerProduct,TDIConvention
from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.circular_fit import CircularAmplitudeBasis,profile_circular_orientation
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits
from egb_multi.native_validation import ChunkedNativeResponse,weighted_relative_residual,normalized_overlap_change_bound
from referee_task_2_objective_orientation import assert_local,write_json


def run(args):
    assert_local();out=args.output.resolve();inputs=args.inputs.resolve();source_file=args.source_file.resolve()
    if not all(p.is_relative_to(ROOT) for p in (out,inputs,source_file)):raise ValueError("Use local files")
    out.mkdir(parents=True,exist_ok=False);hashes={}
    for path in [*list((ROOT/"src/egb_multi").glob("*.py")),Path(__file__),ROOT/"benchmarks/referee_task_2_objective_orientation.py"]:
        rel=path.relative_to(ROOT);target=out/"snapshot"/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    frozen=source_file.read_bytes();(out/"input_candidate_record.txt").write_bytes(frozen)
    if source_file.suffix==".jsonl":
        entries=[json.loads(line) for line in frozen.decode().splitlines()];selected=max(entries,key=lambda r:r["overlap"])
        template=EccentricBinaryParams(**selected["source"]);scope="Interim log-prefix candidate; not completed intrinsic optimization"
    else:
        selected=json.loads(frozen);template=EccentricBinaryParams(**selected["refinement"]["best"]["source"]);scope="Completed annual baseline winner; refinement gates remain"
    meta=json.loads((inputs/f"{args.case}_source.json").read_text());truth=EccentricBinaryParams(**meta["source"])
    duration=meta["duration_s"];base_convention=TDIConvention(**meta["convention"])
    write_json(out/"manifest.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),sha256=hashes,
        input_candidate_file=str(source_file.relative_to(ROOT)),input_candidate_sha256=hashlib.sha256(frozen).hexdigest(),
        case=args.case,scope=scope,physical_band_hz=[2e-5,.04],intrinsic_reoptimized=False,
        source_meta_sha256=hashlib.sha256((inputs/f"{args.case}_source.json").read_bytes()).hexdigest(),
        covariance_sha256=hashlib.sha256((inputs/"annual_covariance.npz").read_bytes()).hexdigest(),
        physical_epochs_s=dict(injection=truth.t0,template=template.t0),pointwise_waveform_allowance=2e-4))
    with np.load(inputs/"annual_covariance.npz") as data:f=data["frequency_hz"];covariance=data["covariance"]
    orbit=default_lisaorbits("esa-trailing");rows=[];failures=[]
    levels=[(10.,61,8192,1e-10,1e-8),(10.,81,8192,1e-10,1e-8),(10.,101,16384,1e-10,1e-8),
        (5.,81,16384,1e-10,1e-8),(2.5,101,32768,1e-10,1e-8),(10.,101,16384,1e-12,1e-10)]
    for dt,order,intervals,rtol,atol in levels:
        start=time.perf_counter();times=np.arange(round(duration/dt))*dt
        convention=replace(base_convention,measurement_order=order,delay_order=7)
        native=ChunkedNativeResponse(orbit,times,convention,block_samples=65536,cached_blocks=2)
        if dt==meta["dt_s"] and order==base_convention.measurement_order and rtol==1e-10:
            with np.load(inputs/f"{args.case}_signal.npz") as data:signal=AESeries(times,{ch:data[ch] for ch in ("A","E")},convention)
        else:signal=native(truth,meta["evolution"],pm_ode_rtol=rtol,pm_ode_phase_atol=atol)
        psd=AEPSD(f,{ch:covariance[:,i,i].real for i,ch in enumerate(("A","E"))},convention,
            dict(sidedness="one_sided",scaling="density",noise_model="native_annual_secondary_diagnostic_fixed_across_waveform_numerics",
                CSD_convention="S_AE=E[tilde_A*conj(tilde_E)]"),covariance[:,0,1])
        metric=AEInnerProduct(signal,psd,(2e-5,.04));envelope=InterpolatedCircularTDI(orbit,times,convention,envelope_intervals=intervals)
        basis=CircularAmplitudeBasis.generate(template,envelope);profile=profile_circular_orientation(metric,basis)
        recovered=replace(template,inclination=profile["inclination_rad"],psi=profile["polarization_rad"],phi0=profile["orbital_phase_rad"])
        approximate=envelope(recovered);exact=native(recovered)
        native_score=metric.compare(exact)["overlap"];candidate_metric=AEInnerProduct(exact,psd,(2e-5,.04))
        relative_error=weighted_relative_residual(candidate_metric,exact,approximate)
        bound=normalized_overlap_change_bound(0,relative_error)
        row=dict(dt_s=dt,measurement_order=order,delay_order=7,envelope_intervals=intervals,pm_ode_rtol=rtol,
            pm_ode_phase_atol=atol,profile=profile,native_physical_candidate_overlap=native_score,
            native_minus_envelope_overlap=native_score-profile["overlap"],weighted_template_relative_error=relative_error,
            selected_template_overlap_error_bound=bound,intrinsic_reoptimized=False,wall_s=time.perf_counter()-start)
        rows.append(row);write_json(out/"candidate_convergence.json",rows)
        print({k:row[k] for k in ("dt_s","measurement_order","envelope_intervals","native_physical_candidate_overlap","selected_template_overlap_error_bound","wall_s")},flush=True)
        if bound>2e-4:failures.append(f"dt={dt},order={order}: template screening bound")
        del native,envelope,basis,signal,metric,approximate,exact,candidate_metric;gc.collect()
    write_json(out/"validation_summary.json",dict(status="passed_selected_annual_candidate_screen" if not failures else "discrepancy_requires_investigation",
        failures=failures,production_certified=False,intrinsic_reoptimized=False,candidate_scope=scope,
        limitations=["Selected candidate with orientation reprofiled, not reoptimized fitting factors", "Fixed secondary covariance diagnostic",
            "No uniform template-family bound or contour convergence inferred"]))
    if failures:raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--source-file",type=Path,required=True);p.add_argument("--case",default="near_contour")
    p.add_argument("--inputs",type=Path,default=ROOT/"outputs/referee/task_2_8_20261008_annual_weighting_pilot")
    run(p.parse_args())
