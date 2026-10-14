"""Task 2.8 response convergence on a common physical observation interval.

Cadence, TDI interpolation, bandwidth, source evolution and block boundaries
are varied separately. These waveform checks do not certify the optimizer or
the stochastic PSD. The one-day pilot is explicitly distinguished from the
one-year production checks. The native exact-link path supplies all responses.
"""

import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/referee/.cache'))
import numpy as np
from egb_multi.comparison import AESeries,AEPSD,AEInnerProduct
from egb_multi.circular_fit import CircularAmplitudeBasis,profile_circular_orientation
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits
from egb_multi.native_validation import ChunkedNativeResponse,weighted_relative_residual,normalized_overlap_change_bound
from referee_task_2_frequency_chirp import load_psd
from referee_task_2_optimizer_certification import provenance
from referee_task_2_objective_orientation import assert_local,write_json


def adapt_psd(psd,convention):
    # Explicitly use the SAME provisional physical weighting curve while
    # varying waveform numerics; this is not a regenerated noise PSD.
    return AEPSD(psd.frequency_hz,psd.channels,convention,
        dict(psd.provenance,numerical_comparison='fixed physical weighting, varied response interpolation'),psd.cross_ae)


def evaluate(out,args):
    assert_local();psd,base_convention=load_psd(args.psd_run)
    orbit=default_lisaorbits('esa-trailing')
    definitions={'low':(.0001,.1,.5,.5,.2,.2,-1.),
        'high':(.003,.5,.8,.6,.8,.2,-1.),
        'high_mass':(.003,.5,1.,.3,1.2,-.7,2.4)}
    records,failures=[],[]
    for name in args.cases:
        frequency,e,m1,m2,inc,beta,longitude=definitions[name]
        source=EccentricBinaryParams(mean_motion=np.pi*frequency,eccentricity=e,
            m1_solar=m1,m2_solar=m2,inclination=inc,beta=beta,lambda_=longitude,
            phi0=.2,psi=.4,t0=0.)
        started=time.perf_counter()
        fine_dt=args.reference_dt
        times=np.arange(int(round(args.duration/fine_dt)))*fine_dt
        if abs(times.size*fine_dt-args.duration)>1e-6: raise ValueError('Duration must be an integer number of samples')
        fine_convention=replace(base_convention,measurement_order=61,delay_order=7)
        fine_response=ChunkedNativeResponse(orbit,times,fine_convention,
            block_samples=args.block_samples,cached_blocks=2)
        print(f'{name}: reference N={len(times)}, dt={fine_dt}, duration={args.duration}s',flush=True)
        fine_signal=fine_response(source,'peters_mathews')
        fine_fixed=fine_response(source,'fixed')
        template=replace(source,eccentricity=0,fdot=0.)
        fine_template=fine_response(template)
        band=(.2*frequency,10*frequency)
        metric=AEInnerProduct(fine_signal,adapt_psd(psd,fine_convention),band)
        fine_score=metric.compare(fine_template)['overlap']
        evolution_error=weighted_relative_residual(metric,fine_signal,fine_fixed)
        metrics={'legacy_source_auto_mismatch_tolerance':.01,
            'source_auto_response_accuracy_certified':False,
            'fixed_vs_full_PM_weighted_relative_residual':evolution_error,
            'fixed_vs_full_PM_overlap_change_bound':normalized_overlap_change_bound(evolution_error,0.) if evolution_error<1 else None}
        comparisons=[]
        for dt,order,delay_order in ((20.,45,5),(10.,15,5),(10.,31,5),(10.,45,5),(10.,61,7),(5.,45,5)):
            if dt<fine_dt: continue
            stride=int(round(dt/fine_dt))
            if abs(stride*fine_dt-dt)>1e-10: raise ValueError('Nested cadences required')
            t=times[::stride]
            if len(t)*dt!=args.duration: raise ValueError('Cadence changes the physical duration')
            conv=replace(base_convention,measurement_order=order,delay_order=delay_order)
            engine=ChunkedNativeResponse(orbit,t,conv,block_samples=args.block_samples)
            signal=engine(source,'peters_mathews');candidate=engine(template)
            ref=AESeries(t,{ch:fine_signal.channels[ch][::stride] for ch in ('A','E')},conv)
            ref_template=AESeries(t,{ch:fine_template.channels[ch][::stride] for ch in ('A','E')},conv)
            valid=band[1]<.5/dt
            # An inadequate Nyquist frequency is recorded as invalid production
            # coverage. A smaller diagnostic band never counts as a pass.
            diagnostic_band=band if valid else (band[0],min(band[1],.45/dt))
            common=AEInnerProduct(ref,adapt_psd(psd,conv),diagnostic_band)
            source_error=weighted_relative_residual(common,ref,signal)
            template_error=weighted_relative_residual(common,ref_template,candidate)
            score=AEInnerProduct(signal,adapt_psd(psd,conv),diagnostic_band).compare(candidate)['overlap']
            row={'dt_s':dt,'measurement_order':order,'delay_order':delay_order,
                'production_band_covered':valid,'diagnostic_band_hz':list(diagnostic_band),
                'weighted_source_relative_residual':source_error,
                'weighted_template_relative_residual':template_error,
                'pointwise_overlap_change_bound':normalized_overlap_change_bound(source_error,template_error) if max(source_error,template_error)<1 else None,
                'same_parameter_overlap':score,'reference_same_parameter_overlap':common.compare(ref_template)['overlap'],
                'norm_relative_change_vs_fine_full_band':float((common.reference_norm/metric.reference_norm)-1) if valid else None}
            comparisons.append(row)
            write_json(out/f'{name}_convergence_progress.json',comparisons)
            print(f'  dt={dt}, order={order}: source error={source_error:.4g}, template error={template_error:.4g}, band_valid={valid}',flush=True)
        # Change the physical band with the high-resolution signal and PSD
        # held fixed. These are same-candidate checks, not reoptimized FFs.
        bands=[]
        for low_factor,high_factor in ((.1,8.),(.2,10.),(.1,12.)):
            test_band=(low_factor*frequency,high_factor*frequency)
            alternate=AEInnerProduct(fine_signal,adapt_psd(psd,fine_convention),test_band)
            bands.append({'band_hz':list(test_band),'reference_norm':alternate.reference_norm,
                'same_parameter_overlap':alternate.compare(fine_template)['overlap']})
        accepted=next(r for r in comparisons if r['dt_s']==10 and r['measurement_order']==45)
        if not accepted['production_band_covered'] or accepted['pointwise_overlap_change_bound'] is None or accepted['pointwise_overlap_change_bound']>args.waveform_budget:
            failures.append(f'{name}: proposed dt10/order45 waveform bound exceeds budget')
        record={'case':name,'source':asdict(source),'duration_s':args.duration,
            'reference_dt_s':fine_dt,'reference_orders':[61,7],
            'reference_same_parameter_overlap':fine_score,'source_evolution':metrics,
            'comparisons':comparisons,'band_comparisons':bands,'wall_s':time.perf_counter()-started}
        records.append(record);write_json(out/'numerical_validation.json',records)
    summary={'status':'passed_declared_waveform_checks' if not failures else 'discrepancy_requires_investigation',
        'failures':failures,'cases':len(records),'duration_s':args.duration,
        'production_duration_tested':args.duration>=31557600,
        'full_production_certified':False,
        'limitations':['Fixed representative templates, not reoptimized fitting factors',
            'Pointwise waveform bounds are not proven uniform over the circular family',
            'PSD/covariance and complete optimizer are separate requirements',
            'The legacy auto mismatch tolerance is not a response-level error guarantee']}
    write_json(out/'validation_summary.json',summary);print(json.dumps(summary,indent=2),flush=True);assert_local()
    return not failures


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--psd-run',type=Path,default=ROOT/'outputs/referee/task_2_3_reference_integrated_20261007')
    p.add_argument('--duration',type=float,default=81920.);p.add_argument('--reference-dt',type=float,default=2.5)
    p.add_argument('--cases',nargs='+',choices=['low','high','high_mass'],default=['low','high','high_mass'])
    p.add_argument('--block-samples',type=int,default=65536);p.add_argument('--waveform-budget',type=float,default=2e-4)
    args=p.parse_args();args.mode='waveform';out=args.output.resolve();args.psd_run=args.psd_run.resolve()
    if not out.is_relative_to(ROOT) or not args.psd_run.is_relative_to(ROOT): raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False)
    # Reuse source archiving; the optimizer-specific gate is removed below.
    provenance(out,argparse.Namespace(**vars(args),optimizer_error=None))
    manifest=json.loads((out/'manifest.json').read_text());relative=Path(__file__).relative_to(ROOT)
    target=out/'snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(__file__).read_bytes())
    manifest['sha256'][str(relative)]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['predetermined_gates']={'waveform_pointwise_overlap_bound':args.waveform_budget}
    write_json(out/'manifest.json',manifest)
    if not evaluate(out,args): raise SystemExit(1)
