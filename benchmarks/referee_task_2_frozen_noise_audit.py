"""Diagnose finite-estimator noise weighting against native frozen-arm spectra.

This does not adopt a production PSD. The deterministic model must still be
checked for time-dependent TDI residuals, finite-window bias and annual use.
"""

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/referee/.cache'))
import numpy as np
from lisainstrument import Instrument
from egb_multi.noise_covariance import frozen_native_secondary_covariance
from egb_multi.external import default_lisaorbits,link_delays_from_lisaorbits
from egb_multi.comparison import AEPSD,AEInnerProduct
from egb_multi.circular_fit import CircularAmplitudeBasis,profile_circular_orientation
from egb_multi.eccentric import EccentricBinaryParams
from referee_task_2_psd_certification import spectra
from referee_task_2_optimizer_certification import archived_cases
from referee_task_2_frequency_chirp import load_psd,prepared_response
from referee_task_2_objective_orientation import assert_local,make_clean_response,write_json


def instrument(cutoff,dt=5.):
    return Instrument(size=524288,dt=dt,physics_upsampling=1,aafilter=None,
        telemetry_downsampling=524288,noises_f_min_hz=cutoff,interpolation=('lagrange',45))


def model(f,orbit,epochs,cutoff,dt=5.):
    definitions=instrument(cutoff,dt);delays=link_delays_from_lisaorbits(orbit,np.asarray(epochs))
    values=[]
    for i in range(len(epochs)):
        covariance,_=frozen_native_secondary_covariance(f,{key:float(v[i]) for key,v in delays.items()},definitions)
        values.append(covariance)
    return np.array(values)


def run(args):
    out=args.output.resolve();ensemble=args.ensemble.resolve()
    if not out.is_relative_to(ROOT) or not ensemble.is_relative_to(ROOT): raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);assert_local()
    manifest={}
    paths=list((ROOT/'src/egb_multi').glob('*.py'))+[Path(__file__),ROOT/'benchmarks/referee_task_2_psd_certification.py']
    for path in paths:
        rel=path.relative_to(ROOT);target=out/'snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        manifest[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':manifest,
        'configuration':{'ensemble':str(ensemble.relative_to(ROOT)),'cutoff_hz':args.cutoff,'annual_epochs':args.epochs},
        'scope':'diagnostic only; frozen arms, native discrete secondary PSD; no laser residual certification'})
    folders=[ROOT/rel for rel in json.loads((ensemble/'ensemble_sources.json').read_text())['folders']]
    f,_,_=spectra(folders[0],65536);selected=(f>0)&(f<=.05);f=f[selected]
    orbit=default_lisaorbits('esa-trailing');year=31557600.
    epochs=(np.arange(args.epochs)+.5)*year/args.epochs
    annual=model(f,orbit,epochs,args.cutoff).mean(axis=0)
    fine=model(f,orbit,(np.arange(2*args.epochs)+.5)*year/(2*args.epochs),args.cutoff).mean(axis=0)
    initial=model(f,orbit,[0.],args.cutoff)[0]
    np.savez_compressed(out/'frozen_covariance.npz',frequency_hz=f,initial=initial,annual=annual,annual_refined=fine)
    comparisons=[];per_seed=[];expected=[]
    for folder in folders:
        metadata=json.loads((folder/'psd_metadata.json').read_text());duration=metadata['dt_s']*metadata['samples_before_endpoint_crop']
        cutoff=metadata['instrument_kwargs'].get('noises_f_min_hz') or 1/duration
        _,values,count=spectra(folder,65536);values=values[:,selected]
        covariance=np.zeros((len(f),2,2),complex)
        covariance[:,0,0]=values[0].real;covariance[:,1,1]=values[1].real
        covariance[:,0,1]=values[2];covariance[:,1,0]=values[2].conj()
        native=model(f,orbit,[duration/2],cutoff)[0]
        per_seed.append(covariance);expected.append(native)
        comparisons.append({'folder':str(folder.relative_to(ROOT)),'resolved_cutoff_hz':cutoff,'duration_s':duration,'overlapping_segments':count})
    observed=np.mean(per_seed,axis=0);expected=np.mean(expected,axis=0)
    # Frequency bin averages quantify broad spectral agreement only. They
    # cannot certify a narrow-line fitting-factor accuracy budget.
    bins=[]
    for low,high in ((2e-5,1e-4),(1e-4,3e-4),(3e-4,1e-3),(1e-3,3e-3),(3e-3,.01),(.01,.02),(.02,.028),(.032,.04)):
        mask=(f>=low)&(f<high)
        sa=observed[mask,0,0].real/expected[mask,0,0].real
        se=observed[mask,1,1].real/expected[mask,1,1].real
        delta_cross=(observed[mask,0,1]-expected[mask,0,1])/np.sqrt(expected[mask,0,0].real*expected[mask,1,1].real)
        bins.append({'band_hz':[low,high],'frequency_bins':int(mask.sum()),'mean_observed_over_model_A':float(np.mean(sa)),
            'mean_observed_over_model_E':float(np.mean(se)),'mean_normalized_CSD_difference_real':float(delta_cross.real.mean()),
            'mean_normalized_CSD_difference_imag':float(delta_cross.imag.mean()),
            'normalized_CSD_difference_rms':float(np.sqrt(np.mean(abs(delta_cross)**2)))})
    write_json(out/'simulation_comparison.json',{'sources':comparisons,'bins':bins})
    psd,conv=load_psd(folders[0]);times=np.arange(8192)*10.
    delays=link_delays_from_lisaorbits(orbit,times);halo=int(np.ceil(8*max(v.max() for v in delays.values())/10))+4*46
    response,_=prepared_response(orbit,times,conv,halo);native,_=make_clean_response(orbit,times,conv,halo)
    alternative_cutoff=model(f,orbit,[0.],1/(262144*5))[0]
    # The 5 s simulated noise filters approximate their intended continuum
    # spectra. Check that discretization separately from waveform cadence.
    noise_cadences={f'noise_dt_{dt:g}':model(f,orbit,epochs,args.cutoff,dt).mean(axis=0) for dt in args.noise_cadences}
    rows=[]
    for name,truth,prototype,evolution,old,path in archived_cases(['eccentric_fixed','eccentric_pm','eccentric_high_e']):
        signal=native(truth,evolution)
        candidates=[EccentricBinaryParams(**b['best']['source']) for b in old['expanded_family_search']['basins_results']]
        candidates.append(EccentricBinaryParams(**old['final_refinement']['best']['source']))
        bases=[CircularAmplitudeBasis.generate(s,response) for s in candidates]
        fits={}
        for label,covariance in [('initial',initial),('annual',annual),('annual_refined',fine),('legacy_cutoff',alternative_cutoff),*noise_cadences.items()]:
            provenance={'sidedness':'one_sided','scaling':'density','noise_model':'native_frozen_secondary_diagnostic',
                'CSD_convention':'S_AE=E[tilde_A*conj(tilde_E)]','production_certified':False}
            for full in (False,True):
                weighting=AEPSD(f,{'A':covariance[:,0,0].real,'E':covariance[:,1,1].real},conv,provenance,covariance[:,0,1] if full else None)
                metric=AEInnerProduct(signal,weighting,(.2*prototype.mean_motion/np.pi,10*prototype.mean_motion/np.pi))
                scores=[profile_circular_orientation(metric,basis)['overlap'] for basis in bases]
                fits[label+('_full' if full else '_diagonal')]={'best_common_candidate_overlap':max(scores),'winning_candidate_index':int(np.argmax(scores))}
        rows.append({'case':name,'fits':fits})
        write_json(out/'weighting_sensitivity.json',rows)
        print(name,json.dumps(fits),flush=True)
    coherence=abs(annual[:,0,1])**2/(annual[:,0,0].real*annual[:,1,1].real)
    write_json(out/'diagnostic_summary.json',{'status':'frozen_model_diagnostic_completed','production_certified':False,
        'annual_coherence_median_20uHz_20mHz':float(np.median(coherence[(f>=2e-5)&(f<=.02)])),
        'annual_coherence_p95_20uHz_20mHz':float(np.quantile(coherence[(f>=2e-5)&(f<=.02)],.95)),
        'limitations':['No moving-arm laser residual in the deterministic covariance',
            'No finite-observation spectral convolution or frequency-bin covariance',
            'Native digital noise filters at 5 s; physical continuum/cadence convergence untested',
            'Broad frequency-bin averages do not bound narrow-line PSD bias','Common candidate pool; no intrinsic reoptimization']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ensemble',type=Path,default=ROOT/'outputs/referee/task_2_8_20261008_noise_ensemble')
    p.add_argument('--cutoff',type=float,default=1/31557600.)
    p.add_argument('--noise-cadences',type=float,nargs='*',default=[1.,.25,.125])
    p.add_argument('--epochs',type=int,default=16);run(p.parse_args())
