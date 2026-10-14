"""Generate and audit a finite ensemble of the adopted native noise reference.

Independent seeds, Welch segment-length changes and A/E covariance are retained
explicitly. No smoothing, covariance flooring or assumed A/E orthogonality is
used. A finite ensemble is not certified merely because it was generated.
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
from scipy.signal import welch,csd
from egb_multi.comparison import AEPSD,AEInnerProduct,TDIConvention
from egb_multi.circular_fit import CircularAmplitudeBasis,profile_circular_orientation
from egb_multi.eccentric import EccentricBinaryParams
from referee_task_2_frequency_chirp import load_psd,prepared_response
from referee_task_2_objective_orientation import build_psd,write_json,assert_local,make_clean_response
from referee_task_2_optimizer_certification import archived_cases
from egb_multi.external import default_lisaorbits,link_delays_from_lisaorbits


def spectra(folder,nperseg):
    metadata=json.loads((folder/'psd_metadata.json').read_text())
    if metadata['noise_model']!='secondary_plus_laser' or metadata['enabled_noise_contributions']!=['laser','test-mass','oms']:
        raise ValueError('Adopted laser/test-mass/OMS reference required')
    with np.load(folder/'noise_and_delays.npz') as arrays:
        halo=metadata['endpoint_halo_samples_each_side']
        a,e=[arrays[f'noise_{ch}_Hz'][halo:-halo]/metadata['central_frequency_hz'] for ch in ('A','E')]
    dt=metadata['dt_s']
    count=1+(len(a)-nperseg)//(nperseg//2)
    if count<4: raise ValueError('At least four complete overlapping Welch segments required')
    kw=dict(fs=1/dt,nperseg=nperseg,noverlap=nperseg//2,window='hann',detrend='constant',scaling='density')
    f,sa=welch(a,**kw);_,se=welch(e,**kw);_,cross=csd(a,e,**kw)
    return f,np.array([sa,se,cross.conj()]),count


def build_ensemble(out,args):
    psd,conv=load_psd(args.psd_run)
    folders=[args.psd_run]
    for seed in args.seeds:
        folder=out/f'seed_{seed}';folder.mkdir()
        cfg=argparse.Namespace(samples=args.noise_samples,dt=5.,seed=seed,order=45,delay_order=5,
            geometry_model='retarded',noise_model='secondary_plus_laser',disable_laser_diagnostic=False,nperseg=65536)
        build_psd(folder,cfg,conv)
        folders.append(folder)
        write_json(out/'generation_progress.json',{'completed_seeds':[int(p.name.split('_')[-1]) for p in folders[1:]],'requested_seeds':args.seeds})
    write_json(out/'ensemble_sources.json',{'folders':[str(p.relative_to(ROOT)) for p in folders],
        'noise_model':'secondary_plus_laser','production_psd_certified':False})
    return folders,conv


def audit(out,folders,conv,args):
    orbit=default_lisaorbits('esa-trailing');times=np.arange(8192)*10.
    delays=link_delays_from_lisaorbits(orbit,times)
    halo=int(np.ceil(8*max(d.max() for d in delays.values())/10))+4*46
    response,_=prepared_response(orbit,times,conv,halo);native,_=make_clean_response(orbit,times,conv,halo)
    problems=[]
    for name,truth,prototype,evolution,old,path in archived_cases(['eccentric_fixed','eccentric_pm','eccentric_high_e']):
        signal=native(truth,evolution)
        # Every previously retained basin winner remains a common candidate
        # pool under each alternative weighting. Intrinsic reoptimization is
        # still required if a weighting choice is adopted for production.
        candidates=[EccentricBinaryParams(**b['best']['source']) for b in old['expanded_family_search']['basins_results']]
        candidates.append(EccentricBinaryParams(**old['final_refinement']['best']['source']))
        bases=[CircularAmplitudeBasis.generate(s,response) for s in candidates]
        problems.append((name,signal,prototype.mean_motion/np.pi,bases))
    rows=[];input_hashes={}
    for folder in folders:
        for filename in ('noise_and_delays.npz','psd_metadata.json'):
            path=folder/filename;input_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for nperseg in args.segment_lengths:
        per_seed=[];segments=[]
        for folder in folders:
            f,values,count=spectra(folder,nperseg);per_seed.append(values);segments.append(count)
        prefixes=sorted(set([1,2,4,8,len(folders)]))
        prefixes=[n for n in prefixes if n<=len(folders)]
        for n in prefixes:
            mean=np.mean(per_seed[:n],axis=0);sa,se,cross=mean[0].real,mean[1].real,mean[2]
            coherence=abs(cross)**2/(sa*se)
            np.savez_compressed(out/f'covariance_n{n}_segment{nperseg}.npz',frequency_hz=f,PSD_A=sa,PSD_E=se,CSD_AE=cross)
            provenance={'sidedness':'one_sided','scaling':'density','noise_model':'secondary_plus_laser',
                'CSD_convention':'S_AE=E[tilde_A*conj(tilde_E)]','seeds':n,'nperseg':nperseg,
                'total_overlapping_segments':sum(segments[:n]),'population_covariance_certified':False,
                'ensemble_weighting':'equal independent realization weights; durations/segment counts may differ'}
            diagonal=AEPSD(f,{'A':sa,'E':se},conv,provenance)
            full=AEPSD(f,{'A':sa,'E':se},conv,provenance,cross)
            for name,signal,carrier,bases in problems:
                band=(.2*carrier,10*carrier)
                fits={}
                for label,psd in (('diagonal',diagonal),('full_covariance',full)):
                    metric=AEInnerProduct(signal,psd,band)
                    scores=[profile_circular_orientation(metric,basis)['overlap'] for basis in bases]
                    fits[label]={'best_common_candidate_overlap':max(scores),'winning_candidate_index':int(np.argmax(scores))}
                selected=(f>=band[0])&(f<=band[1])
                row={'case':name,'independent_seeds':n,'nperseg':nperseg,
                    'frequency_resolution_hz':float(f[1]-f[0]),'overlapping_segments':sum(segments[:n]),
                    'band_hz':list(band),'coherence_median':float(np.median(coherence[selected])),
                    'coherence_p95':float(np.quantile(coherence[selected],.95)),'fits':fits,
                    'full_minus_diagonal_overlap':fits['full_covariance']['best_common_candidate_overlap']-fits['diagonal']['best_common_candidate_overlap']}
                rows.append(row);write_json(out/'weighting_sensitivity.json',rows)
                print(f"{name}: n={n}, segment={nperseg}, Mdiag={fits['diagonal']['best_common_candidate_overlap']:.8g}, covariance shift={row['full_minus_diagonal_overlap']:.4g}",flush=True)
    changes=[]
    for name,_,_,_ in problems:
        chosen=[r for r in rows if r['case']==name and r['independent_seeds']==len(folders)]
        for weighting in ('diagonal','full_covariance'):
            scores=[r['fits'][weighting]['best_common_candidate_overlap'] for r in chosen]
            changes.append({'case':name,'weighting':weighting,'segment_length_score_range':max(scores)-min(scores)})
    failures=[f"{r['case']} {r['weighting']}: segment-length variation exceeds weighting budget" for r in changes if r['segment_length_score_range']>args.psd_budget]
    summary={'status':'passed_finite_ensemble_sensitivity_gate' if not failures else 'discrepancy_requires_investigation',
        'failures':failures,'independent_seeds':len(folders),'changes':changes,
        'production_psd_certified':False,'input_sha256':input_hashes,
        'limitations':['Only common-candidate orientation profiles, no intrinsic reoptimization',
            'Finite ensemble at the initial orbital epoch; annual covariance variation untested',
            'Passing segment-length sensitivity alone does not bound population PSD estimation error',
            'No smoothing or assumed A/E orthogonality']}
    write_json(out/'validation_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--psd-run',type=Path,default=ROOT/'outputs/referee/task_2_3_reference_integrated_20261007')
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(20261008,20261016)))
    p.add_argument('--noise-samples',type=int,default=524288)
    p.add_argument('--segment-lengths',type=int,nargs='+',default=[16384,32768,65536])
    p.add_argument('--psd-budget',type=float,default=4e-4)
    args=p.parse_args();out=args.output.resolve();args.psd_run=args.psd_run.resolve()
    if not out.is_relative_to(ROOT) or not args.psd_run.is_relative_to(ROOT): raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);assert_local()
    sources=list((ROOT/'src/egb_multi').glob('*.py'))+[Path(__file__),ROOT/'benchmarks/referee_task_2_objective_orientation.py',
        ROOT/'benchmarks/referee_task_2_optimizer_certification.py',ROOT/'benchmarks/referee_task_2_frequency_chirp.py']
    hashes={}
    for path in sources:
        rel=path.relative_to(ROOT);target=out/'snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes,
        'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'predetermined_gates':{'weighting_score_range':args.psd_budget}})
    folders,convention=build_ensemble(out,args);audit(out,folders,convention,args);assert_local()
