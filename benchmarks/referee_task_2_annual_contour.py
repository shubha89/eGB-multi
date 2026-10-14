"""Refine a loss crossing using complete annual intrinsic searches at each node.

All six coordinates and physical orientation remain free. The union of previous
basin winners is retained under each new injection/metric, alongside independent
seeds. The resulting empirical crossing is distinct from production certification
and from its displacement under an absolute FF uncertainty.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def local(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('All inputs and outputs must be in this checkout')
    return path


def endpoint(path, case):
    summary = json.loads((path/'validation_summary.json').read_text())
    if summary['status'] != 'annual_baseline_completed':
        raise ValueError('An endpoint search has not completed successfully')
    record = json.loads((path/f'{case}_annual_search.json').read_text())
    return dict(eccentricity=record['source']['eccentricity'],
        overlap=record['native_winner_metrics']['overlap'],
        case=case, output=str(path.relative_to(ROOT)), record=record)


def run(args):
    out = local(args.output)
    paths = [local(p) for p in (args.lower_run,args.upper_run,args.covariance_inputs,args.spectral_validation)]
    lower_run,upper_run,covariance_inputs,control = paths
    if out.exists(): raise ValueError('Refuse to overwrite a contour calculation')
    out.mkdir(parents=True)
    shutil.copy2(__file__,out/'contour_driver.py')
    save(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Empirical optimized annual crossing; no automatic production or global-maximum certification'))
    waiting=[(lower_run,args.lower_case),(upper_run,args.upper_case)]
    resource_dependency=local(args.resource_dependency) if args.resource_dependency else None
    while not all((p/'validation_summary.json').exists() for p,_ in waiting) or (resource_dependency and not resource_dependency.exists()):
        save(out/'work_state.json',dict(status='waiting_for_completed_endpoint_searches',production_certified=False))
        time.sleep(20)
    if resource_dependency:
        dependency=json.loads(resource_dependency.read_text())
        if dependency['status']!='batch_execution_completed' or not dependency['all_child_executions_successful']:
            raise ValueError('Independent-search dependency needs review before crossing refinement')
        if args.resource_only:
            # Scheduling is independent of the high-PM scientific result.
            # The crossing can be computed while that unrelated stress case
            # is repaired; the final joint closure review still requires both.
            save(out/'deferred_joint_review.json',dict(
                high_pm_independent_agreement_required_for_final_closure=True,
                scope='Resource dependency only; crossing execution does not certify the optimizer campaign'))
        elif args.independent_agreement_review:
            review=local(args.independent_agreement_review)
            agreement=json.loads(review.read_text())
            if agreement['status']!='passed_archived_independent_branch_agreement':
                raise ValueError('Replacement baseline independent agreement needs review')
            shutil.copy2(review,out/'input_independent_agreement_review.json')
            save(out/'independent_agreement_hash.json',dict(path=str(review.relative_to(ROOT)),
                sha256=hashlib.sha256(review.read_bytes()).hexdigest()))
        else:
            state=json.loads((resource_dependency.parent/'queue_state.json').read_text())
            for job in state.values():
                child=ROOT/job['child_summary']
                records=json.loads((child.parent/'annual_refinement.json').read_text())
                for row in records:
                    if not row['independent_starts_only'] or row['retained_candidate_count']!=0 or abs(row['independent_baseline_deficit'])>1e-4:
                        raise ValueError('Independent winning-branch agreement has not passed')
    low,high = [endpoint(p,c) for p,c in waiting]
    if low['eccentricity'] >= high['eccentricity']:raise ValueError('Ordered eccentricity endpoints required')
    truth = low['record']['source'].copy()
    other = high['record']['source'].copy()
    for source in (truth,other): source.pop('eccentricity')
    if truth != other or low['record']['evolution'] != high['record']['evolution']:
        raise ValueError('Crossing endpoints must have the same physical source except eccentricity')
    history = [low,high]
    if high['overlap'] > args.target and args.fallback_upper_run:
        fallback_path=local(args.fallback_upper_run)
        fallback=endpoint(fallback_path,args.fallback_upper_case)
        source=fallback['record']['source'].copy();source.pop('eccentricity')
        if source != truth or fallback['record']['evolution'] != low['record']['evolution']:
            raise ValueError('Fallback source does not match the physical crossing')
        if fallback['eccentricity'] <= high['eccentricity']:raise ValueError('Fallback upper endpoint must extend the bracket')
        low,high=high,fallback;history.append(fallback);waiting.append((fallback_path,args.fallback_upper_case))
    if not low['overlap'] >= args.target >= high['overlap']:
        save(out/'validation_summary.json',dict(status='requires_bracket_expansion',production_certified=False,
            endpoints=[{k:v for k,v in r.items() if k!='record'} for r in history],target=args.target))
        return
    inputs = {str((p/f'{c}_annual_search.json').relative_to(ROOT)):
        hashlib.sha256((p/f'{c}_annual_search.json').read_bytes()).hexdigest() for p,c in waiting}
    save(out/'endpoint_input_hashes.json',inputs)
    save(out/'optimized_nodes.json',[{k:v for k,v in r.items() if k!='record'} for r in history])
    environment=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT/'outputs/referee/.precision_deps'),
        MPLCONFIGDIR=str(ROOT/'outputs/referee/.mplconfig'),XDG_CACHE_HOME=str(ROOT/'outputs/referee/.cache'))
    def child(driver,output,arguments,log):
        command=[sys.executable,str(ROOT/'benchmarks'/driver),'--output',str(output),*arguments]
        save(out/'work_state.json',dict(status='running',command=command,log=str(log.relative_to(ROOT)),
            optimized_nodes=len(history),production_certified=False))
        with log.open('w') as stream:
            result=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'{driver} failed with return code {result.returncode}; see {log}')
    for index in range(args.max_nodes):
        if high['eccentricity']-low['eccentricity'] <= args.width: break
        mid=.5*(low['eccentricity']+high['eccentricity'])
        node=out/f'node_{index:02d}';node.mkdir()
        case='closure_contour'
        definition={case:dict(frequency_hz=truth['mean_motion']/3.141592653589793,eccentricity=mid,
            m1_solar=truth['m1_solar'],m2_solar=truth['m2_solar'],inclination_rad=truth['inclination'],
            beta_rad=truth['beta'],longitude_rad=truth['lambda_'],evolution=low['record']['evolution'])}
        definitions=node/'case_definitions.json';save(definitions,definition)
        cache=node/'annual_inputs'
        child('referee_task_2_annual_weighting_pilot.py',cache,['--case-definitions',str(definitions),
            '--cases',case,'--harmonics','2','3','--covariance-inputs',str(covariance_inputs)],node/'cache.log')
        templates={}
        for record in history:
            winners=[r['best'] for r in record['record']['search']['basins_results']]
            winners.append(record['record']['refinement']['best'])
            for winner in winners:
                source=winner['source'];key=json.dumps(source,sort_keys=True)
                templates[key]=dict(source=source,origin=record['output'])
        retained=node/'retained_templates.json';save(retained,list(templates.values()))
        output=node/'annual_search'
        child('referee_task_2_annual_search.py',output,['--inputs',str(cache),'--cases',case,
            '--seed-method','envelope','--spectral-objective','--spectral-validation',str(control),
            '--retain-candidates',str(retained),'--width','12','--chirp','16','--chirp-points','5',
            '--sky-points','48','--exact-seeds','12','--maxiter','60','--ftol','1e-9','--fd-step','1e-6'],node/'search.log')
        current=endpoint(output,case);history.append(current)
        if current['overlap'] >= args.target: low=current
        else: high=current
        save(out/'optimized_nodes.json',[{k:v for k,v in r.items() if k!='record'} for r in history])
        print('Optimized crossing bracket',low['eccentricity'],high['eccentricity'],flush=True)
    slope=(high['overlap']-low['overlap'])/(high['eccentricity']-low['eccentricity'])
    estimate=low['eccentricity']+(args.target-low['overlap'])/slope
    status='optimized_annual_crossing_bracket_completed' if high['eccentricity']-low['eccentricity'] <= args.width else 'requires_more_crossing_nodes'
    save(out/'validation_summary.json',dict(status=status,production_certified=False,target=args.target,
        eccentricity_bracket=[low['eccentricity'],high['eccentricity']],endpoint_overlaps=[low['overlap'],high['overlap']],
        nominal_linear_crossing=estimate,nominal_bracket_width=high['eccentricity']-low['eccentricity'],
        local_empirical_dFF_de=slope,working_absolute_FF_allowance=args.ff_allowance,
        approximate_crossing_displacement_from_FF_allowance=args.ff_allowance/abs(slope),
        optimized_nodes=len(history),limitations=['Finite starts/basins are an empirical maximization check',
            'Baseline secondary-only metric; final combined-reference endpoint controls still required',
            'Small bisection width does not eliminate FF uncertainty or demonstrate monotonicity outside this bracket']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lower-run',type=Path,required=True);p.add_argument('--lower-case',default='contour_e0088')
    p.add_argument('--upper-run',type=Path,required=True);p.add_argument('--upper-case',default='contour_e009')
    p.add_argument('--fallback-upper-run',type=Path,default=ROOT/'outputs/referee/task_2_8_20261008_annual_search_near_contour_spectral')
    p.add_argument('--fallback-upper-case',default='near_contour',help='Completed e=0.1 baseline extends a shifted optimized crossing bracket')
    p.add_argument('--covariance-inputs',type=Path,required=True)
    p.add_argument('--spectral-validation',type=Path,required=True)
    p.add_argument('--resource-dependency',type=Path,default=ROOT/'outputs/referee/task_2_8_20261009_independent_queue_epoch_repaired/validation_summary.json',
        help='Wait for the preceding independent search queue to keep at most two heavy jobs active')
    p.add_argument('--independent-agreement-review',type=Path,
        help='Separate review of archived blind searches against replacement baselines')
    p.add_argument('--resource-only',action='store_true',
        help='Use the execution dependency for scheduling; independent stress-case validation stays in the final joint review')
    p.add_argument('--target',type=float,default=.9);p.add_argument('--width',type=float,default=.0002)
    p.add_argument('--max-nodes',type=int,default=6);p.add_argument('--ff-allowance',type=float,default=.001)
    args=p.parse_args()
    try: run(args)
    except Exception as exc:
        if args.output.exists() and not (args.output/'validation_summary.json').exists():
            save(args.output/'validation_summary.json',dict(status='failed_contour_calculation',
                exception_type=type(exc).__name__,exception=str(exc),production_certified=False))
        raise
