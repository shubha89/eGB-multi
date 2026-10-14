"""Refresh the local closure inventory while the dependency queues execute.

This controller never certifies physics or freezes a production configuration.
It stops on a declared child failure or when all execution queues have finished,
leaving a concrete evidence bundle for scientific review.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/referee'


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def run(args):
    out=args.output.resolve()
    if not out.is_relative_to(ROOT) or args.interval<10 or not re.fullmatch(r'task_2_8_\d{8}_',args.run_prefix):
        raise ValueError('Local output and a polling interval of at least 10 seconds required')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'watch_driver.py')
    save(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),interval_s=args.interval,
        run_prefix=args.run_prefix,queues=args.queues,children=args.children,
        scope='Execution/evidence inventory only; scientific completion requires review'))
    queues=args.queues or ['closure_queue','high_pm_branch_queue','contour_branch_queue',
            'band_campaign_branch_repaired','final_cadence_branch_queue']
    children=args.children or ['high_pm_seed_resolution','annual_search_high_pm_branch_repaired',
        'annual_refinement_high_pm_branch_repaired','branch_agreement_review','finite_high_pm_branch_refined',
        'covariance_closure_grid','primary_laser_extended','optimized_contour_branch_repaired',
        'final_high_pm_cadence_branch_repaired']
    environment=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(BASE/'.precision_deps'))
    while True:
        updated=datetime.now(timezone.utc).isoformat()
        for driver in ('referee_task_2_8_progress.py','referee_task_2_8_closure_review.py'):
            with (out/(driver+'.log')).open('w') as stream:
                result=subprocess.run([sys.executable,str(ROOT/'benchmarks'/driver)],cwd=ROOT,
                    env=environment,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Inventory refresh failed: '+driver)
        inventory=read(BASE/'task_2_8_20261008_summary/work_state.json')
        summaries={};resolved={}
        for name in queues+children:
            folder=BASE/(args.run_prefix+name)
            supersession=read(folder/'supersession.json')
            if supersession and supersession.get('replacement_queue'):
                replacement=(ROOT/supersession['replacement_queue']).resolve()
                if not replacement.is_relative_to(BASE):raise ValueError('Replacement queue must stay local')
                folder=replacement
            resolved[name]=str(folder.relative_to(ROOT))
            summaries[name]=read(folder/'validation_summary.json')
        failures={name:value for name,value in summaries.items() if value and
            (value['status'].startswith('failed_') or value['status']=='requires_investigation'
             or value.get('all_child_executions_successful') is False)}
        audit_errors=[row for row in inventory['provenance_audit'] if row['hash_errors']]
        completed=all(summaries[name] is not None for name in queues)
        status='requires_scientific_intervention' if failures or audit_errors else (
            'execution_bundle_ready_for_scientific_review' if completed else 'monitoring_dependency_queues')
        save(out/'work_state.json',dict(updated_utc=updated,status=status,task_2_8_complete=False,
            production_configuration_frozen=False,declared_failures=failures,provenance_errors=audit_errors,
            execution_summaries=summaries,resolved_execution_folders=resolved,
            scope='A saved running state does not prove process liveness; use a live process check separately'))
        print(updated,status,flush=True)
        if failures or audit_errors or completed:return
        time.sleep(args.interval)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--interval',type=int,default=60)
    p.add_argument('--run-prefix',default='task_2_8_20261009_')
    p.add_argument('--queues',nargs='+');p.add_argument('--children',nargs='+')
    run(p.parse_args())
