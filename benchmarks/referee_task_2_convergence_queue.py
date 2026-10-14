"""Execute this checkout's immediate, dependency-ordered convergence batch.

The queue never marks scientific convergence complete. Each child keeps its
own frozen inputs, source snapshot, results and failure report. A dependency
failure is recorded and prevents the dependent calculation from proceeding.
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

ROOT=Path(__file__).resolve().parents[1]


def save(path, data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


def run(args):
    out=args.output.resolve();plan_path=args.plan.resolve()
    if not out.is_relative_to(ROOT) or not plan_path.is_relative_to(ROOT):
        raise ValueError('Use this checkout')
    plan=json.loads(plan_path.read_text())
    if not plan or len({row['name'] for row in plan})!=len(plan):
        raise ValueError('Distinct named jobs required')
    for row in plan:
        driver=(ROOT/row['driver']).resolve()
        if not driver.is_relative_to(ROOT/'benchmarks') or not driver.name.startswith('referee_task_2_'):
            raise ValueError('Only local referee-task drivers are allowed')
        if not (ROOT/row['output']).resolve().is_relative_to(ROOT):
            raise ValueError('Child output must remain local')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(plan_path,out/'input_plan.json');shutil.copy2(__file__,out/'queue_driver.py')
    save(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python=sys.executable,
        scope='Immediate execution queue; no automatic scientific completion or frozen production decision'))
    state={row['name']:dict(status='waiting_for_dependencies') for row in plan}
    environment=dict(os.environ,MPLCONFIGDIR=str(ROOT/'outputs/referee/.mplconfig'),
        XDG_CACHE_HOME=str(ROOT/'outputs/referee/.cache'),PYTHONPATH=str(ROOT/'outputs/referee/.precision_deps'))
    pending=list(plan)
    while pending:
        progressed=False
        for row in list(pending):
            dependencies=row.get('dependencies',[]);ready=True;failure=None
            for dependency in dependencies:
                summary_path=ROOT/dependency['summary']
                if not summary_path.exists():ready=False;break
                try:summary=json.loads(summary_path.read_text())
                except json.JSONDecodeError:ready=False;break
                if summary['status'] not in dependency['accepted_statuses']:
                    failure=dict(path=str(summary_path.relative_to(ROOT)),status=summary['status']);break
            if failure:
                state[row['name']]=dict(status='dependency_failed_requires_review',dependency=failure)
                pending.remove(row);save(out/'queue_state.json',state);progressed=True;continue
            if not ready:continue
            child_out=ROOT/row['output']
            if child_out.exists():
                raise ValueError('Refuse to overwrite a child run: '+str(child_out))
            command=[sys.executable,str(ROOT/row['driver']),'--output',str(child_out),*row['arguments']]
            state[row['name']]=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),command=command)
            save(out/'queue_state.json',state)
            print('Starting',row['name'],flush=True)
            with (out/(row['name']+'.log')).open('w') as stream:
                result=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            state[row['name']].update(status='execution_completed' if result.returncode==0 else 'execution_failed_requires_review',
                returncode=result.returncode,finished_utc=datetime.now(timezone.utc).isoformat(),
                child_summary=str((child_out/'validation_summary.json').relative_to(ROOT)))
            save(out/'queue_state.json',state);pending.remove(row);progressed=True
            print(row['name'],state[row['name']]['status'],flush=True)
            break  # Run only one child at a time to limit resource use.
        if not progressed:
            save(out/'queue_state.json',state);time.sleep(20)
    save(out/'validation_summary.json',dict(status='batch_execution_completed',production_certified=False,
        all_child_executions_successful=all(item['status']=='execution_completed' for item in state.values()),
        scope='Execution bookkeeping only; physics gates and production choices require a separate evidence review'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    run(p.parse_args())
