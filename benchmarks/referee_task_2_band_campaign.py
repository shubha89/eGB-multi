"""Run combined-reference and band controls after immutable dependencies finish.

The final covariance cache and laser grid cover the whole expanded band. Newly
admitted noise-weighted harmonic peaks are selected and optimized. The driver
records each child's scientific status without freezing production settings.
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
BASE=ROOT/'outputs/referee'


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def run(args):
    out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'campaign_driver.py')
    save(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Combined-reference band controls; no automatic production certification'))
    dependencies=[(BASE/'task_2_8_20261009_closure_queue/validation_summary.json','batch_execution_completed'),
        (BASE/'task_2_8_20261009_optimized_contour_branch_repaired/validation_summary.json','optimized_annual_crossing_bracket_completed'),
        (BASE/'task_2_8_20261009_covariance_closure_grid/validation_summary.json','passed_reference_covariance_numerical_controls')]
    while True:
        ready=True
        for path,accepted in dependencies:
            if not path.exists():ready=False;continue
            result=json.loads(path.read_text())
            if result['status'] != accepted or result.get('all_child_executions_successful') is False:
                raise RuntimeError('Dependency requires review: '+str(path.relative_to(ROOT)))
        if ready:break
        save(out/'work_state.json',dict(status='waiting_for_completed_noise_stress_and_contour_controls',production_certified=False))
        time.sleep(20)
    old_inputs=BASE/'task_2_8_20261008_annual_weighting_pilot'
    stress=BASE/'task_2_8_20261009_stress_annual_inputs'
    old_search=BASE/'task_2_8_20261008_annual_search_'
    definitions=[('near_contour',old_inputs,Path(str(old_search)+'near_contour_spectral'),BASE/'task_2_8_20261008_annual_refinement_near_contour_noise_domain',False),
        ('high_e',old_inputs,Path(str(old_search)+'high_e_spectral'),BASE/'task_2_8_20261008_annual_refinement_high_e',False),
        ('high_pm',old_inputs,BASE/'task_2_8_20261009_annual_search_high_pm_branch_repaired',BASE/'task_2_8_20261009_annual_refinement_high_pm_branch_repaired',False),
        ('low_asymmetric',stress,BASE/'task_2_8_20261009_annual_search_low_asymmetric',None,True),
        ('mid_e',stress,BASE/'task_2_8_20261009_annual_search_mid_e',None,True),
        ('small_e_limit',stress,BASE/'task_2_8_20261009_annual_search_small_e_limit',None,False)]
    contour=BASE/'task_2_8_20261009_optimized_contour_branch_repaired'
    nodes=json.loads((contour/'optimized_nodes.json').read_text())
    bracket=json.loads((contour/'validation_summary.json').read_text())['eccentricity_bracket']
    for side,eccentricity in zip(('lower','upper'),bracket):
        node=next(row for row in nodes if row['eccentricity']==eccentricity)
        search=ROOT/node['output'];case=node['case']
        manifest=json.loads((search/'manifest.json').read_text())
        cache=Path(manifest['configuration']['inputs'])
        if not cache.is_absolute():cache=ROOT/cache
        definitions.append((case,cache,search,None,False,side))
    covariance=BASE/'task_2_8_20261009_covariance_closure_grid/annual_covariance.npz'
    laser=BASE/'task_2_8_20261009_primary_laser_extended/annual_laser_covariance_dt150.npz'
    environment=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(BASE/'.precision_deps'),
        MPLCONFIGDIR=str(BASE/'.mplconfig'),XDG_CACHE_HOME=str(BASE/'.cache'))
    results=[]
    for definition in definitions:
        case,inputs,source,retain,stress_case=definition[:5]
        label=case if len(definition)==5 else 'contour_'+definition[5]
        child=out/label;child.mkdir();cache=child/'annual_inputs';cache.mkdir()
        for filename in (case+'_source.json',case+'_signal.npz'):
            shutil.copy2(inputs/filename,cache/filename)
        shutil.copy2(covariance,cache/'annual_covariance.npz')
        base=dict(dt_s=5.,measurement_order=101,envelope_intervals=16384,
            pm_ode_rtol=1e-12,pm_ode_phase_atol=1e-10,band_hz=[2e-5,.04],
            chirp_bound_scaled=48.,width_factor=2.,include_primary_laser=True,
            diagonal_weighting=False,full_multibasin_search=False)
        plan=[dict(base,name='combined_original_band'),dict(base,name='combined_extended_band',
            band_hz=[1e-5,.08],reselect_frequency_basins=True)]
        if stress_case:
            plan.extend([dict(base,name='extended_cadence_2_5',dt_s=2.5,envelope_intervals=32768,
                band_hz=[1e-5,.08],reselect_frequency_basins=True),
                dict(base,name='extended_independent_domain',band_hz=[1e-5,.08],
                    reselect_frequency_basins=True,full_multibasin_search=True,seed_chirp_points=25)])
        plan_file=child/'refinement_plan.json';save(plan_file,plan)
        output=child/'annual_refinement'
        command=[sys.executable,str(ROOT/'benchmarks/referee_task_2_annual_refinement.py'),
            '--output',str(output),'--inputs',str(cache),'--source-run',str(source),'--case',case,
            '--spectral-validation',str(BASE/'task_2_8_20261008_spectral_objective_provenance_control'),
            '--laser-covariance',str(laser),'--plan',str(plan_file),'--local-basins','2',
            '--starts','2','--maxiter','45','--sky-points','96','--exact-seeds','32']
        if retain:command.extend(['--retain-refinement',str(retain)])
        save(out/'work_state.json',dict(status='running',case=label,command=command,
            completed_cases=len(results),production_certified=False))
        print('Combined-reference band case',label,flush=True)
        with (child/'run.log').open('w') as stream:
            result=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Band child requires review: '+label)
        summary=json.loads((output/'validation_summary.json').read_text())
        if summary['status']!='selected_annual_reoptimization_completed':raise RuntimeError('Unaccepted band control: '+label)
        records=json.loads((output/'annual_refinement.json').read_text())
        original,broad=[row['native_winner_metrics']['overlap'] for row in records[:2]]
        results.append(dict(case=label,output=str(output.relative_to(ROOT)),original_band_overlap=original,
            expanded_band_overlap=broad,band_shift=broad-original,levels=len(records),
            band_power_splits=records[1]['band_power_splits']))
        save(out/'band_campaign_results.json',results)
    save(out/'validation_summary.json',dict(status='combined_reference_band_controls_completed',
        production_certified=False,cases_completed=len(results),
        limitations=['Search agreement is empirical over declared basins and starts',
            'Native winner and power splits do not supply a uniform global FF bound',
            'Combined error-budget review and final settings still required']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        if args.output.exists() and not (args.output/'validation_summary.json').exists():
            save(args.output/'validation_summary.json',dict(status='failed_band_campaign',production_certified=False,
                exception_type=type(exc).__name__,exception=str(exc)))
        raise
