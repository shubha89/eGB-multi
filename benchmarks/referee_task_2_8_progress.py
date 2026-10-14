"""Refresh Task 2.8 evidence and execution status without certifying production."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from referee_task_2_8_completion import reviewed_completion

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/referee'


def read(path):
    if not path.exists():return None
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return None  # A live child may be writing its checkpoint.


def run(args):
    out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=True)
    statuses={};audit=[]
    declared=[
        'window_audit','stage_precision_year','circular_annual_native_bound',
        'annual_candidate_screen','spectral_objective_provenance_control',
        'uniform_metric_bounds','laser_covariance_grid',
        'annual_search_near_contour_spectral','annual_search_high_e_spectral',
        'annual_search_high_pm_spectral','annual_refinement_near_contour_numerical',
        'finite_annual_optimized_near_contour','finite_annual_optimized_high_e',
        'finite_annual_optimized_high_pm','annual_refinement_high_e','annual_refinement_high_pm',
        'annual_refinement_near_contour_noise_domain',
        'boundary_repair','search_repair_review','annual_search_high_pm_repaired',
        'finite_annual_optimized_high_pm_repaired','annual_refinement_high_pm_repaired',
        'stress_annual_inputs','annual_search_low_asymmetric','annual_search_mid_e',
        'annual_search_small_e_limit','annual_search_contour_e0088','annual_search_contour_e009',
        'finite_high_pm_refined','finite_low_asymmetric','covariance_closure_grid',
        'primary_laser_extended','optimized_contour',
        'moving_secondary_coarse','moving_secondary_refined','independent_high_e',
        'independent_high_pm','band_campaign',
        'independent_high_e_original_epoch_setup','independent_high_pm_original_epoch_setup',
        'final_high_pm_cadence',
        'high_pm_seed_resolution','annual_search_high_pm_branch_repaired',
        'annual_refinement_high_pm_branch_repaired','finite_high_pm_branch_refined',
        'branch_agreement_review','optimized_contour_branch_repaired',
        'band_campaign_branch_repaired','final_high_pm_cadence_branch_repaired',
        'followup_high_pm_band_cadence','followup_high_pm_final_metric_independent',
    ]
    repaired_runs={name:'task_2_8_20261009_'+name for name in declared
                   if name in ('boundary_repair','search_repair_review','stress_annual_inputs',
                       'annual_search_low_asymmetric','annual_search_mid_e','annual_search_small_e_limit',
                       'annual_search_contour_e0088','annual_search_contour_e009','finite_high_pm_refined',
                       'finite_low_asymmetric','covariance_closure_grid','primary_laser_extended','optimized_contour',
                       'moving_secondary_coarse','moving_secondary_refined','independent_high_e','independent_high_pm','band_campaign')
                   or name.endswith('_repaired')}
    for case in ('high_e','high_pm'):
        name='independent_'+case
        repaired_runs[name+'_original_epoch_setup']='task_2_8_20261009_'+name
        if (BASE/('task_2_8_20261009_'+name+'_epoch_repaired')).exists():
            repaired_runs[name]='task_2_8_20261009_'+name+'_epoch_repaired'
    repaired_runs['final_high_pm_cadence']='task_2_8_20261009_final_high_pm_cadence'
    for name in declared:
        if '_branch_' in name or name in ('high_pm_seed_resolution','branch_agreement_review'):
            repaired_runs[name]='task_2_8_20261009_'+name
    repaired_runs['followup_high_pm_band_cadence']='task_2_8_20261012_high_pm_band_cadence'
    repaired_runs['followup_high_pm_final_metric_independent']='task_2_8_20261012_high_pm_final_metric_independent'
    for suffix in declared:
        path=BASE/repaired_runs.get(suffix,'task_2_8_20261008_'+suffix)
        summary=read(path/'validation_summary.json')
        status=dict(output=str(path.relative_to(ROOT)),summary=summary,
                    execution_state='has_final_summary' if summary else 'results_incomplete_or_not_started')
        if suffix.startswith('annual_search_'):
            case=suffix.removeprefix('annual_search_').removesuffix('_spectral').removesuffix('_repaired').removesuffix('_branch')
            baseline=read(path/f'{case}_search_baseline.json')
            if baseline:
                status['completed_basins']=baseline.get('completed_basins',len(baseline['basins_results']))
                status['requested_basins']=baseline.get('requested_basins',len(baseline['basins_results']))
                status['interim_search_best_overlap_not_certified_FF']=baseline['best']['profile']['overlap']
            completed=read(path/f'{case}_annual_search.json')
            if completed:
                status['native_bounded_baseline_overlap']=completed['native_winner_metrics']['overlap']
        refinements=read(path/'annual_refinement.json')
        if refinements:
            status['completed_refinement_levels']=[dict(name=r['level']['name'],native_overlap=r['native_winner_metrics']['overlap'],
                full_multibasin_search=r['full_multibasin_search']) for r in refinements]
        statuses[suffix]=status
        if summary:
            manifest=read(path/'manifest.json')
            if manifest:
                errors=[]
                for relative,expected in manifest.get('sha256',{}).items():
                    snapshot=path/'snapshot'/relative
                    if not snapshot.exists() or hashlib.sha256(snapshot.read_bytes()).hexdigest()!=expected:
                        errors.append('snapshot:'+relative)
                for relative,expected in manifest.get('inputs',{}).items():
                    original=ROOT/relative
                    if not original.exists() or hashlib.sha256(original.read_bytes()).hexdigest()!=expected:
                        errors.append('input:'+relative)
                audit.append(dict(run=suffix,checked_snapshot_hashes=len(manifest.get('sha256',{})),
                    checked_input_hashes=len(manifest.get('inputs',{})),hash_errors=errors,
                    scope='Manifest-declared hashes only; old manifests without input inventories retain that limitation'))
    repair_tests=BASE/'task_2_8_20261009_repair_summary/tests.log'
    new_tests=BASE/'task_2_8_20261009_closure_plan/tests_with_batched_seeds.log'
    if not new_tests.exists():new_tests=BASE/'task_2_8_20261009_closure_plan/tests_with_band_bound.log'
    closure_tests=new_tests if new_tests.exists() else BASE/'task_2_8_20261009_closure_plan/tests.log'
    log=closure_tests if closure_tests.exists() else repair_tests if repair_tests.exists() else out/'tests.log';passed=None
    if log.exists():
        found=re.search(r'(\d+) passed in ([\d.]+)s',log.read_text())
        if found:passed=dict(passed=int(found[1]),wall_s=float(found[2]),log=str(log.relative_to(ROOT)),sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    baseline_keys={'high_e':'annual_search_high_e_spectral',
        'high_pm':'annual_search_high_pm_repaired' if (BASE/repaired_runs['annual_search_high_pm_repaired']).exists() else 'annual_search_high_pm_spectral'}
    baseline_keys['high_pm']='annual_search_high_pm_branch_repaired'
    pending_baselines=[case for case,key in baseline_keys.items()
        if not statuses[key]['summary'] or statuses[key]['summary']['status']!='annual_baseline_completed']
    baseline_status=('circular and near-contour complete at declared scopes; outstanding baselines: '+', '.join(pending_baselines)) if pending_baselines else 'declared circular and three eccentric annual baselines complete; scientific coverage and convergence review still required'
    state=dict(updated_utc=datetime.now(timezone.utc).isoformat(),task_2_8_complete=False,
        total_absolute_FF_target=.001,production_configuration_frozen=False,
        package_checkout=str(ROOT),physical_reference=dict(laser=True,test_mass=True,OMS=True,clocks='ideal',ranges='ideal',galactic_foreground=False),
        tests=passed,gate_status={
            '2.8a_noise_weighting':'partial: all-bin pointwise metric bounds and selected finite-covariance checks; model approximations remain',
            '2.8b_laser_precision':'partial: first arithmetic loss localized; physical primary residual quantified; full sampled-instrument repair not claimed',
            '2.8c_annual_baselines':baseline_status,
            '2.8d_numerical_reoptimization':'nine replacement high-PM levels and final-reference 2.5 s control complete; 5-to-2.5 s shift 4.21e-5; further cadence/band controls are optional',
            '2.8e_band_leakage':'all eight expanded-band cases complete; user accepted the minor 2.08388e-4 band shift on 12 October; working band/evolution allocations are 2.1e-4/9e-5, preserving total 0.001; further band controls optional',
            '2.8f_evolution_optimizer':'replacement twelve-basin baseline and archived blind comparison pass within 1.11e-11; evolution/domain controls complete; independent starts under the final broad-band metric running/pending review',
            '2.8g_contours_frozen_settings':'optimized crossing bracket [0.089,0.089125] completed and final-reference endpoint checks preserve the crossing; combined uncertainty review and frozen settings pending'},
        runs=statuses,provenance_audit=audit,queue_state=read(BASE/'task_2_8_20261008_convergence_queue/queue_state.json'),
        repair_queue_state=read(BASE/'task_2_8_20261009_high_pm_repair_queue/queue_state.json'),
        closure_queue_state=read(BASE/'task_2_8_20261009_closure_queue/queue_state.json'),
        contour_queue_state=read(BASE/'task_2_8_20261009_contour_queue/queue_state.json'),
        moving_queue_state=read(BASE/'task_2_8_20261009_moving_queue/queue_state.json'),
        independent_queue_state=read(BASE/'task_2_8_20261009_independent_queue/queue_state.json'),
        independent_epoch_repaired_queue_state=read(BASE/'task_2_8_20261009_independent_queue_epoch_repaired/queue_state.json'),
        band_campaign_state=read(BASE/'task_2_8_20261009_band_campaign/work_state.json'),
        final_cadence_queue_state=read(BASE/'task_2_8_20261009_final_cadence_queue/queue_state.json'),
        high_pm_branch_queue_state=read(BASE/'task_2_8_20261009_high_pm_branch_queue/queue_state.json'),
        contour_branch_queue_state=read(BASE/'task_2_8_20261009_contour_overlap_queue/queue_state.json'),
        superseded_contour_branch_controller=read(BASE/'task_2_8_20261009_contour_branch_queue/supersession.json'),
        band_branch_campaign_state=read(BASE/'task_2_8_20261009_band_campaign_branch_repaired/work_state.json'),
        final_cadence_branch_queue_state=read(BASE/'task_2_8_20261009_final_cadence_branch_queue/queue_state.json'),
        followup_band_queue_state=read(BASE/'task_2_8_20261012_band_queue/queue_state.json'),
        followup_independent_queue_state=read(BASE/'task_2_8_20261012_independent_queue/queue_state.json'),
        superseded_high_pm_branch=dict(old_native_overlap=.47083613744963976,
            independent_native_overlap=.5245472592777878,improvement=.05371112182814802,
            old_refinement_levels=9,old_numerical_controls_preserved=True,
            old_controls_certify_global_search=False),
        limitations=['Task status requires an evidence review; a completed execution does not close a scientific gate',
                     'Incomplete output does not prove that a process is alive; inspect task-owned processes separately'])
    completion=reviewed_completion()
    state['saved_completion_review']=completion
    if completion and completion['accepted'] and not any(item['hash_errors'] for item in audit):
        state.update(task_2_8_complete=True,production_configuration_frozen=True,
            production_grid_completed=False,completion_scope=completion['scope'],
            gate_status=completion['gate_status'])
    (out/'work_state.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(dict(state_file=str((out/'work_state.json').relative_to(ROOT)),tests=passed,
        completed_run_summaries=sum(item['summary'] is not None for item in statuses.values()),
        provenance_hash_errors=sum(len(item['hash_errors']) for item in audit),task_2_8_complete=state['task_2_8_complete']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=BASE/'task_2_8_20261008_summary')
    run(p.parse_args())
