"""Review actual Task 2.8 evidence without treating executions as certification.

Reports separate mathematical metric bounds, empirical search differences,
finite-candidate controls, and still-open physical/reference decisions.
"""

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from referee_task_2_8_completion import reviewed_completion

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/referee'


def read(path):
    if not path.exists():return None
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return None


def run(args):
    out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=True)
    required={
        'task_2_8_20261009_high_pm_seed_resolution':'controlled_seed_resolution_diagnosis_completed',
        'task_2_8_20261009_annual_search_high_pm_branch_repaired':'annual_baseline_completed',
        'task_2_8_20261009_annual_refinement_high_pm_branch_repaired':'selected_annual_reoptimization_completed',
        'task_2_8_20261009_branch_agreement_review':'passed_archived_independent_branch_agreement',
        'task_2_8_20261009_annual_search_low_asymmetric':'annual_baseline_completed',
        'task_2_8_20261009_annual_search_mid_e':'annual_baseline_completed',
        'task_2_8_20261009_annual_search_small_e_limit':'annual_baseline_completed',
        'task_2_8_20261009_finite_high_pm_branch_refined':'finite_covariance_candidate_diagnostic_completed',
        'task_2_8_20261009_finite_low_asymmetric':'finite_covariance_candidate_diagnostic_completed',
        'task_2_8_20261009_covariance_closure_grid':'passed_reference_covariance_numerical_controls',
        'task_2_8_20261009_primary_laser_extended':'dense_laser_covariance_diagnostic_completed',
        'task_2_8_20261009_moving_secondary_coarse':'moving_secondary_symbol_diagnostic_completed',
        'task_2_8_20261009_moving_secondary_refined':'moving_secondary_symbol_diagnostic_completed',
        'task_2_8_20261009_independent_high_e_epoch_repaired':'selected_annual_reoptimization_completed',
        'task_2_8_20261009_optimized_contour_branch_repaired':'optimized_annual_crossing_bracket_completed',
        'task_2_8_20261009_band_campaign_branch_repaired':'combined_reference_band_controls_completed',
        'task_2_8_20261009_final_high_pm_cadence_branch_repaired':'selected_annual_reoptimization_completed',
        'task_2_8_20261012_high_pm_final_metric_independent':'selected_annual_reoptimization_completed'}
    pending,failed,inputs=[],[],{}
    for name,accepted in required.items():
        path=BASE/name/'validation_summary.json';value=read(path)
        if value is None:pending.append(name)
        elif value['status']!=accepted:failed.append(dict(run=name,status=value['status']))
        else:inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    refinements={}
    for case,suffix in [('near_contour','20261008_annual_refinement_near_contour_numerical'),
        ('high_e','20261008_annual_refinement_high_e'),('high_pm','20261009_annual_refinement_high_pm_branch_repaired')]:
        path=BASE/('task_2_8_'+suffix)/'annual_refinement.json';records=read(path)
        if not records:continue
        values={r['level']['name']:r['native_winner_metrics']['overlap'] for r in records}
        baseline=values['baseline_reopt']
        refinements[case]=dict(native_overlaps=values,delta_from_reoptimized_baseline={name:value-baseline for name,value in values.items()},
            native_spectral_winner_discrepancies={r['level']['name']:r['native_minus_spectral'] for r in records},
            scope='Selected optimized numerical/domain levels; no uniform continuum/global-search bound')
    independent={}
    for case in ('high_e','high_pm'):
        records=read(BASE/('task_2_8_20261009_independent_'+case+'_epoch_repaired')/'annual_refinement.json')
        if records:
            row=records[-1]
            independent[case]=dict(native_overlap=row['native_winner_metrics']['overlap'],
                archived_baseline_deficit=row['independent_baseline_deficit'],
                retained_candidates=row['retained_candidate_count'],
                independent_starts_only=row['independent_starts_only'],
                effective_chirp_seeds=row['effective_seed_chirp_points'],
                local_starts=[dict(overlap=r['overlap'],success=r['success'],origin=r['start_origin']) for r in row['local_runs']],
                scope='Selected winning frequency basin; sky, both masses, frequency/chirp and orientation free')
    finite={}
    for case,name in [('near_contour','20261008_finite_annual_optimized_near_contour'),
        ('high_e','20261008_finite_annual_optimized_high_e'),('high_pm_original','20261009_finite_annual_optimized_high_pm_repaired'),
        ('high_pm_refined','20261009_finite_high_pm_branch_refined'),('low_asymmetric','20261009_finite_low_asymmetric')]:
        folder=BASE/('task_2_8_'+name)
        value=read(folder/'fixed_candidate_comparison.json');profile=read(folder/'finite_orientation_profile.json')
        if value:
            finite[case]=dict(fixed_physical_candidate_delta=value['delta_overlap'],
                orientation_reprofiled_delta=profile['delta_overlap'] if profile else None,
                solve_records=(profile or value)['solve_records'],
                scope='Finite stationary covariance control at fixed intrinsic coordinates; not an optimized-family bound')
    band=read(BASE/'task_2_8_20261009_band_campaign_branch_repaired/band_campaign_results.json')
    contour=read(BASE/'task_2_8_20261009_optimized_contour_branch_repaired/validation_summary.json')
    covariance=read(BASE/'task_2_8_20261009_covariance_closure_grid/covariance_controls.json')
    laser=read(BASE/'task_2_8_20261009_primary_laser_extended/covariance_controls.json')
    moving={name:read(BASE/('task_2_8_20261009_moving_secondary_'+name)/'symbol_comparison.json') for name in ('coarse','refined')}
    measured_band_checks=[dict(case=r['case'],shift=r['band_shift'],absolute_shift=abs(r['band_shift']),
        original_allowance=.0002,allowance=.00021,within_allowance=abs(r['band_shift'])<=.00021,
        scope='Same-reference 5 s optimized comparison; not a uniform family or untested-band bound') for r in band or []]
    review=dict(updated_utc=datetime.now(timezone.utc).isoformat(),task_2_8_complete=False,
        production_configuration_frozen=False,execution_status='all_declared_controls_completed' if not pending and not failed else 'incomplete_or_requires_review',
        pending_runs=pending,failed_runs=failed,working_absolute_FF_target=.001,
        error_allocations=dict(noise_weighting=.0004,waveform=.0002,band_leakage=.00021,evolution=.00009,optimization=.0001),
        accepted_band_exception=dict(user_accepted=True,date='2026-10-12',
            observed_magnitude=.00020838839573467105,original_band_allocation=.0002,
            revised_band_allocation=.00021,revised_evolution_allocation=.00009,
            overall_working_target_unchanged=.001,
            rationale='Minor band overshoot accepted by the user; 1e-5 of the working budget is moved from evolution, whose selected tightening changes are below 1.7e-11'),
        optional_followups={'task_2_8_20261012_high_pm_band_cadence':read(BASE/'task_2_8_20261012_high_pm_band_cadence/validation_summary.json')},
        physical_reference=dict(laser=True,test_mass=True,OMS=True,clocks='ideal',ranges='ideal',foreground=False),
        annual_numerical_controls=refinements,independent_search_controls=independent,
        finite_stationary_controls=finite,covariance_numerical_controls=covariance,
        primary_laser_controls=laser,moving_secondary_controls=moving,band_campaign=band,optimized_crossing=contour,
        final_high_pm_cadence=read(BASE/'task_2_8_20261009_final_high_pm_cadence_branch_repaired/annual_refinement.json'),
        replacement_branch_agreement=read(BASE/'task_2_8_20261009_branch_agreement_review/branch_agreement.json'),
        observed_band_allocation_checks=measured_band_checks,
        finer_band_cadence_followup=read(BASE/'task_2_8_20261012_high_pm_band_cadence/annual_refinement.json'),
        final_metric_independent_followup=read(BASE/'task_2_8_20261012_high_pm_final_metric_independent/annual_refinement.json'),
        superseded_high_pm_controls=dict(native_overlap=.47083613744963976,
            blind_native_overlap=.5245472592777878,improvement=.05371112182814802,
            original_failures_and_nine_old_refinement_levels_preserved=True,
            valid_as_selected_candidate_controls_only=True),
        completed_summary_hashes=inputs,
        final_review_required=['Review empirical optimizer and combined-reference convergence, including active boundaries and failed local terminations',
            'Distinguish all-bin specified-metric bounds from sampled physical-model controls and fixed-candidate finite covariance',
            'Review same-family added-band bounds separately from searches over newly admitted harmonic basins',
            'Justify the adopted annual PSD/regularization and state stationary/adiabatic idealized-reference scope',
            'Assign conservative combined FF and crossing uncertainty before freezing the reproducible configuration'],
        limitations=['No automatic certification from successful child executions or passing software tests',
            'Finite starts/basins give empirical global-search evidence, not a rigorous global maximum theorem',
            'Full native sampled-instrument arithmetic has not been repaired by the deterministic covariance workaround'])
    completion=reviewed_completion()
    review['saved_completion_review']=completion
    if completion and completion['accepted'] and not pending and not failed:
        review.update(task_2_8_complete=True,production_configuration_frozen=True,
            production_grid_completed=False,completion_scope=completion['scope'],
            final_review_required=[],gate_status=completion['gate_status'])
    temporary=out/'closure_review.tmp';temporary.write_text(json.dumps(review,indent=2)+'\n');temporary.replace(out/'closure_review.json')
    print(json.dumps(dict(status=review['execution_status'],pending=len(pending),failed=failed,
        review=str((out/'closure_review.json').relative_to(ROOT)),task_2_8_complete=review['task_2_8_complete']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=BASE/'task_2_8_20261009_closure_review')
    run(p.parse_args())
