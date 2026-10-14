"""Collect finished Task 2.7/2.8 evidence without counting active jobs as passes."""

from datetime import datetime,timezone
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
import numpy as np
import matplotlib.pyplot as plt


def run():
    out=ROOT/'outputs/referee/task_2_7_2_8_20261008_summary';out.mkdir(exist_ok=True)
    names=['task_2_7_20261008_sky_partner_probe','task_2_7_20261008_small_e_dense',
        'task_2_7_20261008_small_e_refined_prefix','task_2_7_20261008_small_e_refined_tail',
        'task_2_7_20261008_independent_circular','task_2_7_20261008_independent_circular_guarded',
        'task_2_7_20261008_independent_circular_constrained','task_2_7_20261008_independent_e01',
        'task_2_7_20261008_independent_e03_union','task_2_7_20261008_independent_e06_union',
        'task_2_7_20261008_independent_e06_expanded','task_2_8_20261008_year_low',
        'task_2_8_20261008_year_high','task_2_8_20261008_year_high_mass',
        'task_2_8_20261008_year_evolution_precision','task_2_8_20261008_noise_ensemble',
        'task_2_8_20261008_year_envelope_validation','task_2_8_20261008_year_envelope_order81',
        'task_2_8_20261008_covariance_grid','task_2_8_20261008_covariance_physics_controls',
        'task_2_8_20261008_covariance_physics_refined','task_2_8_20261008_annual_laser_residual',
        'task_2_8_20261008_laser_fixed_30day_cutoff',
        'task_2_8_20261008_laser_fixed_30day_other_epochs',
        'task_2_8_20261008_cutoff_weighting','task_2_8_20261008_frozen_noise_diagnostic',
        'task_2_8_20261008_noise_discretization','task_2_8_20261008_moving_covariance_symbol',
        'task_2_8_20261008_laser_superposition','task_2_8_20261008_laser_arithmetic',
        'task_2_8_20261008_laser_input_scale','task_2_8_20261008_band_pool']
    states=[]
    for name in names:
        folder=ROOT/'outputs/referee'/name
        summary=next((folder/file for file in ('validation_summary.json','diagnostic_summary.json','symbol_comparison.json')
            if (folder/file).exists()),folder/'validation_summary.json')
        states.append({'folder':str(folder.relative_to(ROOT)),
            'status_record':json.loads(summary.read_text()) if summary.exists() else None,
            'record_type':summary.name if summary.exists() else None,
            'interpretation':'scope-specific status recorded; diagnostic completion is not certification' if summary.exists()
                else 'no final status; in progress or not completed'})
    optimizer_rows=[]
    for suffix in ('independent_circular_guarded','independent_e01','independent_e03_union','independent_e06_expanded'):
        folder=ROOT/'outputs/referee'/('task_2_7_20261008_'+suffix)
        path=folder/'optimizer_validation.json'
        if not path.exists():continue
        for record in json.loads(path.read_text()):
            maximum=record['explicit_native_metrics']['overlap']
            same_intrinsics=all(record['prototype'][key]==record['injection'][key]
                for key in ('mean_motion','m1_solar','m2_solar','beta','lambda_'))
            optimizer_rows.append({'case':record['case'],'native_overlap':maximum,
                'same_objective_prototype_restricted_baseline':record['baseline_profile_overlap'],
                'baseline_at_injection_intrinsics':same_intrinsics,
                'baseline_scope':'Nine legacy chirp factors applied at the recorded prototype; the offset circular prototype is deliberately incorrect',
                'retained_archived_overlap':record['retained_archived_overlap'],
                'independent_seed_best':record['independent_seed_best'],
                'largest_independent_seed_deficit':maximum-min(record['independent_seed_best'].values()),
                'basins':len(record['basin_selection']['basins']),
                'fixed_band_hz':record['refined_search']['fixed_band_hz'],
                'intrinsic_evaluations':record['refined_search']['unique_intrinsic_evaluations'],
                'evidence':str(path.relative_to(ROOT))})
    small=[]
    for suffix in ('small_e_refined_prefix','small_e_refined_tail'):
        path=ROOT/'outputs/referee'/('task_2_7_20261008_'+suffix)/'jacobian_refinement.json'
        if path.exists():
            small.extend({'eccentricity':r['injection']['eccentricity'],
                'native_overlap':r['explicit_native_metrics']['overlap'],
                'fitted_relative_residual':r['explicit_native_metrics']['fitted_relative_residual'],
                'evidence':str(path.relative_to(ROOT))} for r in json.loads(path.read_text()))
    (out/'optimizer_checks.json').write_text(json.dumps({'archived_cases':optimizer_rows,'small_e':small},indent=2)+'\n')
    required_optimizer=('small_e_refined_prefix','small_e_refined_tail','independent_circular_guarded',
        'independent_e01','independent_e03_union','independent_e06_expanded')
    optimizer_complete=True
    for suffix in required_optimizer:
        path=ROOT/'outputs/referee'/('task_2_7_20261008_'+suffix)/'validation_summary.json'
        record=json.loads(path.read_text()) if path.exists() else {}
        if not record.get('status','').startswith('passed_declared_') or record.get('failures'):
            optimizer_complete=False
    value={'updated_utc':datetime.now(timezone.utc).isoformat(),'task_2_7_complete':optimizer_complete,'task_2_8_complete':False,
        'task_2_7_scope':'Short native complete-optimizer audit; finite empirical basin/start validation, not annual production certification',
        'production_certified':False,'runs':states,
        'pending_decision':'Production PSD estimator and fixed low-frequency noise regularization; deterministic alternative remains diagnostic',
        'provisional_total_absolute_overlap_budget':.001}
    (out/'work_state.json').write_text(json.dumps(value,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(6.3,4.1),constrained_layout=True)
    for name,label,marker in [('low','0.1 mHz, e=0.1','o'),('high','3 mHz, e=0.5, 0.8+0.6 solar masses','s'),('high_mass','3 mHz, e=0.5, 1.0+0.3 solar masses','^')]:
        path=ROOT/f'outputs/referee/task_2_8_20261008_year_{name}/numerical_validation.json'
        if not path.exists():continue
        record=json.loads(path.read_text())[0]
        rows=[r for r in record['comparisons'] if r['dt_s']==10]
        ax.semilogy([r['measurement_order'] for r in rows],[r['pointwise_overlap_change_bound'] for r in rows],marker+'-',label=label)
    ax.axhline(2e-4,color='.25',ls='--',label='Provisional waveform allowance')
    ax.set(xlabel='TDI measurement interpolation order at 10 s',ylabel='Pointwise overlap-change bound',title='One-year native waveform convergence')
    ax.set_xticks([15,31,45,61]);ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper right')
    fig.savefig(out/'annual_waveform_convergence.png',dpi=180);fig.savefig(out/'annual_waveform_convergence.pdf')
    plt.close(fig)
    if small:
        fig,ax=plt.subplots(figsize=(5.4,3.7),constrained_layout=True)
        positive=sorted((r for r in small if r['eccentricity']>0),key=lambda r:r['eccentricity'])
        ax.loglog([r['eccentricity'] for r in positive],[1-r['native_overlap'] for r in positive],'o-')
        circular=next((r for r in small if r['eccentricity']==0),None)
        if circular:ax.text(.04,.95,f"Circular fitted residual: {circular['fitted_relative_residual']:.2g}",
            transform=ax.transAxes,va='top',fontsize=9)
        ax.set(xlabel='Injection eccentricity',ylabel='1 - recovered overlap',title='Small-eccentricity native optimizer audit')
        ax.grid(alpha=.2,which='both')
        fig.savefig(out/'small_e_recovery.png',dpi=180);fig.savefig(out/'small_e_recovery.pdf');plt.close(fig)


if __name__=='__main__':run()
