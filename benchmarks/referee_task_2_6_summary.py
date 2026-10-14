"""Verify and summarize native Task 2.6 searches without changing their results."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/referee/.cache'))
CASES = ('circular_unknown_sky_mass','eccentric_fixed','eccentric_pm','eccentric_high_e')
TITLES = {'circular_unknown_sky_mass':'Circular: unknown sky and masses',
    'eccentric_fixed':'Fixed evolution: e = 0.1',
    'eccentric_pm':'Peters–Mathews evolution: e = 0.3',
    'eccentric_high_e':'Fixed evolution: e = 0.6'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(output,runs,test_log):
    output,runs,test_log = output.resolve(),[path.resolve() for path in runs],test_log.resolve()
    if any(not path.is_relative_to(ROOT) for path in (output,test_log,*runs)):
        raise ValueError('All paths must lie in this checkout')
    rows,verified,inputs,entries,signatures = [],[],{},{},[]
    for run in runs:
        summary = json.loads((run/'validation_summary.json').read_text())
        manifest = json.loads((run/'manifest.json').read_text())
        if summary['failures'] or summary['status']!='passed_Task_2_6_staged_validation':
            raise ValueError(f'Validation failure needs investigation: {run}')
        signatures.append((manifest['sha256'],manifest['configuration']['samples'],manifest['configuration']['dt']))
        for name,expected in manifest['sha256'].items():
            if digest(ROOT/name)!=expected or digest(run/'snapshot'/name)!=expected:
                raise ValueError(f'Current source/snapshot drift: {run}: {name}')
        verified.append({'run':str(run.relative_to(ROOT)),'snapshot_and_current_hashes_checked':len(manifest['sha256'])})
        result_path = run/'sky_mass_harmonic_validation.json'
        inputs[str(result_path.relative_to(ROOT))] = digest(result_path)
        for entry in json.loads(result_path.read_text()):
            name = entry['case']
            if name in entries:
                raise ValueError(f'Duplicate case: {name}')
            entries[name] = entry
            history_path = run/f'{name}_evaluations.jsonl'
            inputs[str(history_path.relative_to(ROOT))] = digest(history_path)
            final = entry['final_refinement']
            best = final['best']
            basins = entry['expanded_family_search']['basins_results']
            winning = min(basins,key=lambda b:abs(b['basin']['center_hz']-best['frequency_hz']))
            starts = winning['independent_start_results']
            rows.append({'case':name,'samples_before_crop':entry['samples_before_crop'],'dt_s':entry['dt_s'],
                'duration_s':final['duration_s'],'fixed_sky_mass_overlap':entry['fixed_sky_mass_baseline']['best']['profile']['overlap'],
                'initial_family_overlap':entry['initial_refinement']['best']['profile']['overlap'],
                'expanded_family_overlap':best['profile']['overlap'],'explicit_native_overlap':entry['explicit_native_metrics']['overlap'],
                'expansion_overlap_change':entry['expansion_overlap_change'],
                'refinement_overlap_change':entry['final_refinement_overlap_change'],
                'initial_basins':len(entry['initial_family_search']['basins_results']),
                'expanded_basins':len(basins),'winning_frequency_hz':best['frequency_hz'],'winning_fdot_hz_s':best['fdot_hz_s'],
                'winning_frequency_over_nominal_carrier':entry['winning_frequency_over_nominal_carrier'],
                'nearest_labelled_quadrupole_sideband':entry['nearest_labelled_quadrupole_sideband'],
                'winning_sky_rad':[best['source']['beta'],best['source']['lambda_']],
                'winning_masses_solar':[best['source']['m1_solar'],best['source']['m2_solar']],
                'mass_shape_max_weighted_relative_residual_at_best':max(m['amplitude_fitted_weighted_shape_residual'] for m in entry['mass_shape_audit_at_best']),
                'initial_seed_spectrum_coverage':entry['initial_basin_selection']['covered_seed_spectrum_power_fraction'],
                'expanded_seed_spectrum_coverage':entry['expanded_basin_selection']['covered_seed_spectrum_power_fraction'],
                'native_vs_cached_weighted_shape_residual':entry['native_vs_cached_weighted_shape_residual'],
                'global_minus_profile_overlap':entry['global_orientation_check']['overlap']-best['profile']['overlap'],
                'explicit_minus_profile_overlap':entry['explicit_native_metrics']['overlap']-best['profile']['overlap'],
                'winning_basin_independent_start_spread':max(s['overlap'] for s in starts)-min(s['overlap'] for s in starts),
                'winning_basin_start_successes':[s['success'] for s in starts],
                'winning_frequency_chirp_boundary_distance_scaled':winning['frequency_chirp_boundary_distance_scaled'],
                'circular_recovery_sky_direction_error':entry['circular_recovery_sky_direction_error'],
                'initial_unique_intrinsic_evaluations':entry['initial_refinement']['unique_intrinsic_evaluations'],
                'expanded_unique_intrinsic_evaluations':final['unique_intrinsic_evaluations'],
                'wall_s':entry['wall_s']})
    if len(signatures)!=4 or any(signature!=signatures[0] for signature in signatures[1:]):
        raise ValueError('Four runs with identical source/PSD hashes and sampling required')
    if set(entries)!=set(CASES):
        raise ValueError('All four declared cases required')
    matches = re.findall(r'(\d+) passed.* in ([\d.]+)s',test_log.read_text())
    if not matches or 'FAILED ' in test_log.read_text():
        raise ValueError('Completed passing test suite required')
    output.mkdir(parents=True,exist_ok=False)
    record = {'created_utc':datetime.now(timezone.utc).isoformat(),'status':'passed_Task_2_6_staged_validation',
        'rows':rows,'verified_runs':verified,'input_sha256':inputs,
        'tests':{'passed':int(matches[-1][0]),'wall_s':float(matches[-1][1]),
            'log':str(test_log.relative_to(ROOT)),'sha256':digest(test_log)},
        'summary_script_sha256':digest(Path(__file__)),
        'scope':'Full sky and ordered masses with physical orientation, frequency/chirp and finite spectral basins',
        'full_fitting_factor_certified':False,'production_convergence_certified':False,
        'limitations':['Short native observation and finite diagonal reference PSD',
            'Seed-spectrum coverage is not an optimizer error bound; comprehensive validation remains 2.7',
            'Native sky/mass parameter recovery may be degenerate; no mass-fixed reduction is certified',
            'Cadence, PSD/covariance, bandwidth and production domain certification remain 2.8']}
    (output/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    shutil.copy2(Path(__file__),output/'referee_task_2_6_summary.py')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(9,7),layout='constrained')
    for ax,name in zip(axes.flat,CASES,strict=True):
        entry = entries[name]
        for key,marker,label in (('initial_family_search','o','Initial bounds and starts'),
                                  ('expanded_family_search','s','Expanded bounds and starts')):
            basins = entry[key]['basins_results']
            ax.scatter([b['best']['frequency_hz']*1000 for b in basins],
                [b['best']['profile']['overlap'] for b in basins],s=24,marker=marker,label=label)
        best = entry['final_refinement']['best']
        ax.plot(best['frequency_hz']*1000,best['profile']['overlap'],'*',color='black',markersize=12,label='Best retained fit')
        ax.axhline(entry['fixed_sky_mass_baseline']['best']['profile']['overlap'],color='gray',linestyle=':',label='Fixed sky and masses')
        ax.set_title(TITLES[name],fontsize=10)
        ax.set_xlabel('Circular frequency at midpoint [mHz]')
        ax.set_ylabel('Coherent physical overlap')
        ax.set_ylim(0,1.02)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        if name=='circular_unknown_sky_mass':
            # A single circular peak leaves nearly coincident winning points.
            # Show its declared basin rather than a roundoff-scale axis zoom.
            basins = entry['expanded_family_search']['basins_results']
            duration = entry['final_refinement']['duration_s']
            ax.set_xlim(min(b['basin']['center_hz']-b['basin']['half_width_cycles']/duration for b in basins)*1000,
                max(b['basin']['center_hz']+b['basin']['half_width_cycles']/duration for b in basins)*1000)
    axes[0,0].legend(fontsize=7,loc='lower left')
    fig.suptitle('Task 2.6: sky, masses and multiple frequency basins\nOne fixed injection normalization per case',fontsize=11)
    fig.savefig(output/'sky_mass_harmonic_validation.png',dpi=180)
    fig.savefig(output/'sky_mass_harmonic_validation.pdf')
    plt.close(fig)
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--test-log',type=Path,required=True)
    args = parser.parse_args()
    summarize(args.output,args.runs,args.test_log)
