"""Verify, summarize and plot the four independent Task 2.5 validation runs."""

import argparse
from datetime import datetime,timezone
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
CASES = ('circular_positive_offset','circular_negative_offset','eccentric_fixed','eccentric_pm')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(output,runs,test_log):
    output,runs,test_log = output.resolve(),[path.resolve() for path in runs],test_log.resolve()
    if any(not path.is_relative_to(ROOT) for path in (output,test_log,*runs)):
        raise ValueError('All paths must be inside this checkout')
    rows,verified,inputs,plot_data = [],[],{},[]
    signatures = []
    for run in runs:
        summary = json.loads((run/'validation_summary.json').read_text())
        manifest = json.loads((run/'manifest.json').read_text())
        if summary['failures'] or summary['status']!='passed_staged_validation':
            raise ValueError(f'Failed validation must be investigated: {run}')
        signatures.append((manifest['sha256'],manifest['configuration']['samples'],manifest['configuration']['dt']))
        for name,expected in manifest['sha256'].items():
            if digest(ROOT/name)!=expected or digest(run/'snapshot'/name)!=expected:
                raise ValueError(f'Source/snapshot drift: {run}: {name}')
        verified.append({'run':str(run.relative_to(ROOT)),'snapshot_and_current_hashes_checked':len(manifest['sha256'])})
        input_path = run/'frequency_chirp_validation.json'
        inputs[str(input_path.relative_to(ROOT))] = digest(input_path)
        for entry in json.loads(input_path.read_text()):
            best = entry['refined_search']['best']
            starts = entry['initial_search']['independent_start_results']
            row = {'case':entry['case'],'duration_s':entry['refined_search']['duration_s'],
                'samples_before_crop':entry['samples_before_crop'],'dt_s':entry['dt_s'],
                'restricted_best_overlap':entry['restricted_best_overlap'],
                'continuous_profile_overlap':best['profile']['overlap'],
                'explicit_native_overlap':entry['explicit_native_recovered_metrics']['overlap'],
                'frequency_hz_at_midpoint':best['frequency_hz'],'fdot_hz_s':best['fdot_hz_s'],
                'scaled_recovery_error':entry['scaled_recovery_error'],
                'refinement_overlap_change':entry['refinement_overlap_change'],
                'refinement_coordinate_change':entry['refinement_coordinate_change'],
                'independent_start_overlap_spread':max(s['overlap'] for s in starts)-min(s['overlap'] for s in starts),
                'native_vs_cached_weighted_relative_error':entry['native_vs_cached_weighted_relative_error'],
                'explicit_minus_profile_overlap':entry['explicit_native_recovered_metrics']['overlap']-best['profile']['overlap'],
                'global_orientation_minus_profile_overlap':entry['global_orientation_at_best']['overlap']-best['profile']['overlap'],
                'off_maximum_global_profile_differences':[check['difference'] for check in entry['global_orientation_off_maximum_checks']],
                'unique_intrinsic_evaluations_through_refinement':entry['refined_search']['unique_intrinsic_evaluations'],
                'fitted_relative_residual':entry['explicit_native_recovered_metrics']['fitted_relative_residual'],
                'wall_s':entry['wall_s']}
            rows.append(row)
            history_path = run/f"{entry['case']}_evaluations.jsonl"
            inputs[str(history_path.relative_to(ROOT))] = digest(history_path)
            history = [json.loads(line) for line in history_path.read_text().splitlines()]
            plot_data.append((entry,history))
    if len(signatures)!=4 or any(signature!=signatures[0] for signature in signatures[1:]):
        raise ValueError('Four runs with identical source/PSD hashes and sampling required')
    if sorted(row['case'] for row in rows)!=sorted(CASES):
        raise ValueError('Exactly the four declared cases are required')
    matches = re.findall(r'(\d+) passed.* in ([\d.]+)s',test_log.read_text())
    if not matches or 'FAILED ' in test_log.read_text():
        raise ValueError('Completed passing test suite required')
    output.mkdir(parents=True,exist_ok=False)
    record = {'created_utc':datetime.now(timezone.utc).isoformat(),'status':'passed_Task_2_5_staged_validation',
        'rows':rows,'verified_runs':verified,'input_sha256':inputs,
        'tests':{'passed':int(matches[-1][0]),'wall_s':float(matches[-1][1]),
            'log':str(test_log.relative_to(ROOT)),'sha256':digest(test_log)},
        'summary_script_sha256':digest(Path(__file__)),
        'scope':'Bounded continuous frequency/chirp and physical orientation fitting at fixed sky and masses',
        'production_convergence_certified':False,'full_fitting_factor_certified':False,
        'limitations':['One local carrier basin; sky/mass/harmonic searches remain 2.6',
            'Short deliberately measurable circular chirps validate the algorithm rather than an astrophysical population',
            'One fixed finite diagonal reference PSD; PSD/covariance/cadence certification remains 2.8']}
    (output/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    shutil.copy2(Path(__file__),output/'referee_task_2_5_summary.py')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(9,7.4),layout='constrained',sharex=True,sharey=True)
    for ax,name in zip(axes.flat,CASES,strict=True):
        entry,history = next(value for value in plot_data if value[0]['case']==name)
        shape = entry['initial_search']['coarse_shape']
        alpha = np.linspace(*entry['refined_search']['frequency_scaled_bounds'],shape[0])
        gamma = np.linspace(*entry['refined_search']['chirp_scaled_bounds'],shape[1])
        lookup = {tuple(point['coordinates']):point['overlap'] for point in history}
        scores = np.array([[lookup[(float(a),float(g))] for a in alpha] for g in gamma])
        mesh = ax.pcolormesh(alpha,gamma,scores,shading='nearest',vmin=0,vmax=1,cmap='viridis',rasterized=True)
        a,g = entry['refined_search']['best']['coordinates']
        ax.plot(a,g,'*',color='white',markeredgecolor='black',markersize=12,label='Continuous best fit')
        ax.set_title(name.replace('_',' '),fontsize=10)
        ax.set_xlim(-3.5,3.5);ax.set_ylim(-4,4)
        ax.set_xlabel(r'$(f_{\rm ref}-f_{\rm center})T$')
        ax.set_ylabel(r'$(\dot f-\dot f_{\rm center})T^2$')
    axes[0,0].legend(fontsize=8,loc='lower left')
    fig.colorbar(mesh,ax=axes,label='Physical orientation-profile overlap at coarse grid points')
    fig.suptitle('Task 2.5: one carrier basin, fixed sky and masses\nRaw coarse-grid values; no contour interpolation',fontsize=11)
    fig.savefig(output/'frequency_chirp_validation.png',dpi=180)
    fig.savefig(output/'frequency_chirp_validation.pdf')
    plt.close(fig)
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--test-log',type=Path,required=True)
    args = parser.parse_args()
    summarize(args.output,args.runs,args.test_log)
