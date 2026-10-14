"""Audit archived blind searches against replacement annual baselines.

No optimization is manufactured or relabelled. The failed comparison to an
older, inferior baseline stays preserved. This separate review checks the
native results of those same independent searches against a replacement
baseline calculated with the same physical objective and injection.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def run(args):
    out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'review_driver.py')
    pairs=[('high_e',args.high_e_baseline,args.high_e_independent),
           ('high_pm',args.high_pm_baseline,args.high_pm_independent)]
    hashes={};rows=[];failures=[]
    for case,baseline_path,independent_path in pairs:
        baseline_path,independent_path=baseline_path.resolve(),independent_path.resolve()
        if not all(p.is_relative_to(ROOT) for p in (baseline_path,independent_path)):
            raise ValueError('Use local calculations')
        files=[baseline_path/(case+'_annual_search.json'),baseline_path/'manifest.json',
            baseline_path/'validation_summary.json',independent_path/'annual_refinement.json',
            independent_path/'manifest.json',independent_path/'validation_summary.json']
        for p in files:hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        baseline=json.loads(files[0].read_text());bm=json.loads(files[1].read_text())
        bs=json.loads(files[2].read_text());records=json.loads(files[3].read_text())
        im=json.loads(files[4].read_text());archive=json.loads(files[5].read_text())
        if bs['status']!='annual_baseline_completed':failures.append(case+': baseline incomplete')
        # Require the same immutable injection, covariance, band and native
        # conventions; a noise/cadence change needs a new independent run.
        objective_files=[case+'_source.json',case+'_signal.npz','annual_covariance.npz']
        for filename in objective_files:
            b={v for k,v in bm['inputs'].items() if Path(k).name==filename}
            i={v for k,v in im['inputs'].items() if Path(k).name==filename}
            if len(b)!=1 or b!=i:failures.append(case+': different objective input '+filename)
        for row in records:
            level=row['level']
            if level['band_hz']!=bm['configuration']['band']:
                failures.append(case+': physical band differs')
            if (level['dt_s']!=10. or level['measurement_order']!=81 or
                level['pm_ode_rtol']!=1e-10 or level['pm_ode_phase_atol']!=1e-8 or
                level['include_primary_laser'] or level['diagonal_weighting']):
                failures.append(case+': independent metric/native convention differs')
            deficit=baseline['native_winner_metrics']['overlap']-row['native_winner_metrics']['overlap']
            fits=row['local_runs']
            recoveries=sum(abs(fit['overlap']-row['native_winner_metrics']['overlap'])<=1e-4 for fit in fits)
            if (not row['independent_starts_only'] or row['retained_candidate_count']!=0 or
                not row['all_six_intrinsic_coordinates_free'] or abs(deficit)>1e-4 or
                abs(row['native_minus_spectral'])>2e-4 or len(fits)<6 or recoveries<2):
                failures.append(case+': blind branch agreement not established')
            rows.append(dict(case=case,baseline_run=str(baseline_path.relative_to(ROOT)),
                independent_run=str(independent_path.relative_to(ROOT)),native_baseline_overlap=baseline['native_winner_metrics']['overlap'],
                native_independent_overlap=row['native_winner_metrics']['overlap'],baseline_minus_independent=deficit,
                independent_native_minus_spectral=row['native_minus_spectral'],blind_starts=True,
                retained_templates=0,local_fits=len(fits),fits_recovering_native_winner_within_allowance=recoveries,
                original_validation_status=archive['status'],
                original_failed_comparison_preserved=True,unchanged_objective_inputs=True))
    save(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),inputs=hashes,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Reassessment of archived independent calculations; no new optimizer run'))
    save(out/'branch_agreement.json',rows)
    save(out/'validation_summary.json',dict(status='passed_archived_independent_branch_agreement' if not failures else 'requires_investigation',
        failures=failures,production_certified=False,independent_controls_reused_without_relabelling=True,
        scope='Six blind local starts in each declared winning basin; empirical agreement, not global proof'))
    if failures:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    for case in ('high_e','high_pm'):
        p.add_argument('--'+case.replace('_','-')+'-baseline',type=Path,required=True)
        p.add_argument('--'+case.replace('_','-')+'-independent',type=Path,required=True)
    run(p.parse_args())
