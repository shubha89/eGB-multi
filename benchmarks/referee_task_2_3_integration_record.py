"""Verify and record a completed Task 2.3 reference-integration run."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(run, test_log):
    run, test_log = run.resolve(), test_log.resolve()
    if not run.is_relative_to(ROOT) or not test_log.is_relative_to(ROOT):
        raise ValueError('All evidence must remain inside this checkout')
    manifest = json.loads((run/'manifest.json').read_text())
    summary = json.loads((run/'validation_summary.json').read_text())
    if summary['failures'] or summary['cases'] != 6 or summary['noise_model'] != 'secondary_plus_laser':
        raise ValueError('Adopted reference did not pass all six cases')
    checked = []
    for relative, expected in manifest['sha256'].items():
        if sha256(run/'snapshot'/relative) != expected or sha256(ROOT/relative) != expected:
            raise ValueError(f'Snapshot/current source drift: {relative}')
        checked.append(relative)
    contract = json.loads((run/'clean_response_contract.json').read_text())
    with np.load(run/'noise_and_delays.npz') as noise, np.load(run/'clean_times_and_delays.npz') as clean:
        np.testing.assert_array_equal(noise['t'],clean['t'])
        delays = {label:float(np.max(abs(noise[f'delay_{label}_s']-clean[f'delay_{label}_s'])))
            for label in ('21','32','13','31','23','12')}
    if max(delays.values()) > contract['ideal_reference_agreement_tolerance_s']:
        raise ValueError('Ideal-reference delays fail agreement check')
    text = test_log.read_text()
    matches = re.findall(r'(\d+) passed.* in ([\d.]+)s', text)
    if not matches or 'FAILED ' in text:
        raise ValueError('A completed passing test suite is required')
    laser = []
    # The low-frequency circular case requests 20 microHz, below the earlier
    # 0.1 mHz laser audit. Reassess archived paired PSDs across this full band.
    for seed in (20261007,20261008,20261009):
        path = ROOT/f'outputs/referee/laser_on_off_20261007/secondary_seed{seed}/paired_psd.csv'
        data = np.genfromtxt(path,delimiter=',',names=True)
        selected = (data['frequency_hz']>=2e-5)&(data['frequency_hz']<=.03)
        row = {'seed':seed,'band_hz':[2e-5,.03], 'paired_PSD_sha256':sha256(path)}
        for channel in ('A','E'):
            ratio = data[f'PSD_{channel}_increment'][selected]/data[f'PSD_{channel}_off'][selected]
            ids = np.flatnonzero(selected)
            row[channel] = {'max_increment_PSD_to_laser_off':float(ratio.max()),
                'maximum_at_frequency_hz':float(data['frequency_hz'][ids[np.argmax(ratio)]]),
                'p95_abs_fractional_total_PSD_change':float(np.quantile(abs(
                    data[f'PSD_{channel}_on'][selected]/data[f'PSD_{channel}_off'][selected]-1.),.95))}
        laser.append(row)
    value = {'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'passed_implementation_and_reference_integration_checks',
        'checked_snapshot_and_current_source_files':checked,
        'tests':{'passed':int(matches[-1][0]),'wall_s':float(matches[-1][1]),
            'log':str(test_log.relative_to(ROOT)), 'log_sha256':sha256(test_log)},
        'noise_MPRS_minus_clean_delay_max_abs_s_by_link':delays,
        'paired_laser_band_extension':laser,
        'production_covariance_and_PSD_certified':False,
        'scope':'Task 2.3 implementation/reference integration and fixed-intrinsic 2.4 confirmation; production convergence remains 2.8',
        'record_script_sha256':sha256(Path(__file__))}
    (run/'integration_validation_record.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    shutil.copy2(Path(__file__),run/'referee_task_2_3_integration_record.py')
    print(json.dumps(value,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--test-log',type=Path,required=True)
    args = parser.parse_args()
    record(args.run,args.test_log)
