"""Uniform all-bin bounds between declared annual pointwise A/E metrics.

These bounds hold for all waveform pairs and maxima over the same family.
They do not control finite-window covariance or unmodeled physical noise.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from egb_multi.metric_perturbation import relative_covariance_eigenvalues, normalized_overlap_change_bound
from referee_task_2_objective_orientation import assert_local, write_json


def load(path, f):
    with np.load(path) as data:
        frequencies, covariance = data['frequency_hz'], data['covariance']
    if frequencies[0] > f[0] or frequencies[-1] < f[-1]:
        raise ValueError('No covariance extrapolation permitted')
    return np.stack([np.interp(f, frequencies, covariance[:, a, b]) for a in range(2) for b in range(2)], axis=-1).reshape(-1, 2, 2)


def run(args):
    assert_local()
    out, secondary, laser = [p.resolve() for p in (args.output, args.secondary, args.laser)]
    if not all(p.is_relative_to(ROOT) for p in (out, secondary, laser)):
        raise ValueError('Use local files')
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in [ROOT/'src/egb_multi/metric_perturbation.py', Path(__file__), ROOT/'benchmarks/referee_task_2_objective_orientation.py']:
        relative = path.relative_to(ROOT); target = out/'snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    first, last = int(np.ceil(args.band[0]*args.duration)), int(np.floor(args.band[1]*args.duration))
    frequency = np.arange(first, last+1)/args.duration
    write_json(out/'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), sha256=hashes,
        inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (secondary, laser)},
        duration_s=args.duration, band_hz=args.band, annual_bins=len(frequency),
        scope='Uniform bounds between implemented pointwise covariance metrics only', production_certified=False))
    reference = load(secondary, frequency); addition = load(laser, frequency)
    records = []
    for name, candidate in [('include_physical_primary_laser', reference+addition),
                            ('diagonal_AE', reference*np.eye(2)[None])]:
        eigenvalues = relative_covariance_eigenvalues(reference, candidate)
        errors = np.max(abs(eigenvalues-1.), axis=1)
        epsilon = float(errors.max())
        row = dict(control=name, complete_annual_bins=len(frequency), relative_covariance_epsilon=epsilon,
            maximum_error_frequency_hz=float(frequency[np.argmax(errors)]),
            uniform_absolute_overlap_and_same_family_maximum_bound=normalized_overlap_change_bound(epsilon))
        records.append(row); print(row, flush=True)
    write_json(out/'metric_bounds.json', records)
    write_json(out/'validation_summary.json', dict(status='completed_declared_all_bin_metric_bounds', production_certified=False,
        limitations=['Bounds control specified interpolated pointwise metrics, not precision of the physical covariance model',
                     'Full finite or time-dependent covariance errors are separate',
                     'A/E diagonal comparison is a sensitivity control; full covariance remains the reference']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--secondary', type=Path, default=ROOT/'outputs/referee/task_2_8_20261008_annual_weighting_pilot/annual_covariance.npz')
    p.add_argument('--laser', type=Path, default=ROOT/'outputs/referee/task_2_8_20261008_laser_covariance_grid/annual_laser_covariance_dt150.npz')
    p.add_argument('--duration', type=float, default=31557600.)
    p.add_argument('--band', type=float, nargs=2, default=[2e-5, .04])
    run(p.parse_args())
