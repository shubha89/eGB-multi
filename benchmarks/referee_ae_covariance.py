"""Finite-ensemble A/E covariance diagnostic for the adopted reference.

Reuses the three verified, laser-enabled, ideal-clock/ranging realizations.
Shorter Welch segments increase averaging at the cost of frequency resolution.
No frequency smoothing, covariance shrinkage or eigenvalue flooring is used.
This estimates a diagnostic covariance, not a certified population PSD.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.signal import csd, get_window, welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from egb_multi.comparison import AEPSD


def load_diagnostic_psds(path, convention):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Covariance diagnostic must be inside this checkout')
    record = json.loads(path.with_suffix('.json').read_text())
    if record['noise_model'] != 'secondary_plus_laser' or record['dt_s'] <= 0:
        raise ValueError('Unexpected covariance reference model')
    if (convention.generation, convention.measurement_order, convention.delay_order, convention.unit) != (2,45,5,'fractional_frequency'):
        raise ValueError('Diagnostic covariance uses TDI2/order45/delay-order5 fractional-frequency responses')
    provenance = dict(record, sidedness='one_sided', scaling='density',
        CSD_convention='S_AE=E[tilde_A*conj(tilde_E)]')
    with np.load(path) as arrays:
        f, a, e, cross = [arrays[key].copy() for key in ('frequency_hz','PSD_A','PSD_E','CSD_AE')]
    return {'diagonal':AEPSD(f, {'A':a, 'E':e}, convention, provenance),
            'full_covariance':AEPSD(f, {'A':a, 'E':e}, convention, provenance, cross)}


def build(output, source, nperseg):
    output, source = output.resolve(), source.resolve()
    if not output.is_relative_to(ROOT) or not source.is_relative_to(ROOT):
        raise ValueError('All inputs and outputs must be inside this checkout')
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source/'manifest.json').read_text())
    configuration = manifest['configuration']
    dt, seeds = configuration['dt'], configuration['seeds']
    spectra, segments, hashes = [], [], {}
    for seed in seeds:
        path = source/f'secondary_seed{seed}/paired_noise.npz'
        metadata_path = path.parent/'on_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        # Original paired-run metadata predates the optional control field.
        if metadata['model'] != 'secondary' or not metadata['laser_on'] or metadata.get('disabled_noises', []):
            raise ValueError('Covariance inputs must be the laser-enabled ideal-clock/ranging reference')
        hashes[str(metadata_path.relative_to(ROOT))] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as data:
            t, a, e = data['t'], data['on_A_fractional'], data['on_E_fractional']
            if not np.allclose(np.diff(t), dt, rtol=0, atol=1e-9):
                raise ValueError('Archived noise cadence differs')
            if a.size < 4*nperseg:
                raise ValueError('At least four complete segments required per realization')
            kw = dict(fs=1./dt, nperseg=nperseg, noverlap=nperseg//2,
                window='hann', detrend='constant', scaling='density')
            f, sa = welch(a, **kw)
            _, se = welch(e, **kw)
            _, cross = csd(a, e, **kw)
            spectra.append((sa, se, cross.conj()))
            segments.append(1+(a.size-nperseg)//(nperseg-nperseg//2))
    # All realizations have equal duration/segment count: equal ensemble weights.
    if len(set(segments)) != 1:
        raise ValueError('Unequal segment counts require explicit ensemble weighting')
    mean = np.mean(np.asarray(spectra), axis=0)
    sa, se, cross = mean[0].real, mean[1].real, mean[2]
    coherence = abs(cross)**2/(sa*se)
    # Approximate effective number under stationary zero cross covariance:
    # adjacent overlapping periodograms correlate by the squared normalized
    # window overlap. Independent realizations have no overlap covariance.
    window = get_window('hann', nperseg)
    overlap = np.dot(window[nperseg//2:], window[:nperseg//2])/np.dot(window, window)
    total = sum(segments)
    effective = total**2/(total + 2.*sum(v-1 for v in segments)*overlap**2)
    null_median = 1.-.5**(1./(effective-1.))
    null_p95 = 1.-.05**(1./(effective-1.))
    bands = []
    for low, high in ((2e-5,.001),(.0002,.01),(.0006,.03),(.0001,.03)):
        selected = (f>=low)&(f<=high)
        values = coherence[selected]
        bands.append({'band_hz':[low,high], 'frequency_bins':int(selected.sum()),
            'coherence_median':float(np.median(values)), 'coherence_p95':float(np.quantile(values,.95)),
            'coherence_maximum':float(values.max()),
            'fraction_above_approximate_zero_covariance_p95':float(np.mean(values>null_p95))})
    record = {'created_utc':datetime.now(timezone.utc).isoformat(),
        'noise_model':'secondary_plus_laser', 'noise_components':['laser','test-mass','oms'],
        'TDI_generation':2, 'measurement_order':45, 'delay_order':5,
        'geometry_model':'retarded', 'orbit':'local_analytic_esa_trailing_BME_J2000',
        'ideal_clocks_and_ranging':True, 'unit':'fractional_frequency_squared_per_hz',
        'source_archive':str(source.relative_to(ROOT)), 'seeds':seeds,
        'dt_s':dt, 'nperseg':nperseg, 'noverlap':nperseg//2, 'window':'periodic Hann',
        'detrend':'constant', 'frequency_resolution_hz':float(f[1]-f[0]),
        'Welch_segments_per_realization':segments, 'total_overlapping_segments':total,
        'approximate_effective_averages_under_stationary_null':float(effective),
        'approximate_stationary_zero_covariance_coherence_median':float(null_median),
        'approximate_stationary_zero_covariance_coherence_p95':float(null_p95),
        'bands':bands, 'input_sha256':hashes,
        'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
        'CSD_convention':'S_AE=E[tilde_A*conj(tilde_E)]',
        'limitations':['Finite noisy covariance; three independent seeds, overlapping Hann windows',
            'Null scale assumes stationary Gaussian uncorrelated channels; low-frequency colored leakage can violate this approximation',
            'Shorter segments reduce spectral resolution; not a replacement production PSD',
            'Full/diagonal sensitivity must use identical diagonal PSDs and clean signals',
            'No exact population orthogonality or production convergence certified']}
    np.savez_compressed(output/'covariance.npz', frequency_hz=f, PSD_A=sa, PSD_E=se, CSD_AE=cross)
    (output/'covariance.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    shutil.copy2(Path(__file__), output/'referee_ae_covariance.py')
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/referee/laser_on_off_20261007')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nperseg', type=int, default=16384)
    args = parser.parse_args()
    if args.nperseg < 4 or args.nperseg%2:
        raise ValueError('Even positive segment length required')
    build(args.output, args.source, args.nperseg)
