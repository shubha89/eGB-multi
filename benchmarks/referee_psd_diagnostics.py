"""Compare archived candidate PSDs without choosing the production noise model.

python benchmarks/referee_psd_diagnostics.py RUN_DIRECTORY [RUN_DIRECTORY ...]
Linear detrending below is a diagnostic of spectral leakage, not a PSD change.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/referee/.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'outputs/referee/.cache'))


def diagnose(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Diagnostics must remain in this checkout')
    metadata = json.loads((path/'psd_metadata.json').read_text())
    data = np.genfromtxt(path/'tdi2_psd.csv', delimiter=',', names=True)
    frequency = data['frequency_hz']
    psd_a, psd_e = data['PSD_A_Hz2_per_hz'], data['PSD_E_Hz2_per_hz']
    cross = data['CSD_AE_real_Hz2_per_hz']+1j*data['CSD_AE_imag_Hz2_per_hz']
    positive = (frequency >= 2e-5)&(frequency <= .03)
    coherence = abs(cross)**2/(psd_a*psd_e)
    records = []
    for freq in (.0001, .001, .003, .01, .03):
        index = int(np.argmin(abs(frequency-freq)))
        records.append({'requested_frequency_hz':freq, 'PSD_frequency_hz':float(frequency[index]),
            'PSD_A_fractional_per_hz':float(data['PSD_A_fractional_per_hz'][index]),
            'PSD_E_fractional_per_hz':float(data['PSD_E_fractional_per_hz'][index]),
            'magnitude_squared_AE_coherence':float(coherence[index])})
    with np.load(path/'noise_and_delays.npz') as noise:
        halo = metadata['endpoint_halo_samples_each_side']
        values = {ch:noise[f'noise_{ch}_Hz'][halo:-halo] for ch in ('A','E')}
        linear = {}
        for ch in ('A','E'):
            _, lin = welch(values[ch], fs=1./metadata['dt_s'], nperseg=metadata['nperseg'],
                noverlap=metadata['noverlap'], window='hann', detrend='linear', scaling='density')
            original = psd_a if ch=='A' else psd_e
            linear[ch] = {'median_linear_to_constant_detrend_PSD_ratio':float(np.median(lin[positive]/original[positive])),
                'minimum_linear_to_constant_detrend_PSD_ratio':float(np.min(lin[positive]/original[positive]))}
    n = metadata['samples_before_endpoint_crop']-2*halo
    segments = 1+(n-metadata['nperseg'])//(metadata['nperseg']-metadata['noverlap'])
    result = {'noise_model':metadata['noise_model'],
        'laser_disabled_for_diagnostic':metadata.get('laser_disabled_for_diagnostic', False),
        'diagnostic_band_hz':[2e-5,.03],
        'magnitude_squared_AE_coherence_median':float(np.median(coherence[positive])),
        'magnitude_squared_AE_coherence_p95':float(np.quantile(coherence[positive],.95)),
        'welch_overlapping_segments':int(segments),
        'independent_zero_coherence_estimator_scale_approximate':1./segments,
        'sample_frequencies':records, 'linear_detrend_diagnostic':linear,
        'production_PSD_changed':False,
        'interpretation':'Finite-realization coherence is diagnostic, not an estimate of the population covariance. Near-unity broadband coherence requires investigation before diagonal weighting is interpreted as physical network sensitivity.'}
    (path/'psd_diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def plot_comparison(directories, output):
    import matplotlib.pyplot as plt
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Plot must remain in this checkout')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for directory in directories:
        directory = directory.resolve()
        metadata = json.loads((directory/'psd_metadata.json').read_text())
        data = np.genfromtxt(directory/'tdi2_psd.csv', delimiter=',', names=True)
        f = data['frequency_hz']
        selected = (f >= 2e-5)&(f <= .03)
        label = 'Test mass + optical metrology' if metadata['noise_model']=='secondary_only' else 'All default noise amplitudes'
        if metadata.get('laser_disabled_for_diagnostic', False):
            label = 'All defaults, laser disabled'
        line, = axes[0].loglog(f[selected], data['PSD_A_fractional_per_hz'][selected], label=label)
        axes[0].loglog(f[selected], data['PSD_E_fractional_per_hz'][selected], '--', color=line.get_color(), alpha=.6)
        coherence = (data['CSD_AE_real_Hz2_per_hz']**2+data['CSD_AE_imag_Hz2_per_hz']**2)/(data['PSD_A_Hz2_per_hz']*data['PSD_E_Hz2_per_hz'])
        axes[1].semilogx(f[selected], coherence[selected], '.', markersize=1.5, alpha=.2, color=line.get_color(), rasterized=True)
    axes[0].set_xlabel('Frequency [Hz]')
    axes[0].set_ylabel('Fractional-frequency PSD [1/Hz]')
    axes[0].set_title('Solid: A; dashed: E')
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_ylabel('A/E magnitude-squared coherence')
    axes[1].set_title('Raw estimates from six overlapping Welch segments')
    for ax in axes:
        ax.grid(True, which='both', alpha=.2)
    fig.suptitle('Validation noise candidates: TDI2, 10 s cadence')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--plot', type=Path, help='Optional PSD/coherence comparison plot inside this checkout.')
    args = parser.parse_args()
    for directory in args.directories:
        diagnose(directory)
    if args.plot:
        plot_comparison(args.directories, args.plot)
