"""Summarize archived paired laser tests, verify provenance and draw raw PSD ratios."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'outputs/referee/.cache'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify_archive(folder, wrapped_hash=False):
    manifest = read(folder/'manifest.json')
    for rel, expected in manifest['source_hashes'].items():
        payload = (folder/'snapshot'/rel).read_bytes()
        if wrapped_hash:
            # The original paired run used the array hashing function for files.
            # Preserve and verify that recorded convention instead of relabeling it.
            array = np.frombuffer(payload, dtype=np.uint8)
            digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
            actual = digest.hexdigest()
        else:
            actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Source snapshot mismatch: {folder}/{rel}')
    for result in read(folder/'noise_comparison.json'):
        pair = folder/f"{result['model']}_seed{result['seed']}"
        on = read(pair/'on_metadata.json')['nonlaser_noise_hashes']
        off = read(pair/'off_metadata.json')['nonlaser_noise_hashes']
        assert len(on) == len(off) == 93 and on == off
        assert result['pairing']['timestamps_bit_identical'] and result['pairing']['MPRS_bit_identical']
    return dict(source_files_verified=len(manifest['source_hashes']),
        source_hash_algorithm=('SHA256 of uint8 shape/dtype header followed by file bytes'
            if wrapped_hash else 'SHA256 of file bytes'),
        paired_nonlaser_hashes_verified=True)


def load_psd(folder, result):
    return np.genfromtxt(folder/f"{result['model']}_seed{result['seed']}"/'paired_psd.csv', delimiter=',', names=True)


def maximum_by_channel(result, key, quantile='maximum'):
    return max(result[key][ch][quantile] for ch in ('A', 'E'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--isolation', type=Path, required=True)
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paired, isolation, smoke, out = (getattr(args, name).resolve() for name in ('paired', 'isolation', 'smoke', 'output'))
    if not all(path.is_relative_to(ROOT) for path in (paired, isolation, smoke, out)):
        raise ValueError('Inputs and outputs must stay in this checkout')
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out/'snapshot'/Path(__file__).relative_to(ROOT)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, snapshot)
    old_source = (paired/'snapshot/benchmarks/referee_laser_on_off.py').read_text()
    wrapped_hash = 'hashes[str(rel)]=digest(' in old_source
    verification = {'paired': verify_archive(paired, wrapped_hash), 'isolation': verify_archive(isolation)}
    assert read(paired/'manifest.json')['versions'] == read(isolation/'manifest.json')['versions']
    smoke_metadata = read(smoke/'psd_metadata.json')
    assert smoke_metadata['enabled_noise_contributions'] == ['laser', 'test-mass', 'oms']
    assert smoke_metadata['laser_disabled_for_diagnostic'] is False
    assert read(smoke/'validation_summary.json')['orientation_validation_run'] is False
    defaults = read(isolation/'instrument_defaults.json')
    assert defaults['ranging_biases'] == 0 and defaults['ranging_asds'] == 3e-9

    pairs = read(paired/'noise_comparison.json')
    controls = read(isolation/'noise_comparison.json')
    waveform = read(paired/'waveform_comparison.json')
    rows = pairs+controls
    metrics = []
    for result in rows:
        metrics.append(dict(model=result['model'], seed=result['seed'],
            maximum_increment_PSD_ratio=maximum_by_channel(result, 'PSD_increment_to_off'),
            p95_absolute_fractional_PSD_change=maximum_by_channel(result, 'absolute_fractional_PSD_change', 'p95'),
            maximum_coherence_change=result['coherence']['absolute_change']['maximum']))
    tolerances = read(paired/'manifest.json')['predetermined_acceptance_tolerances']
    controls_by_name = {r['model']:r for r in controls}
    assert maximum_by_channel(controls_by_name['ranging_disabled'], 'PSD_increment_to_off') < tolerances['maximum_increment_PSD_ratio']
    assert maximum_by_channel(controls_by_name['clocks_disabled'], 'PSD_increment_to_off') > tolerances['maximum_increment_PSD_ratio']
    for result in pairs:
        if result['model'] == 'secondary':
            assert maximum_by_channel(result, 'PSD_increment_to_off') < tolerances['maximum_increment_PSD_ratio']
            assert maximum_by_channel(result, 'absolute_fractional_PSD_change', 'p95') < tolerances['p95_absolute_fractional_PSD_change']
            assert result['coherence']['absolute_change']['maximum'] < tolerances['maximum_coherence_change']
    for result in waveform:
        if result['model'] == 'secondary':
            assert abs(result['relative_SNR_change']) < tolerances['relative_SNR_change']
            assert abs(result['extrinsic_maximum_change']) < tolerances['extrinsic_maximum_change']

    # Keep the unmodified, unsmoothed estimates; no PSD floor or clipping is used.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout='constrained')
    colors = {'A':'#0072B2', 'E':'#D55E00'}
    for ax, model, title in zip(axes[0], ('all_defaults', 'secondary'),
            ('Raw all-default background: FAIL', 'Test-mass + OMS, ideal clocks/ranges: PASS'), strict=True):
        for result in (r for r in pairs if r['model'] == model):
            psd = load_psd(paired, result); f = psd['frequency_hz']; band = (f >= 1e-4)&(f <= .03)
            for ch in ('A', 'E'):
                ax.loglog(f[band]*1000, (psd[f'PSD_{ch}_increment']/psd[f'PSD_{ch}_off'])[band],
                    color=colors[ch], alpha=.55, lw=.7, label=f'{ch}, seed {result["seed"]}')
        ax.axhline(tolerances['maximum_increment_PSD_ratio'], color='black', ls='--', lw=1, label='Declared max tolerance')
        ax.set(title=title, ylabel=r'$S_{\mathrm{on-off}}/S_{\mathrm{off}}$')
        ax.legend(fontsize=8, loc='best')

    ax = axes[1, 0]
    styles = [('clocks_disabled', '#CC79A7', 'Clocks disabled; noisy ranges'),
              ('ranging_disabled', '#009E73', 'Ranging disabled; default clocks'),
              ('clocks_and_ranging_disabled', '#333333', 'Both disabled')]
    for model, color, label in styles:
        psd = load_psd(isolation, controls_by_name[model]); f = psd['frequency_hz']; band = (f >= 1e-4)&(f <= .03)
        for ch, ls in (('A', '-'), ('E', ':')):
            ax.loglog(f[band]*1000, (psd[f'PSD_{ch}_increment']/psd[f'PSD_{ch}_off'])[band],
                color=color, ls=ls, lw=.8, label=f'{label}, {ch}')
    ax.axhline(tolerances['maximum_increment_PSD_ratio'], color='black', ls='--', lw=1)
    ax.set(title='Controlled clock/ranging isolation, seed 20261007', ylabel=r'$S_{\mathrm{on-off}}/S_{\mathrm{off}}$')
    ax.legend(fontsize=8, loc='best')

    ax = axes[1, 1]
    for result, folder, color, label in ((pairs[0], paired, '#D55E00', 'Raw defaults'),
            (next(r for r in pairs if r['model']=='secondary'), paired, '#0072B2', 'Test-mass + OMS'),
            (controls_by_name['ranging_disabled'], isolation, '#009E73', 'Defaults, ranging disabled')):
        psd = load_psd(folder, result); f = psd['frequency_hz']; band = (f >= 1e-4)&(f <= .03)
        for ch, ls in (('A', '-'), ('E', ':')):
            difference = np.abs(psd[f'PSD_{ch}_on']/psd[f'PSD_{ch}_off']-1)
            ax.loglog(f[band]*1000, difference[band], color=color, ls=ls, lw=.7, label=f'{label}, {ch}')
    ax.axhline(.01, color='black', ls='--', lw=1, label='1% change scale')
    ax.set(title='Total PSD change, seed 20261007', ylabel=r'$|S_{\mathrm{on}}/S_{\mathrm{off}}-1|$')
    ax.legend(fontsize=8, loc='best')
    for ax in axes.flat:
        ax.set_xlabel('Frequency [mHz]'); ax.set_xlim(.1, 30); ax.grid(True, which='both', alpha=.15)
    fig.suptitle('Paired laser ON/OFF: 15.17 days, dt=5 s, TDI2, interpolation 45/45\n93 nonlaser arrays and MPRS identical within every pair; raw Welch estimates', fontsize=12)
    fig.savefig(out/'laser_on_off_and_ranging.png', dpi=180)
    fig.savefig(out/'laser_on_off_and_ranging.pdf')
    plt.close(fig)

    frequency_details = []
    psd = load_psd(paired, pairs[0]); f = psd['frequency_hz']
    for ch in ('A', 'E'):
        ratio = psd[f'PSD_{ch}_increment']/psd[f'PSD_{ch}_off']
        for low, high in ((1e-4, 3e-4), (3e-4, .001), (.001, .003), (.003, .01), (.01, .03)):
            selected = (f >= low)&(f <= high); indices = np.flatnonzero(selected)
            peak = indices[np.argmax(ratio[selected])]
            frequency_details.append(dict(channel=ch, band_hz=[low, high], median_increment_PSD_ratio=float(np.median(ratio[selected])),
                maximum_increment_PSD_ratio=float(ratio[peak]), maximum_at_hz=float(f[peak])))
    write(out/'summary.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        conclusion='Laser retained in tested ideal-clock/range secondary reference; raw default ranging leaves a material laser residual.',
        input_folders=[str(p.relative_to(ROOT)) for p in (paired, isolation, smoke)],
        provenance_verification=verification, predetermined_acceptance_tolerances=tolerances,
        noise_comparisons=metrics, seed20261007_frequency_details=frequency_details,
        waveform_comparisons=[{k:v for k,v in result.items() if k!='results'} for result in waveform],
        raw_default_test_status=read(paired/'validation_summary.json')['status'],
        summary_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        figures_visually_inspected=False))
    print(json.dumps(dict(provenance_verification=verification,noise_comparisons=metrics), indent=2))


if __name__ == '__main__':
    main()
