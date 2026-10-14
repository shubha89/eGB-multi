"""Tasks 2.3/2.4: labelled TDI2 PSD and physical circular orientation audit.

python benchmarks/referee_task_2_objective_orientation.py --noise-model instrument_defaults
The PSD is a validation realization, not a convergence-certified production PSD.
All eGB imports and output paths are restricted to this checkout.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/referee/.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'outputs/referee/.cache'))
sys.path.insert(0, str(ROOT/'src'))

import h5py
import numpy as np
from scipy.signal import csd

from egb_multi.comparison import AESeries, AEPSD, AEInnerProduct, TDIConvention
from egb_multi.circular_fit import CircularAmplitudeBasis, maximize_circular_orientation
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits, link_delays_from_lisaorbits
from egb_multi.noise import simulate_lisainstrument_tdi_noise, estimate_channel_psds, xyz_from_pytdi_data, aet_from_xyz
from egb_multi.tdi import LINK_LABELS, LINKS, pytdi_data_from_links


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def assert_local():
    for name, module in tuple(sys.modules.items()):
        if name == 'egb_multi' or name.startswith('egb_multi.'):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(ROOT/'src/egb_multi'):
                raise RuntimeError(f'Nonlocal eGB import: {name}: {path}')


def make_common_orbit_file(path, duration, *, model='instantaneous'):
    """Write a native retarded orbit file or a historical geometric-arm model.

    The supplied orbit model's native retarded LTT is deliberately not mixed
    with the local response's instantaneous geometric-arm approximation.
    model='retarded' leaves the native LTT and derivative fields intact.
    This is a generated analytic model input rather than a mission ephemeris.
    """
    orbit = default_lisaorbits('esa-trailing')
    orbit_dt, start = 600., -1800.
    size = int(np.ceil((duration+3600.)/orbit_dt))+1
    orbit.write(str(path), t0=start, dt=orbit_dt, size=size)
    if model == 'retarded':
        return orbit
    if model != 'instantaneous':
        raise ValueError('Unknown link geometry model')
    times = start + np.arange(size)*orbit_dt
    state = state_from_lisaorbits(orbit, times)
    # Orbit-file columns use receiver-emitter order (12,23,31,13,32,21).
    by_label = {label: state.arm_length(sender, receiver) for label, (sender, receiver) in zip(LINK_LABELS, LINKS, strict=True)}
    file_labels = ('12', '23', '31', '13', '32', '21')
    geometric = np.column_stack([by_label[label] for label in file_labels])
    with h5py.File(path, 'r+') as stream:
        stream['tcb/ltt'][...] = geometric
        stream['tcb/d_ltt'][...] = np.gradient(geometric, orbit_dt, axis=0, edge_order=2)
        stream.attrs['referee_delay_model'] = 'instantaneous_geometric_arms_matching_egb_multi'
    return orbit


def instrument_kwargs(noise_model, orbit_path):
    kwargs = {'orbits': str(orbit_path), 'orbit_dataset': 'tcb/ltt', 't0': 0.}
    return kwargs


def enabled_noise_contributions(noise_model):
    # Use native component selection so deterministic clock imperfections and
    # ranging biases are also disabled in the explicitly idealized references.
    return {'instrument_defaults': None,
            'secondary_only': ('test-mass', 'oms'),
            'secondary_plus_laser': ('laser', 'test-mass', 'oms')}[noise_model]


def deterministic_delays(orbit, times, model):
    if model == 'retarded':
        return link_delays_from_lisaorbits(orbit, times)
    if model != 'instantaneous':
        raise ValueError('Unknown link geometry model')
    state = state_from_lisaorbits(orbit, times, frame='ecliptic')
    return {label:state.arm_length(sender, receiver) for label, (sender, receiver) in zip(LINK_LABELS, LINKS, strict=True)}


def make_clean_response(orbit, times, convention, halo, *, geometry_model='retarded'):
    """GW-only response: inputs contain no simulated instrument measurements.

    TCB times, retarded emitter geometry and TDI delays are deterministic.
    One common endpoint crop is applied to every injection and template.
    """
    state = state_from_lisaorbits(orbit, times, frame='ecliptic')
    geometry = precompute_jax_link_geometry(state, orbits=orbit if geometry_model=='retarded' else None, frame='ecliptic')
    delays = deterministic_delays(orbit, times, geometry_model)
    if halo < 1 or 2*halo >= times.size-4:
        raise ValueError('Endpoint crop leaves too few clean waveform samples')
    retained_t = state.t[halo:-halo]

    def response(source, evolution='fixed'):
        links = eccentric_links_jax(source, geometry, physics_mode='1pn', evolution_mode=evolution, batch_size=1)
        data = pytdi_data_from_links(state, links, link_delays=delays)
        xyz = xyz_from_pytdi_data(data, generation=convention.generation,
            order=convention.measurement_order, delay_order=convention.delay_order)
        aet = aet_from_xyz(xyz)
        real = {}
        for ch in ('A', 'E'):
            value = np.asarray(aet[ch])[halo:-halo]
            if np.iscomplexobj(value) and np.any(value.imag != 0.):
                raise ValueError('Real source generated nonzero imaginary TDI')
            real[ch] = value.real
        return AESeries(retained_t, real, convention)

    return response, delays


def build_psd(out, args, convention):
    orbit_path = out/'matching_geometric_orbits.h5'
    orbit = make_common_orbit_file(orbit_path, args.samples*args.dt,model=args.geometry_model)
    kwargs = instrument_kwargs(args.noise_model, orbit_path)
    if args.disable_laser_diagnostic:
        kwargs['laser_asds'] = 0.
    print(f'Generating {args.noise_model} noise: N={args.samples}, dt={args.dt}, TDI2 with SCI/TMI/REF', flush=True)
    started = time.perf_counter()
    noise = simulate_lisainstrument_tdi_noise(size=args.samples, dt=args.dt, seed=args.seed,
        generation=2, order=args.order, delay_order=args.delay_order,
        instrument_kwargs=kwargs, include_local_metrology=True,
        enabled_noises=enabled_noise_contributions(args.noise_model))
    # TDI2 contains nested delays; omit endpoint samples lacking complete
    # interpolation support. Include a conservative additional kernel margin.
    times = np.arange(args.samples, dtype=float)*args.dt
    true_delays = deterministic_delays(orbit, times, args.geometry_model)
    max_delay = max(float(np.max(v)) for v in true_delays.values())
    halo = int(np.ceil(8.*max_delay/args.dt))+4*(args.order+1)
    if not np.array_equal(noise.t, times):
        raise ValueError('Noise PSD timestamps differ from the predetermined TCB grid')
    selected = slice(halo, -halo)
    channels_hz = {ch: value[selected] for ch, value in noise.aet.items()}
    freq, values = estimate_channel_psds(channels_hz, noise.dt, nperseg=args.nperseg)
    _, cross = csd(channels_hz['A'], channels_hz['E'], fs=1./noise.dt,
        nperseg=args.nperseg, noverlap=args.nperseg//2, detrend='constant', window='hann', scaling='density')
    # scipy.csd uses conj(A)*E; the noise covariance S_AE uses A*conj(E).
    cross=cross.conjugate()
    provenance = {
        'sidedness':'one_sided', 'scaling':'density', 'noise_model':args.noise_model,
        'CSD_convention':'S_AE=E[tilde_A*conj(tilde_E)]',
        'seed':args.seed, 'samples_before_endpoint_crop':args.samples, 'dt_s':noise.dt,
        'nperseg':args.nperseg, 'noverlap':args.nperseg//2, 'window':'hann', 'detrend':'constant',
        'endpoint_halo_samples_each_side':halo,
        'endpoint_halo_rule':'ceil(8*maximum_link_delay/dt)+4*(measurement_order+1)',
        'local_metrology':'SCI/TMI/REF carriers; sideband clock-noise calibration not applied',
        'noise_frequency_unit':'Hz', 'central_frequency_hz':noise.central_frequency_hz,
        'orbit_file':str(orbit_path.relative_to(ROOT)), 'orbit_sample_dt_s':600.,
        'instrument_kwargs':kwargs, 'geometry_model':args.geometry_model,
        'enabled_noise_contributions':enabled_noise_contributions(args.noise_model),
        'noise_reference_assumptions': ('Raw uncalibrated instrument defaults' if args.noise_model=='instrument_defaults'
            else 'Ideal clocks and ranging; native test-mass/OMS levels; laser enabled only for secondary_plus_laser'),
        'sky_frame':'barycentric_mean_ecliptic_J2000',
        'instrument_interpolation':['lagrange',45],
        'instrument_physics_upsampling':1,'instrument_antialias_filter':None,
        'time_model':'Instrument tcb/ltt mode identifies proper time with TCB; all-default clocks add SCET errors',
        'laser_disabled_for_diagnostic':args.disable_laser_diagnostic,
        'unmodified_defaults': 'Remaining Instrument defaults recorded in instrument_defaults.json',
        'simulation_wall_s':time.perf_counter()-started,
        'signal_time_and_delay_model':'Predetermined TCB grid and deterministic orbit delays; noise seed does not enter clean responses',
        'noise_TDI_delay_model':'Recorded instrument MPRS; ideal-reference agreement checked separately',
        'limitations':['Single finite Welch realization', 'Stationary diagonal A/E weighting approximation',
                       'No Galactic confusion foreground', 'Interpolation/cadence/PSD realization convergence pending Task 2.8',
                       'All-default noise is raw carrier TDI, not a clock-calibrated mission sensitivity']}
    hz = AEPSD(freq, values, replace(convention, unit='frequency_hz'), provenance)
    fractional = hz.as_fractional_frequency(noise.central_frequency_hz)
    with (out/'tdi2_psd.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frequency_hz', 'PSD_A_fractional_per_hz', 'PSD_E_fractional_per_hz', 'PSD_A_Hz2_per_hz', 'PSD_E_Hz2_per_hz', 'CSD_AE_real_Hz2_per_hz', 'CSD_AE_imag_Hz2_per_hz'])
        writer.writerows(zip(freq, fractional.channels['A'], fractional.channels['E'], values['A'], values['E'], cross.real, cross.imag, strict=True))
    write_json(out/'psd_metadata.json', dict(fractional.provenance, convention=asdict(convention)))
    from lisainstrument import Instrument
    import inspect
    defaults = {name: parameter.default for name, parameter in inspect.signature(Instrument).parameters.items() if parameter.default is not inspect.Parameter.empty}
    write_json(out/'instrument_defaults.json', defaults)
    np.savez_compressed(out/'noise_and_delays.npz', t=noise.t, **{f'noise_{ch}_Hz':v for ch, v in noise.aet.items()}, **{f'delay_{label}_s':v for label, v in noise.delays.items()})
    return orbit, noise, fractional, halo


def run_validation(out, args):
    convention = TDIConvention(2, args.order, args.delay_order, 'local_analytic_esa_trailing_BME_J2000',
        f'deterministic_TCB_{args.geometry_model}_orbit_delays_ideal_timing_ranging')
    orbit, noise, psd, halo = build_psd(out, args, convention)
    if args.psd_only:
        result = {'status':'PSD_generated_for_diagnostic_only', 'noise_model':args.noise_model,
                  'laser_disabled_for_diagnostic':args.disable_laser_diagnostic,
                  'orientation_validation_run':False, 'failures':[]}
        write_json(out/'validation_summary.json', result)
        return result
    times = np.arange(args.samples, dtype=float)*args.dt
    response, delays = make_clean_response(orbit, times, convention, halo, geometry_model=args.geometry_model)
    retained_t = times[halo:-halo]
    delay_errors = {label:float(np.max(abs(noise.delays[label]-value))) for label,value in delays.items()}
    agreement_tolerance_s = 1e-10  # 3 cm: allows orbit-file interpolation roundoff, not physical ranging noise.
    if args.noise_model != 'instrument_defaults' and max(delay_errors.values()) > agreement_tolerance_s:
        raise ValueError('Ideal-reference MPRS disagree with deterministic orbit delays')
    write_json(out/'clean_response_contract.json', {
        'noise_seed_enters_clean_response':False, 'additive_noise_in_injections_or_templates':False,
        'timestamps':'t0=0 TCB seconds + arange(N)*dt', 'delays':convention.delays,
        'sky_frame':'barycentric_mean_ecliptic_J2000', 'physics_mode':'1pn',
        'noise_MPRS_minus_deterministic_delay_max_abs_s_by_link':delay_errors,
        'ideal_reference_agreement_tolerance_s':agreement_tolerance_s,
        'agreement_tolerance_reason':'Conservative 3 cm allowance for interpolating the 600 s orbit file; not a ranging-error model',
        'endpoint_halo_samples_each_side':halo,
        'unit':'fractional_frequency', 'tdi_convention':asdict(convention)})
    np.savez_compressed(out/'clean_times_and_delays.npz', t=times, **{f'delay_{label}_s':v for label,v in delays.items()})
    covariance_psds = None
    if args.covariance_diagnostic:
        if args.noise_model != 'secondary_plus_laser' or args.geometry_model != 'retarded' or args.dt != 5.:
            raise ValueError('This diagnostic covariance is tied to the adopted retarded reference at 5 s cadence')
        from referee_ae_covariance import load_diagnostic_psds
        covariance_psds = load_diagnostic_psds(args.covariance_diagnostic, convention)

    cases = [
        ('circular_low', .0001, 0., 1.9, .617, 1.13, 0., 'fixed'),
        ('circular_face_on', .001, 0., 0., 1.213, .83, 0., 'fixed'),
        ('circular_edge_on_negative_chirp', .003, 0., np.pi/2, .613, 1.12, -2.604193438218414e-15, 'fixed'),
        ('circular_positive_chirp', .003, 0., 1.217, .923, .67, 2.604193438218414e-15, 'fixed'),
        ('eccentric_modest', .001, .1, 1.2, .713, .83, 0., 'fixed'),
        ('eccentric_pm', .003, .3, .8, .213, .4, 0., 'peters_mathews'),
    ]
    results, failures = [], []
    for name, f0, e, inclination, phase, psi, chirp, evolution in cases:
        print(f'Validating {name}: f={f0:g}, e={e:g}, evolution={evolution}', flush=True)
        injection = EccentricBinaryParams(mean_motion=np.pi*f0, eccentricity=e, inclination=inclination,
            phi0=phase, psi=psi, fdot=chirp, beta=.2, lambda_=-1., m1_solar=.8, m2_solar=.6)
        template = replace(injection, eccentricity=0., inclination=.6, psi=.2, phi0=.1)
        signal = response(injection, evolution)
        # Deliberately change common amplitude for circular recovery checks.
        target_scale = 2.3 if e == 0 else 1.
        signal = AESeries(signal.t, {ch: target_scale*v for ch,v in signal.channels.items()}, convention)
        band = (.2*f0, 10.*f0)
        metric = AEInnerProduct(signal, psd, band)
        baseline = metric.compare(response(template))
        basis = CircularAmplitudeBasis.generate(template, response)
        maximum = maximize_circular_orientation(metric, basis)
        recovered = replace(template, inclination=maximum['inclination_rad'], psi=maximum['polarization_rad'], phi0=maximum['orbital_phase_rad'])
        explicit = response(recovered)
        reconstructed = basis.reconstruct(inclination=recovered.inclination, psi=recovered.psi, phi0=recovered.phi0)
        # Residual is evaluated directly, avoiding subtracting near-unit scores.
        prepared_explicit, prepared_basis = metric.prepare(explicit), metric.prepare(reconstructed)
        delta = prepared_explicit-prepared_basis
        basis_error = float(np.sqrt(metric.inner(delta, delta)/metric.norm(prepared_explicit)))
        explicit_metrics = metric.compare(explicit)
        # Independent explicit physical variation near the solution checks that
        # the fast Gram objective is stationary in the original waveform path.
        neighbor_scores = []
        if name == 'circular_positive_chirp':
            for parameter, step in (('inclination', 1e-4), ('psi', 1e-4), ('phi0', 1e-4)):
                for sign in (-1., 1.):
                    neighbor = replace(recovered, **{parameter:getattr(recovered, parameter)+sign*step})
                    neighbor_scores.append(metric.compare(response(neighbor))['overlap'])
        entry = {'case':name, 'injection':asdict(injection), 'injection_amplitude_scale':target_scale,
            'observation_duration_s':retained_t.size*args.dt, 'samples':retained_t.size, 'band_hz':band,
            'baseline':baseline, 'physical_orientation_maximum':maximum,
            'explicit_recovered_metrics':explicit_metrics, 'basis_weighted_relative_error':basis_error,
            'explicit_minus_basis_overlap':explicit_metrics['overlap']-maximum['overlap'],
            'nearby_explicit_overlap_values':neighbor_scores}
        if covariance_psds:
            diagnostics = {}
            for weighting, covariance_psd in covariance_psds.items():
                diagnostic_metric = AEInnerProduct(signal, covariance_psd, band)
                diagnostics[weighting] = maximize_circular_orientation(diagnostic_metric, basis)
            diagonal, full = diagnostics['diagonal'], diagnostics['full_covariance']
            entry['covariance_sensitivity'] = {
                'quantity':'Finite-ensemble covariance sensitivity; not a population-covariance or production certification',
                'diagonal':diagonal, 'full_covariance':full,
                'relative_reference_SNR_change':float(np.sqrt(full['reference_norm']/diagonal['reference_norm'])-1.),
                'extrinsic_maximum_overlap_change':full['overlap']-diagonal['overlap']}
            if e == 0 and (full['overlap'] < 1.-1e-9 or full['fitted_relative_residual'] > 5e-5):
                failures.append(f'{name}: full-covariance circular orientation recovery failed')
        results.append(entry)
        if basis_error > 1e-8 or abs(entry['explicit_minus_basis_overlap']) > 1e-9:
            failures.append(f'{name}: physical basis disagrees with explicit eGB/TDI response')
        if maximum['overlap'] < baseline['overlap']-1e-10:
            failures.append(f'{name}: orientation maximum lost the original physical candidate')
        if e == 0 and (explicit_metrics['overlap'] < 1.-1e-9 or explicit_metrics['fitted_relative_residual'] > 5e-5):
            failures.append(f'{name}: circular orientation/amplitude recovery failed')
        if neighbor_scores and max(neighbor_scores) > explicit_metrics['overlap']+1e-9:
            failures.append(f'{name}: explicit physical variation improves the claimed maximum')
        write_json(out/'orientation_validation.json', results)
        print(f"  overlap {baseline['overlap']:.9g} -> {maximum['overlap']:.12g}; explicit basis error={basis_error:.3g}", flush=True)
    result = {'status':'passed_validation_checks' if not failures else 'discrepancy_requires_investigation',
        'failures':failures, 'noise_model':args.noise_model, 'cases':len(results),
        'max_basis_weighted_relative_error':max(r['basis_weighted_relative_error'] for r in results),
        'max_explicit_minus_basis_overlap_absolute_difference':max(abs(r['explicit_minus_basis_overlap']) for r in results),
        'maximum_circular_fitted_relative_residual':max(r['explicit_recovered_metrics']['fitted_relative_residual'] for r in results if r['injection']['eccentricity']==0),
        'limitation':'Validation at fixed frequency/chirp/sky/masses with one finite PSD realization; full fitting-factor search and production convergence pending.'}
    write_json(out/'validation_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noise-model', choices=('instrument_defaults','secondary_only','secondary_plus_laser'), required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--samples', type=int, default=262144)
    parser.add_argument('--dt', type=float, default=5.)
    parser.add_argument('--nperseg', type=int, default=65536)
    parser.add_argument('--order', type=int, default=45)
    parser.add_argument('--geometry-model',choices=('retarded','instantaneous'),default='retarded')
    parser.add_argument('--delay-order', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--psd-only', action='store_true', help='Generate and diagnose noise without waveform fitting.')
    parser.add_argument('--disable-laser-diagnostic', action='store_true', help='One-factor noise diagnostic; not a replacement reference model.')
    parser.add_argument('--covariance-diagnostic', type=Path, help='Finite-ensemble covariance NPZ from referee_ae_covariance.py; diagnostic only.')
    args = parser.parse_args()
    assert_local()
    if args.samples < 4*args.nperseg or args.nperseg < 4 or args.dt <= 0:
        raise ValueError('At least four Welch segments and positive cadence required')
    out = (args.output or ROOT/f'outputs/referee/task_2_3_2_4_20261007_{args.noise_model}').resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('All outputs must remain inside this checkout')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    files = sorted((ROOT/'src/egb_multi').glob('*.py'))+[Path(__file__).resolve(), ROOT/'pyproject.toml']
    files += [ROOT/'tests'/name for name in ('test_comparison.py','test_circular_fit.py','test_noise.py','test_tdi.py','test_external.py','test_referee_clean_response.py')]
    files += [ROOT/'benchmarks/referee_ae_covariance.py']
    if args.covariance_diagnostic:
        dependency = args.covariance_diagnostic.resolve()
        if not dependency.is_relative_to(ROOT):
            raise ValueError('Covariance input must remain inside this checkout')
        files += [dependency, dependency.with_suffix('.json')]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = out/'snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/'manifest.json', {'created_utc':datetime.now(timezone.utc).isoformat(), 'project_root':str(ROOT),
        'python_executable':sys.executable, 'configuration':dict(vars(args), output=str(out), covariance_diagnostic=str(args.covariance_diagnostic) if args.covariance_diagnostic else None),
        'versions':{name:version(name) for name in ('numpy','scipy','jax','jaxlib','pytdi','lisaorbits','lisainstrument')}, 'sha256':hashes})
    result = run_validation(out, args)
    from referee_psd_diagnostics import diagnose
    diagnose(out)
    assert_local()
    print(json.dumps(dict(result, wall_s=time.perf_counter()-started), indent=2), flush=True)
    if result['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
