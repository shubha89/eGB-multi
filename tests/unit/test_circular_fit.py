import numpy as np
import pytest

from egb_multi.comparison import AESeries, AEPSD, AEInnerProduct, TDIConvention
from egb_multi.circular_fit import (CircularAmplitudeBasis, circular_coefficients,
    circular_coefficient_derivatives, orientation_seed_from_oscillatory_coefficients,
    maximize_circular_orientation, profile_circular_orientation)
from egb_multi.eccentric import EccentricBinaryParams


def test_polarization_phase_coefficients_match_rotated_tensor_expression():
    # Independent direct tensor projection on arbitrary antenna/time functions.
    rng = np.random.default_rng(90210)
    u, psi, phase = -.37, .827, .913
    pc, ps, qc, qs, pr, qr = rng.normal(size=(6, 53))
    plus = (1+u*u)*(pc*np.cos(2*phase)-ps*np.sin(2*phase))+(1-u*u)*pr
    cross = 2*u*(ps*np.cos(2*phase)+pc*np.sin(2*phase))
    plus_q = (1+u*u)*(qc*np.cos(2*phase)-qs*np.sin(2*phase))+(1-u*u)*qr
    cross_q = 2*u*(qs*np.cos(2*phase)+qc*np.sin(2*phase))
    expected = np.cos(2*psi)*plus + np.sin(2*psi)*cross - np.sin(2*psi)*plus_q + np.cos(2*psi)*cross_q
    actual = circular_coefficients(u, psi, phase) @ np.array([pc, ps, qc, qs, pr, qr])
    np.testing.assert_allclose(actual, expected, atol=3e-15)


@pytest.mark.parametrize('u', [-1., -.42, 0., .8, 1.])
def test_physical_orientation_recovery_with_nonorthogonal_basis_and_radial_term(u):
    rng = np.random.default_rng(54321)
    n = 128
    t = np.arange(n)*.5
    convention = TDIConvention(2, 3, 3, 'test_orbit', 'test_delays')
    # Radial responses intentionally appreciable: test the constraint, not its
    # small astrophysical limit or an unrestricted six-amplitude regression.
    values = rng.normal(size=(6, 2, n))
    values[1] += .7*values[0]
    responses = tuple(AESeries(t, {'A': v[0], 'E': v[1]}, convention) for v in values)
    source = EccentricBinaryParams(mean_motion=np.pi*.003, eccentricity=0., phi0=.2)
    basis = CircularAmplitudeBasis(source, responses)
    phase, psi, amplitude = 1.213, .839, 2.7
    data = basis.reconstruct(inclination=np.arccos(u), psi=psi, phi0=phase, amplitude_scale=amplitude)
    f = np.fft.rfftfreq(n, .5)
    psd = AEPSD(f, {'A': np.ones(f.size), 'E': np.full(f.size, 3.)}, convention,
                {'sidedness':'one_sided', 'scaling':'density', 'noise_model':'test_noise'})
    metric = AEInnerProduct(data, psd, (f[1]/2, f[-1]-f[1]/2))
    result = maximize_circular_orientation(metric, basis)
    assert result['overlap'] > 1.-2e-12
    assert result['fitted_relative_residual'] < 2e-6
    assert result['positive_amplitude_scale'] == pytest.approx(amplitude, rel=1e-6)
    recovered = basis.reconstruct(inclination=result['inclination_rad'], psi=result['polarization_rad'], phi0=result['orbital_phase_rad'], amplitude_scale=result['positive_amplitude_scale'])
    assert metric.compare(recovered)['raw_relative_residual'] < 2e-6
    fast = profile_circular_orientation(metric,basis)
    assert fast['overlap'] > 1.-2e-12
    assert fast['fitted_relative_residual'] < 2e-6


def test_physical_coefficient_jacobian_matches_independent_finite_differences():
    angles = np.array([-.37,.827,.913])
    actual = circular_coefficient_derivatives(*angles)
    for index in range(3):
        delta = np.eye(3)[index]*1e-6
        expected = (circular_coefficients(*(angles+delta))-circular_coefficients(*(angles-delta)))/2e-6
        np.testing.assert_allclose(actual[:,index],expected,rtol=2e-9,atol=2e-10)


@pytest.mark.parametrize('u',[-1.,-.37,0.,.62,1.])
def test_SVD_orientation_seed_reconstructs_four_physical_coefficients(u):
    original = 2.3*circular_coefficients(u,.827,.913)[:4]
    angles = orientation_seed_from_oscillatory_coefficients(original)
    recovered = circular_coefficients(*angles)[:4]
    amplitude = np.dot(original,recovered)/np.dot(recovered,recovered)
    assert amplitude > 0
    np.testing.assert_allclose(amplitude*recovered,original,rtol=2e-14,atol=2e-14)


@pytest.mark.parametrize('seed',[31,72,109])
def test_fast_profile_matches_global_fit_for_signal_outside_physical_family(seed):
    rng = np.random.default_rng(seed)
    t = np.arange(128)*.5
    convention = TDIConvention(2,3,3,'test_orbit','test_delays')
    values = rng.normal(size=(6,2,t.size))
    source = EccentricBinaryParams(mean_motion=np.pi*.003,eccentricity=0.,phi0=.2)
    basis = CircularAmplitudeBasis(source,tuple(AESeries(t,{'A':v[0],'E':v[1]},convention) for v in values))
    data = AESeries(t,{'A':rng.normal(size=t.size),'E':rng.normal(size=t.size)},convention)
    f = np.fft.rfftfreq(t.size,.5)
    psd = AEPSD(f,{'A':np.ones(f.size),'E':np.full(f.size,3.)},convention,
        {'sidedness':'one_sided','scaling':'density','noise_model':'test_noise'})
    metric = AEInnerProduct(data,psd,(f[1]/2,f[-1]-f[1]/2))
    fast, global_fit = profile_circular_orientation(metric,basis), maximize_circular_orientation(metric,basis)
    assert fast['overlap'] == pytest.approx(global_fit['overlap'],abs=2e-9)


def test_basis_rejects_eccentric_source_before_generation():
    with pytest.raises(ValueError, match='eccentricity=0'):
        CircularAmplitudeBasis.generate(EccentricBinaryParams(mean_motion=.01, eccentricity=.1), lambda source: None)
