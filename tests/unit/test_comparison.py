from dataclasses import replace

import numpy as np
import pytest

from egb_multi.comparison import AESeries, AEPSD, AEInnerProduct, TDIConvention


def fixture_data(n=128, dt=.5):
    convention = TDIConvention(2, 3, 3, "test_orbit", "test_delays")
    t = 100. + np.arange(n)*dt
    phase = 2.*np.pi*np.arange(n)/n
    channels = {"A": np.cos(7*phase)+.4*np.sin(11*phase), "E": .7*np.sin(7*phase)}
    data = AESeries(t, channels, convention)
    f = np.fft.rfftfreq(n, dt)
    psd = AEPSD(f, {"A": np.full(f.size, 2.), "E": np.full(f.size, 3.)}, convention,
                {"sidedness": "one_sided", "scaling": "density", "noise_model": "test_white_noise"})
    return data, psd, (f[1]/2, f[-1]-f[1]/2)


def test_white_noise_inner_matches_time_domain_parseval():
    data, psd, band = fixture_data()
    metric = AEInnerProduct(data, psd, band)
    expected = 2.*data.dt*sum(np.sum(data.channels[ch]**2)/psd.channels[ch][0] for ch in ("A", "E"))
    assert metric.reference_norm == pytest.approx(expected, rel=2e-15)


def test_inner_matches_independent_direct_dft_and_channel_sum():
    data, psd, band = fixture_data(n=32)
    metric = AEInnerProduct(data, psd, band)
    expected = 0.
    for ch in ("A", "E"):
        for k, f in enumerate(np.fft.rfftfreq(data.t.size, data.dt)):
            if band[0] <= f <= band[1]:
                coeff = data.dt*sum(data.channels[ch][j]*np.exp(-2j*np.pi*k*j/data.t.size) for j in range(data.t.size))
                expected += 4.*metric.df*abs(coeff)**2/psd.channels[ch][k]
    assert metric.reference_norm == pytest.approx(expected, rel=2e-15)


def test_overlap_sign_and_positive_amplitude_fit_are_distinct():
    data, psd, band = fixture_data()
    metric = AEInnerProduct(data, psd, band)
    twice = AESeries(data.t, {ch: 2*v for ch, v in data.channels.items()}, data.convention)
    fitted = metric.compare(twice)
    assert fitted['overlap'] == pytest.approx(1.)
    assert fitted['positive_amplitude_scale'] == pytest.approx(.5)
    assert fitted['raw_relative_residual'] == pytest.approx(1.)
    assert fitted['fitted_relative_residual'] < 1e-14
    negative = AESeries(data.t, {ch: -v for ch, v in data.channels.items()}, data.convention)
    result = metric.compare(negative)
    assert result['overlap'] == pytest.approx(-1.)
    assert result['positive_amplitude_scale'] == 0.
    assert result['fitted_relative_residual'] == pytest.approx(1.)


def test_frequency_to_fractional_conversion_preserves_absolute_norm():
    data, psd, band = fixture_data()
    optical_frequency = 2.816e14
    hz = replace(data.convention, unit="frequency_hz")
    data_hz = AESeries(data.t, {ch: v*optical_frequency for ch, v in data.channels.items()}, hz)
    psd_hz = AEPSD(psd.frequency_hz, {ch: v*optical_frequency**2 for ch, v in psd.channels.items()}, hz, psd.provenance)
    fractional = psd_hz.as_fractional_frequency(optical_frequency)
    assert AEInnerProduct(data, fractional, band).reference_norm == pytest.approx(AEInnerProduct(data_hz, psd_hz, band).reference_norm, rel=3e-15)


@pytest.mark.parametrize('kind', ['generation', 'timestamps', 'unit', 'zero_norm', 'nonfinite'])
def test_invalid_candidate_fails(kind):
    data, psd, band = fixture_data()
    metric = AEInnerProduct(data, psd, band)
    t, values, convention = data.t, data.channels, data.convention
    if kind == 'generation':
        convention = replace(convention, generation=1)
    elif kind == 'timestamps':
        t = t+.1
    elif kind == 'unit':
        convention = replace(convention, unit='frequency_hz')
    elif kind == 'zero_norm':
        values = {ch: np.zeros_like(v) for ch, v in values.items()}
    else:
        values = {ch: np.full_like(v, np.nan) for ch, v in values.items()}
    with pytest.raises(ValueError):
        metric.compare(AESeries(t, values, convention))


@pytest.mark.parametrize('bad', [0., -1., np.inf, np.nan])
def test_invalid_psd_bin_is_not_silently_masked(bad):
    data, psd, band = fixture_data()
    psd.channels['A'][7] = bad
    with pytest.raises(ValueError, match='PSD'):
        AEInnerProduct(data, psd, band)


def test_band_rejects_nyquist_truncation_and_psd_extrapolation():
    data, psd, band = fixture_data()
    with pytest.raises(ValueError, match='Nyquist'):
        AEInnerProduct(data, psd, (band[0], 1./data.dt))
    limited = AEPSD(psd.frequency_hz[:10], {ch: v[:10] for ch, v in psd.channels.items()}, data.convention, psd.provenance)
    with pytest.raises(ValueError, match='cover'):
        AEInnerProduct(data, limited, band)


def test_signal_and_psd_inputs_are_not_mutated():
    data, psd, band = fixture_data()
    original = {ch: v.copy() for ch, v in psd.channels.items()}
    metric = AEInnerProduct(data, psd, band)
    mask = metric.mask.copy()
    metric.compare(data)
    np.testing.assert_array_equal(metric.mask, mask)
    for ch in ('A', 'E'):
        np.testing.assert_array_equal(psd.channels[ch], original[ch])


def correlated_psd(psd, cross):
    provenance = dict(psd.provenance, CSD_convention='S_AE=E[tilde_A*conj(tilde_E)]')
    return AEPSD(psd.frequency_hz, psd.channels, psd.convention, provenance,
        np.full(psd.frequency_hz.size, cross, dtype=complex))


def test_full_covariance_matches_independent_inverse_quadratic_form():
    # Imaginary cross spectrum exercises the conjugation convention: A's
    # cosine and E's sine have different Fourier phases at the common peak.
    data, psd, band = fixture_data(n=32)
    cross = .4+.7j
    metric = AEInnerProduct(data, correlated_psd(psd, cross), band)
    covariance = np.array([[2., cross], [cross.conjugate(), 3.]])
    expected = 0.
    for k, f in enumerate(np.fft.rfftfreq(data.t.size, data.dt)):
        if band[0] <= f <= band[1]:
            vector = np.array([data.dt*sum(data.channels[ch][j]*np.exp(-2j*np.pi*k*j/data.t.size)
                for j in range(data.t.size)) for ch in ('A','E')])
            expected += 4.*metric.df*np.vdot(vector, np.linalg.solve(covariance, vector)).real
    assert metric.reference_norm == pytest.approx(expected, rel=3e-15)
    wrong = AEInnerProduct(data, correlated_psd(psd, cross.conjugate()), band)
    assert abs(metric.reference_norm-wrong.reference_norm)/expected > .1


def test_zero_cross_covariance_reduces_to_diagonal_objective():
    data, psd, band = fixture_data()
    diagonal = AEInnerProduct(data, psd, band)
    full = AEInnerProduct(data, correlated_psd(psd, 0.), band)
    np.testing.assert_array_equal(full.reference, diagonal.reference)
    assert full.compare(data) == diagonal.compare(data)


def test_complex_cross_psd_unit_conversion_preserves_norm():
    data, psd, band = fixture_data()
    full = correlated_psd(psd, .4+.7j)
    optical_frequency = 2.816e14
    hz = replace(data.convention, unit='frequency_hz')
    data_hz = AESeries(data.t, {ch:v*optical_frequency for ch,v in data.channels.items()}, hz)
    psd_hz = AEPSD(psd.frequency_hz, {ch:v*optical_frequency**2 for ch,v in psd.channels.items()},
        hz, full.provenance, full.cross_ae*optical_frequency**2)
    fractional = psd_hz.as_fractional_frequency(optical_frequency)
    np.testing.assert_allclose(fractional.cross_ae, full.cross_ae, rtol=1e-15)
    assert AEInnerProduct(data, fractional, band).reference_norm == pytest.approx(
        AEInnerProduct(data_hz, psd_hz, band).reference_norm, rel=3e-15)


@pytest.mark.parametrize('cross', [np.sqrt(6.), 3., np.nan, np.inf])
def test_invalid_cross_covariance_is_rejected_without_flooring(cross):
    data, psd, band = fixture_data()
    with pytest.raises(ValueError, match='covariance'):
        AEInnerProduct(data, correlated_psd(psd, cross), band)


def test_cross_covariance_requires_explicit_convention():
    data, psd, _ = fixture_data()
    with pytest.raises(ValueError, match='convention'):
        AEPSD(psd.frequency_hz, psd.channels, data.convention, psd.provenance,
            np.zeros(psd.frequency_hz.size))
