"""Independent mathematical checks of the Task 2.1/2.2 audit helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from referee_task_2_audit import assert_local_package, parameter_contract, physical_phase_maximum, rereference_circular
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.constants import C_M_PER_S, G_M3_KG_S2


def test_phase_maximum_recovers_nonorthogonal_unequal_norm_quadratures():
    # An independent Euclidean vector example, including a phase-independent term.
    basis = np.array([[.3, -.1, .2, .05], [1., .4, -.1, .2], [.7, 1.4, .5, -.3]])
    expected_theta = .817
    coeff = np.array([1., np.cos(expected_theta), np.sin(expected_theta)])
    signal = 2.7 * (coeff @ basis)
    maximum, theta = physical_phase_maximum(basis @ basis.T, basis @ signal, signal @ signal)
    assert maximum == pytest.approx(1., abs=2e-13)
    assert np.angle(np.exp(1j * (theta - expected_theta))) == pytest.approx(0., abs=1e-7)


def test_rereference_preserves_circular_phase_polynomial():
    source = EccentricBinaryParams(mean_motion=np.pi * .003, eccentricity=0., phi0=.2, fdot=3e-15, t0=123.)
    changed = rereference_circular(source, 123. + 15_778_800.)
    # Check source phase/frequency at fixed absolute times, not a shifted series.
    times = np.array([-500., 0., 1e6, 3e7])
    def phase(item):
        tau = times - item.t0
        return item.phi0 + item.mean_motion * tau + .5 * np.pi * item.fdot * tau**2
    phase_error = np.angle(np.exp(2j * (phase(source) - phase(changed))))
    assert np.max(np.abs(phase_error)) < 2e-10
    np.testing.assert_allclose(source.mean_motion + np.pi * source.fdot * (times-source.t0), changed.mean_motion + np.pi * changed.fdot * (times-changed.t0), rtol=2e-15)


def test_rereference_rejects_eccentric_source():
    source = EccentricBinaryParams(mean_motion=np.pi * .001, eccentricity=.1)
    with pytest.raises(ValueError, match="only for circular"):
        rereference_circular(source, 100.)


def test_parameter_domain_contains_original_fdot_candidates_and_signed_chirp():
    source = EccentricBinaryParams(mean_motion=np.pi * .003, eccentricity=.3)
    domain = parameter_contract(source, 365.25 * 86400., .0006, .03)
    lo, hi = domain["bounds"]["fdot_hz_s"]
    assert lo < 0 < hi
    assert domain["reference_epoch_s"] == 15_778_800.
    assert domain["bounds"]["orbital_phase_rad"] == [0., np.pi]
    assert domain["eccentricity"] == 0.
    assert domain["template_evolution_mode"] == "fixed"


def test_audit_uses_this_checkout():
    assert_local_package()


@pytest.mark.parametrize("fdot", [0., 2.604193438218414e-15, -2.604193438218414e-15])
def test_jax_circular_1pn_chirp_matches_independent_closed_form(fdot):
    from egb_multi.eccentric_jax import eccentric_complex_strain_batch_jax

    source = EccentricBinaryParams(mean_motion=np.pi * .003, eccentricity=0., m1_solar=.8, m2_solar=.6, inclination=1.2, phi0=.413, t0=123., fdot=fdot)
    times = np.linspace(-500., 365.25 * 86400., 1537)
    tau = times - source.t0
    omega = source.mean_motion + np.pi * fdot * tau
    gm = G_M3_KG_S2 * source.total_mass_kg
    x = (gm * omega / C_M_PER_S**3)**(2./3.)
    q = 1. + (-1. + source.symmetric_mass_ratio/3.) * x
    oscillatory = C_M_PER_S**2 * x * (1./q + q*q)
    constant = C_M_PER_S**2 * x * (1./q - q*q)
    phase = source.phi0 + source.mean_motion*tau + .5*np.pi*fdot*tau*tau
    scale = gm*source.symmetric_mass_ratio/(C_M_PER_S**4*source.distance_m)
    cos_i = np.cos(source.inclination)
    plus = -scale*((1.+cos_i*cos_i)*oscillatory*np.cos(2.*phase) + (1.-cos_i*cos_i)*constant)
    cross = -2.*scale*cos_i*oscillatory*np.sin(2.*phase)
    expected = plus - 1j*cross
    actual = eccentric_complex_strain_batch_jax(source, times)[0]
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 2e-10
