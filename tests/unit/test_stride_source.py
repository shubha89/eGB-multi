"""Source/basis integration checks independent of detector response agreement."""
from dataclasses import replace
import numpy as np
import pytest
import jax
from egb_multi.eccentric import EccentricBinaryParams, eccentric_polarizations
from egb_multi.stride import sampled_polarizations
from egb_multi.fastlisa import polarization_basis_matrix, complex_polarizations


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64():
        yield


@pytest.mark.parametrize("physics", ["newtonian", "1pn_no_periastron", "1pn"])
@pytest.mark.parametrize("eccentricity", [0., .1, .6])
def test_sampled_source_backends_preserve_native_polarizations(physics, eccentricity):
    source = EccentricBinaryParams(
        mean_motion=np.pi*.003, eccentricity=eccentricity,
        m1_solar=1.1, m2_solar=.4, inclination=.9, phi0=.7, psi=.4,
    )
    times = 27612900.+np.arange(512)*1.25+.125
    expected = np.array(eccentric_polarizations(source, times, physics_mode=physics))
    for backend in ("numpy", "jax"):
        sampled = sampled_polarizations(
            source, times, physics_mode=physics, source_backend=backend, device="cpu",
        )
        assert isinstance(sampled.values, jax.Array)
        assert sampled.values.dtype == np.float64
        assert float(sampled.origin_s) == times[0] and float(sampled.dt_s) == 1.25
        actual = np.asarray(sampled.values)
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 3e-12


def test_pm_uses_existing_source_and_jax_rejects_unsupported_evolution():
    source = EccentricBinaryParams(mean_motion=np.pi*.003, eccentricity=.1)
    times = np.arange(512)*2.5
    expected = np.array(eccentric_polarizations(source, times, evolution_mode="peters_mathews"))
    actual = sampled_polarizations(source, times, evolution_mode="peters_mathews")
    np.testing.assert_array_equal(actual.values, expected)
    with pytest.raises(ValueError, match="fixed elements"):
        sampled_polarizations(source, times, evolution_mode="peters_mathews", source_backend="jax")
    for invalid in [times[::-1], times+np.sin(times)*.01, times[:2], times*np.nan]:
        with pytest.raises(ValueError):
            sampled_polarizations(source, invalid)
    with pytest.raises(ValueError, match="fdot=0"):
        sampled_polarizations(replace(source, fdot=1e-16), times)


@pytest.mark.parametrize("sky", [(1.2, .3), (4.9, -1.4), (0., np.pi/2)])
@pytest.mark.parametrize("psi", [0., .4, -.7])
def test_native_basis_preserves_tensor_strain_without_a_fit(sky, psi):
    lam, beta = sky
    source = EccentricBinaryParams(
        mean_motion=np.pi*.003, eccentricity=.1,
        lambda_=lam, beta=beta, psi=psi,
    )
    values = np.random.default_rng(193).normal(size=(2, 101))
    strain = complex_polarizations(source, values)
    converted = np.stack((strain.real, strain.imag))
    u = np.array([np.sin(lam), -np.cos(lam), 0.])
    v = np.array([-np.sin(beta)*np.cos(lam), -np.sin(beta)*np.sin(lam), np.cos(beta)])
    native = np.array([np.outer(u, u)-np.outer(v, v), np.outer(u, v)+np.outer(v, u)])
    plus, cross = source.polarization_basis()
    own = np.array([np.cos(2*psi)*plus-np.sin(2*psi)*cross,
                    np.sin(2*psi)*plus+np.cos(2*psi)*cross])
    np.testing.assert_allclose(
        np.einsum("an,aij->nij", converted, native),
        np.einsum("an,aij->nij", values, own), rtol=3e-15, atol=3e-15,
    )
    matrix = polarization_basis_matrix(source)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(2), atol=2e-15)
