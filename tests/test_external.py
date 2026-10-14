import numpy as np
import pytest

from egb_multi.external import default_lisaorbits, has_package, state_from_lisaorbits, link_delays_from_lisaorbits


@pytest.mark.skipif(not has_package("lisaorbits"), reason="lisaorbits not installed")
def test_state_from_lisaorbits_shape():
    orbits = default_lisaorbits("equal")
    state = state_from_lisaorbits(orbits, np.array([0.0, 10.0]))

    assert state.positions.shape == (2, 3, 3)
    assert np.all(state.arm_length(0, 1) > 0.0)


@pytest.mark.skipif(not has_package("lisaorbits"), reason="lisaorbits not installed")
def test_esa_trailing_orbit_has_flexing():
    orbits = default_lisaorbits("esa-trailing")
    t = np.linspace(0.0, 365.25 * 24.0 * 3600.0, 128)
    ltt = np.asarray(orbits.compute_ltt(t, [12, 23, 31, 13, 32, 21]))

    assert ltt.shape == (128, 6)
    assert (np.max(ltt) - np.min(ltt)) / np.mean(ltt) > 1.0e-3


def test_deterministic_delays_keep_native_receiver_emitter_labels():
    class LabelledOrbits:
        def compute_ltt(self, times, labels):
            return np.broadcast_to(np.asarray(labels)/10., (times.size,len(labels)))
    values = link_delays_from_lisaorbits(LabelledOrbits(), np.arange(3.))
    assert list(values) == ['21','32','13','31','23','12']
    for label, array in values.items():
        np.testing.assert_array_equal(array, np.full(3, int(label)/10.))


@pytest.mark.parametrize('bad', [0., -1., np.nan])
def test_invalid_orbit_delay_fails(bad):
    class BadOrbits:
        def compute_ltt(self, times, labels):
            return np.full((times.size,len(labels)), bad)
    with pytest.raises(ValueError, match='light travel'):
        link_delays_from_lisaorbits(BadOrbits(), np.arange(3.))


@pytest.mark.skipif(not has_package("fastgb") or not has_package("lisaorbits"), reason="FastGB/lisaorbits not installed")
def test_fastgb_convention_circular_xyz_matches_fastgb():
    from egb_multi.external import FastGBBackend
    from egb_multi.fastgb_convention import fastgb_convention_circular_xyz
    from egb_multi.sources import GalacticBinaryParams

    source = GalacticBinaryParams(
        f0=1.0e-4,
        amplitude=1.0e-22,
        beta=0.3,
        lambda_=-0.7,
        psi=0.4,
        inclination=1.0,
        phi0=0.2,
    )
    backend = FastGBBackend(
        delta_t=100.0,
        duration=30.0 * 24.0 * 3600.0,
        n_samples=64,
        orbits=default_lisaorbits("equal"),
    )

    expected = backend.frequency_domain_xyz(source)
    actual = fastgb_convention_circular_xyz(backend, source)

    assert np.array_equal(actual.kmin, expected.kmin)
    assert np.allclose(actual.x, expected.x, rtol=1.0e-12, atol=1.0e-35)
    assert np.allclose(actual.y, expected.y, rtol=1.0e-12, atol=1.0e-35)
    assert np.allclose(actual.z, expected.z, rtol=1.0e-12, atol=1.0e-35)
