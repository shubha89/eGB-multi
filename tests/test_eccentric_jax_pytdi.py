import numpy as np
import pytest

from egb_jax_eccentric import (
    EccentricBinaryParams,
    eccentric_complex_strain,
    eccentric_complex_strain_batch_jax,
    eccentric_links_jax,
    eccentric_xyz_jax,
    has_package,
    lisa_orbit,
    precompute_jax_link_geometry,
)


def test_numpy_physics_modes_are_finite_and_distinct():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.3,
        m1_solar=0.6,
        m2_solar=0.4,
        inclination=0.8,
        phi0=0.2,
    )
    t = np.linspace(0.0, 100_000.0, 512)

    newtonian = eccentric_complex_strain(source, t, physics_mode="newtonian")
    no_periastron = eccentric_complex_strain(source, t, physics_mode="1pn_no_periastron")
    full = eccentric_complex_strain(source, t, physics_mode="1pn")

    assert np.all(np.isfinite(newtonian))
    assert np.all(np.isfinite(no_periastron))
    assert np.all(np.isfinite(full))
    assert np.linalg.norm(full - newtonian) > 0.0


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_jax_strain_matches_numpy_switchable_model():
    sources = [
        EccentricBinaryParams(
            mean_motion=np.pi * 1.0e-3,
            eccentricity=0.1,
            m1_solar=0.6,
            m2_solar=0.4,
            inclination=0.8,
            phi0=0.2,
        ),
        EccentricBinaryParams(
            mean_motion=np.pi * 1.4e-3,
            eccentricity=0.3,
            m1_solar=0.5,
            m2_solar=0.7,
            inclination=1.1,
            phi0=0.5,
        ),
    ]
    times = np.linspace(0.0, 20_000.0, 256)

    expected = np.stack([eccentric_complex_strain(source, times) for source in sources], axis=0)
    actual = eccentric_complex_strain_batch_jax(sources, times)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=5.0e-10, atol=1.0e-28)


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_jax_links_have_pytdi_labels_and_shapes():
    state = lisa_orbit(np.linspace(0.0, 20_000.0, 128))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-4,
        eccentricity=0.1,
        m1_solar=0.6,
        m2_solar=0.4,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
        phi0=0.2,
    )

    links = eccentric_links_jax(source, precompute_jax_link_geometry(state), batch_size=1)

    assert set(links) == {"12", "23", "31", "13", "32", "21"}
    for value in links.values():
        assert value.shape == state.t.shape
        assert np.iscomplexobj(value)
        assert np.all(np.isfinite(value))


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
@pytest.mark.skipif(not has_package("pytdi"), reason="pytdi not installed")
def test_eccentric_xyz_jax_returns_pytdi_channels():
    state = lisa_orbit(np.arange(512.0))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-4,
        eccentricity=0.05,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
        phi0=0.2,
    )

    xyz = eccentric_xyz_jax(state, source, batch_size=1, physics_mode="1pn")

    assert set(xyz) == {"X", "Y", "Z"}
    for value in xyz.values():
        assert value.shape == state.t.shape
        assert np.iscomplexobj(value)
        assert np.all(np.isfinite(value))
