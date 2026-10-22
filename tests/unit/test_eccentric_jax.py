import numpy as np
import pytest
import sys
from pathlib import Path

from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain
from egb_multi.external import has_package
from egb_multi.geometry import lisa_orbit


def test_jax_geometry_defaults_to_ssb_positions():
    from egb_multi.eccentric_jax import JAX_LINK_LABELS, precompute_jax_link_geometry

    state = lisa_orbit(np.linspace(0.0, 100.0, 4))
    geometry = precompute_jax_link_geometry(state)

    assert JAX_LINK_LABELS == ("21", "32", "13", "31", "23", "12")
    assert geometry["position_origin"] == "ssb"
    assert np.allclose(geometry["p_sender"][0], state.positions[:, 0, :])
    assert np.allclose(geometry["p_receiver"][0], state.positions[:, 1, :])


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_jax_eccentric_complex_strain_matches_numpy():
    from egb_multi.eccentric_jax import eccentric_complex_strain_batch_jax

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
def test_jax_physics_mode_accepts_explicit_periastron_alias():
    from egb_multi.eccentric_jax import eccentric_complex_strain_batch_jax

    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.2,
        m1_solar=0.6,
        m2_solar=0.4,
        inclination=0.8,
        phi0=0.2,
    )
    times = np.linspace(0.0, 20_000.0, 128)

    shorthand = eccentric_complex_strain_batch_jax(source, times, physics_mode="1pn")
    explicit = eccentric_complex_strain_batch_jax(source, times, physics_mode="1pn_periastron")

    assert np.allclose(explicit, shorthand, rtol=5.0e-10, atol=1.0e-28)


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_jax_eccentric_links_match_numpy_bruteforce():
    from egb_multi.eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))
    from brute_force_links import brute_force_eccentric_links

    state = lisa_orbit(np.linspace(0.0, 20_000.0, 128))
    sources = [
        EccentricBinaryParams(
            mean_motion=np.pi * 1.0e-4,
            eccentricity=0.1,
            m1_solar=0.6,
            m2_solar=0.4,
            beta=0.2,
            lambda_=0.4,
            psi=0.3,
            inclination=0.8,
            phi0=0.2,
        ),
        EccentricBinaryParams(
            mean_motion=np.pi * 1.5e-4,
            eccentricity=0.2,
            m1_solar=0.5,
            m2_solar=0.7,
            beta=-0.3,
            lambda_=0.1,
            psi=0.6,
            inclination=1.1,
            phi0=0.5,
        ),
    ]

    expected = brute_force_eccentric_links(state, sources)
    actual = eccentric_links_jax(sources, precompute_jax_link_geometry(state), batch_size=4)

    assert set(actual) == set(expected)
    for label in expected:
        assert actual[label].shape == expected[label].shape
        assert np.allclose(actual[label], expected[label], rtol=5.0e-9, atol=1.0e-28)


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_jax_eccentric_links_auto_matches_fixed_for_loose_pm_tolerance():
    from egb_multi.eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry, resolve_response_evolution_modes

    state = lisa_orbit(np.linspace(0.0, 10.0 * 86400.0, 96))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 3.0e-3,
        eccentricity=0.2,
        m1_solar=1.4,
        m2_solar=1.4,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
        phi0=0.2,
    )
    geometry = precompute_jax_link_geometry(state)

    assert resolve_response_evolution_modes(source, state.t, evolution_mode="auto", pm_mismatch_tolerance=0.99) == ["fixed"]
    fixed = eccentric_links_jax(source, geometry, batch_size=2, evolution_mode="fixed")
    auto = eccentric_links_jax(source, geometry, batch_size=2, evolution_mode="auto", pm_mismatch_tolerance=0.99)

    for label in fixed:
        assert np.allclose(auto[label], fixed[label], rtol=5.0e-10, atol=1.0e-28)


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_jax_eccentric_links_auto_can_select_peters_mathews_fallback():
    from egb_multi.eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry, resolve_response_evolution_modes

    state = lisa_orbit(np.linspace(0.0, 20.0 * 86400.0, 128))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 5.0e-3,
        eccentricity=0.2,
        m1_solar=1.4,
        m2_solar=1.4,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
        phi0=0.2,
    )
    geometry = precompute_jax_link_geometry(state)

    assert resolve_response_evolution_modes(source, state.t, evolution_mode="auto", pm_mismatch_tolerance=0.0) == [
        "peters_mathews"
    ]
    explicit_pm = eccentric_links_jax(source, geometry, batch_size=2, evolution_mode="peters_mathews")
    auto_pm = eccentric_links_jax(source, geometry, batch_size=2, evolution_mode="auto", pm_mismatch_tolerance=0.0)
    fixed = eccentric_links_jax(source, geometry, batch_size=2, evolution_mode="fixed")

    for label in explicit_pm:
        assert np.all(np.isfinite(explicit_pm[label]))
        assert np.allclose(auto_pm[label], explicit_pm[label], rtol=5.0e-10, atol=1.0e-28)
    assert any(np.linalg.norm(explicit_pm[label] - fixed[label]) > 0.0 for label in explicit_pm)


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_jax_interpolated_links_are_close_to_exact_jax_links():
    from egb_multi.eccentric_jax import eccentric_links_interpolated_jax, eccentric_links_jax, precompute_jax_link_geometry

    state = lisa_orbit(np.linspace(0.0, 20_000.0, 4096))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-4,
        eccentricity=0.05,
        m1_solar=0.6,
        m2_solar=0.4,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
        phi0=0.2,
    )
    geometry = precompute_jax_link_geometry(state)

    exact = eccentric_links_jax(source, geometry, batch_size=2)
    interpolated = eccentric_links_interpolated_jax(source, geometry, batch_size=2)

    for label in exact:
        residual = np.linalg.norm(interpolated[label] - exact[label]) / np.linalg.norm(exact[label])
        assert residual < 2.0e-3


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_jax_interpolated_links_reject_bad_order():
    from egb_multi.eccentric_jax import eccentric_links_interpolated_jax, precompute_jax_link_geometry

    state = lisa_orbit(np.linspace(0.0, 20_000.0, 256))
    source = EccentricBinaryParams(mean_motion=np.pi * 1.0e-4, eccentricity=0.05)

    with pytest.raises(ValueError, match="interpolation_order"):
        eccentric_links_interpolated_jax(
            source,
            precompute_jax_link_geometry(state),
            batch_size=2,
            interpolation_order=2,
        )


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_jax_harmonic_links_match_python_harmonic_links():
    from egb_multi.eccentric import eccentric_harmonic_source_batch
    from egb_multi.eccentric_jax import harmonic_links_jax, precompute_jax_link_geometry
    from egb_multi.tdi import harmonic_beta_link_responses

    state = lisa_orbit(np.linspace(0.0, 20_000.0, 512))
    sources = [
        EccentricBinaryParams(
            mean_motion=np.pi * 1.0e-4,
            eccentricity=0.05,
            m1_solar=0.6,
            m2_solar=0.4,
            beta=0.2,
            lambda_=0.4,
            psi=0.3,
            inclination=0.8,
            phi0=0.2,
        ),
        EccentricBinaryParams(
            mean_motion=np.pi * 1.5e-4,
            eccentricity=0.05,
            m1_solar=0.5,
            m2_solar=0.7,
            beta=-0.3,
            lambda_=0.1,
            psi=0.6,
            inclination=1.1,
            phi0=0.5,
        ),
    ]
    components = eccentric_harmonic_source_batch(sources, order=1, samples=512)

    expected = harmonic_beta_link_responses(state, components, order=1)
    actual = harmonic_links_jax(components, precompute_jax_link_geometry(state), batch_size=8)

    assert set(actual) == set(expected)
    for label in expected:
        assert actual[label].shape == expected[label].shape
        assert np.allclose(actual[label], expected[label], rtol=5.0e-10, atol=1.0e-28)


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_jax_harmonic_component_packing_matches_python():
    from egb_multi.eccentric import eccentric_harmonic_source_batch
    from egb_multi.eccentric_jax import pack_harmonic_components, packed_eccentric_harmonic_components_jax

    sources = [
        EccentricBinaryParams(
            mean_motion=np.pi * 1.0e-4,
            eccentricity=0.03,
            m1_solar=0.6,
            m2_solar=0.4,
            beta=0.2,
            lambda_=0.4,
            psi=0.3,
            inclination=0.8,
            phi0=0.2,
        ),
        EccentricBinaryParams(
            mean_motion=np.pi * 1.5e-4,
            eccentricity=0.03,
            m1_solar=0.5,
            m2_solar=0.7,
            beta=-0.3,
            lambda_=0.1,
            psi=0.6,
            inclination=1.1,
            phi0=0.5,
        ),
    ]

    expected = pack_harmonic_components(
        eccentric_harmonic_source_batch(
            sources,
            order=2,
            samples=512,
            include_negative_frequencies=True,
        )
    )
    actual = packed_eccentric_harmonic_components_jax(
        sources,
        order=2,
        samples=512,
        include_negative_frequencies=True,
    )

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=5.0e-9, atol=1.0e-28)


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
@pytest.mark.parametrize("response_mode", ["triliear_beta", "fastgb_transfer"])
def test_grouped_jax_harmonic_links_match_component_path(response_mode):
    from egb_multi.eccentric_jax import eccentric_harmonic_links_grouped_jax, eccentric_harmonic_links_jax, precompute_jax_link_geometry

    state = lisa_orbit(np.linspace(0.0, 20_000.0, 512))
    sources = [
        EccentricBinaryParams(
            mean_motion=np.pi * 1.0e-4,
            eccentricity=0.03,
            m1_solar=0.6,
            m2_solar=0.4,
            beta=0.2,
            lambda_=0.4,
            psi=0.3,
            inclination=0.8,
            phi0=0.2,
        ),
        EccentricBinaryParams(
            mean_motion=np.pi * 1.5e-4,
            eccentricity=0.05,
            m1_solar=0.5,
            m2_solar=0.7,
            beta=-0.3,
            lambda_=0.1,
            psi=0.6,
            inclination=1.1,
            phi0=0.5,
        ),
    ]
    geometry = precompute_jax_link_geometry(state)

    expected = eccentric_harmonic_links_jax(
        sources,
        geometry,
        eccentric_order=2,
        harmonic_samples=256,
        batch_size=8,
        response_mode=response_mode,
    )
    actual = eccentric_harmonic_links_grouped_jax(
        sources,
        geometry,
        eccentric_order=2,
        harmonic_samples=256,
        batch_size=2,
        response_mode=response_mode,
    )

    assert set(actual) == set(expected)
    for label in expected:
        assert actual[label].shape == expected[label].shape
        assert np.allclose(actual[label], expected[label], rtol=5.0e-9, atol=1.0e-28)


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_grouped_jax_harmonic_links_support_newtonian_fixed_eccentricity():
    from egb_multi.eccentric import eccentric_harmonic_source_batch
    from egb_multi.eccentric_jax import eccentric_harmonic_links_grouped_jax, harmonic_links_jax, precompute_jax_link_geometry

    state = lisa_orbit(np.linspace(0.0, 20_000.0, 512))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-4,
        eccentricity=0.25,
        m1_solar=0.6,
        m2_solar=0.4,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
        phi0=0.2,
    )
    geometry = precompute_jax_link_geometry(state)
    components = eccentric_harmonic_source_batch(
        source,
        order=6,
        samples=512,
        include_negative_frequencies=True,
        physics_mode="newtonian",
    )

    expected = harmonic_links_jax(
        components,
        geometry,
        batch_size=32,
        response_mode="fastgb_transfer",
    )
    actual = eccentric_harmonic_links_grouped_jax(
        source,
        geometry,
        eccentric_order=6,
        harmonic_samples=512,
        batch_size=1,
        response_mode="fastgb_transfer",
        physics_mode="newtonian",
    )

    for label in expected:
        assert np.allclose(actual[label], expected[label], rtol=5.0e-9, atol=1.0e-28)


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_jax_sideband_links_return_finite_link_set():
    from egb_multi.eccentric_jax import eccentric_sideband_links_jax, precompute_jax_link_geometry

    state = lisa_orbit(np.linspace(0.0, 20_000.0, 256))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-4,
        eccentricity=0.2,
        m1_solar=0.6,
        m2_solar=0.4,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
        phi0=0.2,
    )

    links = eccentric_sideband_links_jax(
        source,
        precompute_jax_link_geometry(state),
        q_max=4,
        harmonic_samples=512,
        fit_cycles=3.0,
        batch_size=8,
    )

    assert set(links) == {"12", "23", "31", "13", "32", "21"}
    for value in links.values():
        assert value.shape == state.t.shape
        assert np.iscomplexobj(value)
        assert np.all(np.isfinite(value))


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_grouped_jax_sideband_links_match_component_path():
    from egb_multi.eccentric_jax import eccentric_sideband_links_grouped_jax, eccentric_sideband_links_jax, precompute_jax_link_geometry

    state = lisa_orbit(np.linspace(0.0, 20_000.0, 256))
    sources = [
        EccentricBinaryParams(
            mean_motion=np.pi * 1.0e-4,
            eccentricity=0.15,
            m1_solar=0.6,
            m2_solar=0.4,
            beta=0.2,
            lambda_=0.4,
            psi=0.3,
            inclination=0.8,
            phi0=0.2,
        ),
        EccentricBinaryParams(
            mean_motion=np.pi * 1.3e-4,
            eccentricity=0.2,
            m1_solar=0.5,
            m2_solar=0.7,
            beta=-0.3,
            lambda_=0.1,
            psi=0.6,
            inclination=1.1,
            phi0=0.5,
        ),
    ]
    geometry = precompute_jax_link_geometry(state)

    expected = eccentric_sideband_links_jax(
        sources,
        geometry,
        q_max=4,
        harmonic_samples=512,
        fit_cycles=3.0,
        batch_size=8,
        response_mode="fastgb_transfer",
        include_negative_frequencies=True,
    )
    actual = eccentric_sideband_links_grouped_jax(
        sources,
        geometry,
        q_max=4,
        harmonic_samples=512,
        fit_cycles=3.0,
        batch_size=2,
        response_mode="fastgb_transfer",
        include_negative_frequencies=True,
    )

    assert set(actual) == set(expected)
    for label in expected:
        assert actual[label].shape == expected[label].shape
        assert np.allclose(actual[label], expected[label], rtol=2.0e-7, atol=1.0e-28)


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_jax_fastgb_transfer_approaches_beta_at_low_frequency():
    from egb_multi.eccentric import eccentric_harmonic_source_batch
    from egb_multi.eccentric_jax import harmonic_links_jax, precompute_jax_link_geometry

    state = lisa_orbit(np.linspace(0.0, 20_000.0, 512))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-6,
        eccentricity=0.0,
        m1_solar=0.6,
        m2_solar=0.4,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
        phi0=0.2,
    )
    components = eccentric_harmonic_source_batch(source, order=0, samples=512)
    geometry = precompute_jax_link_geometry(state)

    beta = harmonic_links_jax(components, geometry, batch_size=2, response_mode="triliear_beta")
    fastgb = harmonic_links_jax(components, geometry, batch_size=2, response_mode="fastgb_transfer")

    for label in beta:
        residual = np.linalg.norm(fastgb[label] - beta[label]) / np.linalg.norm(beta[label])
        assert residual < 1.0e-3
