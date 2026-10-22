import numpy as np
import pytest

import egb_jax_eccentric.eccentric as eccentric_module
from egb_jax_eccentric import (
    EccentricBinaryParams,
    PetersMathewsSwitchPoint,
    PetersMathewsSwitchRule,
    calibrate_peters_mathews_switch_rule,
    complex_strain_mismatch,
    eccentric_complex_strain,
    eccentric_complex_strain_batch_jax,
    eccentric_evolution_mode_label,
    eccentric_links_jax,
    eccentric_physics_mode_label,
    eccentric_xyz_jax,
    has_package,
    lisa_orbit,
    peters_mathews_source_mismatch,
    precompute_jax_link_geometry,
    select_peters_mathews_evolution_mode,
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
    explicit_full = eccentric_complex_strain(source, t, physics_mode="1pn_periastron")

    assert np.all(np.isfinite(newtonian))
    assert np.all(np.isfinite(no_periastron))
    assert np.all(np.isfinite(full))
    assert np.allclose(explicit_full, full)
    assert np.linalg.norm(full - newtonian) > 0.0
    assert np.linalg.norm(full - no_periastron) > 0.0


def test_mode_labels_keep_physics_and_evolution_axes_separate():
    assert eccentric_evolution_mode_label("fixed") == "fixed"
    assert eccentric_evolution_mode_label("pm") == "peters_mathews"
    assert eccentric_evolution_mode_label("adaptive") == "auto"
    assert eccentric_physics_mode_label("newtonian") == "newtonian"
    assert eccentric_physics_mode_label("1pn_no_periastron") == "1pn_no_periastron"
    assert eccentric_physics_mode_label("1pn_periastron") == "1pn"

    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.2,
        m1_solar=0.6,
        m2_solar=0.4,
        inclination=0.8,
        phi0=0.2,
    )
    t = np.linspace(0.0, 365.25 * 24.0 * 3600.0, 128)

    pm_alias = eccentric_complex_strain(source, t, physics_mode="1pn_periastron", evolution_mode="pm")
    pm_explicit = eccentric_complex_strain(source, t, physics_mode="1pn", evolution_mode="peters_mathews")

    assert np.allclose(pm_alias, pm_explicit)


def test_auto_peters_mathews_switch_uses_requested_tolerance():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 5.0e-3,
        eccentricity=0.2,
        m1_solar=1.4,
        m2_solar=1.4,
        inclination=0.8,
        phi0=0.2,
    )
    t = np.linspace(0.0, 365.25 * 24.0 * 3600.0, 256)

    fixed = eccentric_complex_strain(source, t, evolution_mode="fixed")
    full_pm = eccentric_complex_strain(source, t, evolution_mode="peters_mathews")
    direct_mismatch = peters_mathews_source_mismatch(source, t)

    assert direct_mismatch == pytest.approx(complex_strain_mismatch(full_pm, fixed))
    assert select_peters_mathews_evolution_mode(source, t, pm_mismatch_tolerance=0.99) == "fixed"
    assert select_peters_mathews_evolution_mode(source, t, pm_mismatch_tolerance=0.0) == "peters_mathews"
    assert np.allclose(eccentric_complex_strain(source, t, evolution_mode="auto", pm_mismatch_tolerance=0.99), fixed)
    assert np.allclose(eccentric_complex_strain(source, t, evolution_mode="auto", pm_mismatch_tolerance=0.0), full_pm)


def test_calibrated_peters_mathews_switch_rule_can_drive_auto_mode():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 3.0e-3,
        eccentricity=0.1,
        m1_solar=0.8,
        m2_solar=0.6,
        inclination=0.8,
        phi0=0.2,
    )
    t = np.linspace(0.0, 60.0 * 24.0 * 3600.0, 128)

    rule = calibrate_peters_mathews_switch_rule([source], t, mismatch_tolerance=1.0e-2)
    predicted = rule.predict_mismatch(source, duration_s=float(t[-1] - t[0]))
    direct = peters_mathews_source_mismatch(source, t)

    assert predicted == pytest.approx(direct)
    assert rule.evolution_mode_for(source, float(t[-1] - t[0]), mismatch_tolerance=0.99) == "fixed"
    assert rule.evolution_mode_for(source, float(t[-1] - t[0]), mismatch_tolerance=0.0) == "peters_mathews"
    assert np.allclose(
        eccentric_complex_strain(source, t, evolution_mode="auto", pm_switch_rule=rule, pm_mismatch_tolerance=0.99),
        eccentric_complex_strain(source, t, evolution_mode="fixed"),
    )


def test_peters_mathews_switch_rule_interpolates_between_points():
    source = EccentricBinaryParams(mean_motion=np.pi * 1.5e-3, eccentricity=0.15, m1_solar=0.7, m2_solar=0.5)
    duration_s = 100.0
    rule = PetersMathewsSwitchRule(
        (
            PetersMathewsSwitchPoint(1.0e-3, 0.1, 1.2, source.symmetric_mass_ratio, duration_s, 1.0e-3),
            PetersMathewsSwitchPoint(2.0e-3, 0.2, 1.2, source.symmetric_mass_ratio, duration_s, 1.0e-1),
        ),
        mismatch_tolerance=1.0e-2,
    )

    predicted = rule.predict_mismatch(source, duration_s)

    assert 1.0e-3 < predicted < 1.0e-1


def test_peters_mathews_derivatives_reject_nonphysical_eccentricity_arrays():
    source = EccentricBinaryParams(mean_motion=np.pi * 1.0e-3, eccentricity=0.2, m1_solar=0.6, m2_solar=0.4)
    n = np.array([source.mean_motion, source.mean_motion])

    with pytest.raises(ValueError, match="eccentricity"):
        eccentric_module.peters_mathews_derivatives(n, np.array([-0.1, 0.2]), source.total_mass_kg, source.symmetric_mass_ratio)

    with pytest.raises(ValueError, match="eccentricity"):
        eccentric_module.peters_mathews_derivatives(n, np.array([0.2, 1.0]), source.total_mass_kg, source.symmetric_mass_ratio)

    with pytest.raises(ValueError, match="eccentricity"):
        eccentric_module.peters_mathews_derivatives(n, np.array([0.2, np.nan]), source.total_mass_kg, source.symmetric_mass_ratio)


def test_peters_mathews_private_state_derivative_rejects_nonphysical_states():
    source = EccentricBinaryParams(mean_motion=np.pi * 1.0e-3, eccentricity=0.2, m1_solar=0.6, m2_solar=0.4)

    with pytest.raises(ValueError, match="eccentricity"):
        eccentric_module._peters_mathews_state_derivative(
            np.array([source.mean_motion, -1.0e-6, 0.0]),
            source.total_mass_kg,
            source.symmetric_mass_ratio,
        )

    with pytest.raises(ValueError, match="eccentricity"):
        eccentric_module._peters_mathews_state_derivative(
            np.array([source.mean_motion, 1.0, 0.0]),
            source.total_mass_kg,
            source.symmetric_mass_ratio,
        )

    with pytest.raises(ValueError, match="mean_motion"):
        eccentric_module._peters_mathews_state_derivative(
            np.array([0.0, source.eccentricity, 0.0]),
            source.total_mass_kg,
            source.symmetric_mass_ratio,
        )


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
def test_jax_physics_mode_accepts_explicit_periastron_alias():
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
