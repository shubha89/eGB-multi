import numpy as np
import pytest

import egb_multi.eccentric as eccentric_module
from egb_multi.constants import C_M_PER_S, G_M3_KG_S2
from egb_multi.eccentric import (
    EccentricBinaryParams,
    PetersMathewsSwitchPoint,
    PetersMathewsSwitchRule,
    calibrate_peters_mathews_switch_rule,
    complex_strain_mismatch,
    eccentric_complex_strain,
    eccentric_dynamics,
    eccentric_evolution_mode_label,
    eccentric_fundamental_angular_frequencies,
    eccentric_harmonic_angular_frequency,
    eccentric_harmonic_components,
    eccentric_harmonic_indices,
    eccentric_harmonic_projection_angular_frequency,
    eccentric_physics_mode_label,
    eccentric_polarizations,
    eccentric_sideband_components,
    eccentric_sideband_specs,
    mikkola_kepler,
    peters_mathews_derivatives,
    peters_mathews_evolution,
    peters_mathews_source_mismatch,
    recommended_eccentric_order,
    select_peters_mathews_evolution_mode,
)
from egb_multi.external import has_package
from egb_multi.geometry import lisa_orbit
from egb_multi.tensors import tracefree_matrix
from egb_multi.tdi import compute_triliear_xyz_eccentric, eccentric_beta_link_responses
from egb_multi.tdi import (
    compute_fixed_eccentric_sideband_xyz_jax,
    compute_fixed_eccentric_triliear_xyz,
    compute_fixed_eccentric_triliear_xyz_jax,
    fixed_eccentric_triliear_links_jax,
    fixed_eccentric_triliear_links,
)


def test_mikkola_kepler_residual_is_small():
    mean_anomaly = np.linspace(-200.0, 200.0, 4096)

    for eccentricity in [0.0, 0.1, 0.7, 0.95]:
        u = mikkola_kepler(mean_anomaly, eccentricity)
        residual = u - eccentricity * np.sin(u) - mean_anomaly
        assert np.max(np.abs(residual)) < 1.0e-13


def test_eccentric_dynamics_and_polarizations_are_finite():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.3,
        m1_solar=0.6,
        m2_solar=0.4,
        inclination=0.8,
        phi0=0.2,
    )
    t = np.linspace(0.0, 20_000.0, 256)

    dynamics = eccentric_dynamics(source, t)
    h_plus, h_cross = eccentric_polarizations(source, t)

    assert dynamics["r"].shape == t.shape
    assert h_plus.shape == t.shape
    assert h_cross.shape == t.shape
    assert np.all(np.isfinite(dynamics["r"]))
    assert np.all(dynamics["r"] > 0.0)
    assert np.all(np.isfinite(h_plus))
    assert np.all(np.isfinite(h_cross))


def test_eccentric_polarization_changes_with_eccentricity():
    t = np.linspace(0.0, 10_000.0, 512)
    circular = EccentricBinaryParams(mean_motion=np.pi * 1.0e-3, eccentricity=0.0, inclination=0.4)
    eccentric = EccentricBinaryParams(mean_motion=np.pi * 1.0e-3, eccentricity=0.5, inclination=0.4)

    h_plus_circular, _ = eccentric_polarizations(circular, t)
    h_plus_eccentric, _ = eccentric_polarizations(eccentric, t)

    relative_difference = np.linalg.norm(h_plus_circular - h_plus_eccentric) / np.linalg.norm(h_plus_circular)
    assert relative_difference > 0.1


def test_eccentric_physics_modes_are_finite_and_distinct():
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


def test_eccentric_mode_labels_keep_physics_and_evolution_axes_separate():
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


def test_1pn_circular_phase_uses_azimuthal_frequency():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 3.0e-3,
        eccentricity=0.0,
        m1_solar=1.4,
        m2_solar=1.4,
        phi0=0.2,
    )
    t = np.linspace(0.0, 200_000.0, 1024)

    dynamics = eccentric_dynamics(source, t, physics_mode="1pn")
    phase = np.unwrap(dynamics["phi"])
    fitted_slope = np.polyfit(t - source.t0, phase - phase[0], deg=1)[0]

    assert np.isclose(fitted_slope, source.mean_motion, rtol=1.0e-10)
    assert np.all(dynamics["radial_mean_motion"] < dynamics["azimuthal_frequency"])


def test_peters_mathews_derivatives_follow_expected_signs():
    source = EccentricBinaryParams(mean_motion=np.pi * 1.0e-3, eccentricity=0.2, m1_solar=0.6, m2_solar=0.4)

    ndot, edot = peters_mathews_derivatives(
        source.mean_motion,
        source.eccentricity,
        source.total_mass_kg,
        source.symmetric_mass_ratio,
    )

    assert ndot > 0.0
    assert edot < 0.0


def test_peters_mathews_derivatives_match_quadrupole_reference_formula():
    source = EccentricBinaryParams(mean_motion=np.pi * 1.0e-3, eccentricity=0.2, m1_solar=0.6, m2_solar=0.4)
    n = np.array([np.pi * 1.0e-3, np.pi * 2.0e-3])
    e = np.array([0.0, 0.4])

    ndot, edot = peters_mathews_derivatives(n, e, source.total_mass_kg, source.symmetric_mass_ratio)

    mass_factor = source.symmetric_mass_ratio * (G_M3_KG_S2 * source.total_mass_kg) ** (5.0 / 3.0) / C_M_PER_S**5
    one_minus_e2 = 1.0 - e * e
    eccentric_frequency_boost = (1.0 + 73.0 * e * e / 24.0 + 37.0 * e**4 / 96.0) / one_minus_e2 ** (7.0 / 2.0)
    eccentric_decay_boost = e * (1.0 + 121.0 * e * e / 304.0) / one_minus_e2 ** (5.0 / 2.0)
    expected_ndot = (96.0 / 5.0) * mass_factor * n ** (11.0 / 3.0) * eccentric_frequency_boost
    expected_edot = -(304.0 / 15.0) * mass_factor * n ** (8.0 / 3.0) * eccentric_decay_boost

    assert np.allclose(ndot, expected_ndot, rtol=1.0e-15, atol=0.0)
    assert np.allclose(edot, expected_edot, rtol=1.0e-15, atol=0.0)
    assert edot[0] == pytest.approx(0.0)


def test_peters_mathews_derivatives_reject_nonphysical_eccentricity_arrays():
    source = EccentricBinaryParams(mean_motion=np.pi * 1.0e-3, eccentricity=0.2, m1_solar=0.6, m2_solar=0.4)
    n = np.array([source.mean_motion, source.mean_motion])

    with pytest.raises(ValueError, match="eccentricity"):
        peters_mathews_derivatives(n, np.array([-0.1, 0.2]), source.total_mass_kg, source.symmetric_mass_ratio)

    with pytest.raises(ValueError, match="eccentricity"):
        peters_mathews_derivatives(n, np.array([0.2, 1.0]), source.total_mass_kg, source.symmetric_mass_ratio)


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


def test_peters_mathews_evolution_changes_frequency_and_eccentricity():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.3,
        m1_solar=0.6,
        m2_solar=0.4,
        inclination=0.8,
        phi0=0.2,
    )
    t = np.linspace(0.0, 365.25 * 24.0 * 3600.0, 128)

    evolution = peters_mathews_evolution(source, t)
    fixed = eccentric_complex_strain(source, t, evolution_mode="fixed")
    evolved = eccentric_complex_strain(source, t, evolution_mode="peters_mathews")

    assert evolution["mean_motion"][-1] > evolution["mean_motion"][0]
    assert evolution["eccentricity"][-1] < evolution["eccentricity"][0]
    assert np.all(np.isfinite(evolved))
    assert np.linalg.norm(evolved - fixed) > 0.0


def test_newtonian_dynamics_match_keplerian_parametric_solution():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.4,
        m1_solar=0.7,
        m2_solar=0.5,
        phi0=0.3,
    )
    t = source.t0 + np.linspace(0.0, 2.0 * np.pi / source.mean_motion, 256)

    dynamics = eccentric_dynamics(source, t, physics_mode="newtonian")

    gm = G_M3_KG_S2 * source.total_mass_kg
    mean_anomaly = source.mean_motion * (t - source.t0)
    u = mikkola_kepler(mean_anomaly, source.eccentricity)
    one_minus_ecosu = 1.0 - source.eccentricity * np.cos(u)
    one_minus_e2 = 1.0 - source.eccentricity**2
    semimajor_axis = (gm / source.mean_motion**2) ** (1.0 / 3.0)
    beta_phi = (1.0 - np.sqrt(one_minus_e2)) / source.eccentricity
    v_minus_u = np.unwrap(2.0 * np.arctan2(beta_phi * np.sin(u), 1.0 - beta_phi * np.cos(u)))
    expected_phi = source.phi0 + u + v_minus_u

    assert np.allclose(dynamics["r"], semimajor_axis * one_minus_ecosu, rtol=1.0e-14)
    assert np.allclose(
        dynamics["rdot"],
        source.eccentricity * (gm * source.mean_motion) ** (1.0 / 3.0) * np.sin(u) / one_minus_ecosu,
        rtol=1.0e-14,
        atol=1.0e-12,
    )
    assert np.allclose(dynamics["phidot"], source.mean_motion * np.sqrt(one_minus_e2) / one_minus_ecosu**2, rtol=1.0e-14)
    assert np.allclose(np.unwrap(dynamics["phi"]), expected_phi, rtol=1.0e-14, atol=1.0e-14)


def test_1pn_no_periastron_dynamics_match_tanay_coefficients():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.4,
        m1_solar=0.7,
        m2_solar=0.5,
        phi0=0.3,
    )
    t = source.t0 + np.linspace(0.0, 2.0 * np.pi / source.mean_motion, 256)

    dynamics = eccentric_dynamics(source, t, physics_mode="1pn_no_periastron")

    gm = G_M3_KG_S2 * source.total_mass_kg
    eta = source.symmetric_mass_ratio
    e = source.eccentricity
    mean_anomaly = source.mean_motion * (t - source.t0)
    u = mikkola_kepler(mean_anomaly, e)
    nu = 1.0 - e * np.cos(u)
    one_minus_e2 = 1.0 - e * e
    sqrt_one_minus_e2 = np.sqrt(one_minus_e2)
    x = (gm * source.mean_motion / C_M_PER_S**3) ** (2.0 / 3.0)

    r_1pn = (
        -24.0
        + 9.0 * eta
        + nu * (18.0 - 7.0 * eta)
        + e * e * (24.0 - 9.0 * eta + nu * (-6.0 + 7.0 * eta))
    ) / (6.0 * nu * one_minus_e2)
    rdot_1pn = (-7.0 * eta + e * e * (-6.0 + 7.0 * eta)) / (6.0 * one_minus_e2)
    phidot_1pn = ((-1.0 + nu + e * e) * (-4.0 + eta)) / (nu * one_minus_e2)
    beta_phi = (1.0 - sqrt_one_minus_e2) / e + x * (
        -4.0
        + eta
        + e * e * (8.0 - 2.0 * eta)
        + (4.0 - eta) * sqrt_one_minus_e2
    ) / (e * sqrt_one_minus_e2)
    v_minus_u = np.unwrap(2.0 * np.arctan2(beta_phi * np.sin(u), 1.0 - beta_phi * np.cos(u)))
    w = v_minus_u + e * np.sin(u) + x * 3.0 * (e * np.sin(u) + v_minus_u) / one_minus_e2

    expected_r = gm * nu * (1.0 + x * r_1pn) / (C_M_PER_S**2 * x)
    expected_rdot = C_M_PER_S * np.sqrt(x) * e * np.sin(u) * (1.0 + x * rdot_1pn) / nu
    expected_phi = source.phi0 + mean_anomaly + w
    expected_phidot = source.mean_motion * sqrt_one_minus_e2 * (1.0 + x * phidot_1pn) / nu**2

    assert np.allclose(dynamics["r"], expected_r, rtol=1.0e-14)
    assert np.allclose(dynamics["rdot"], expected_rdot, rtol=1.0e-14, atol=1.0e-12)
    assert np.allclose(np.unwrap(dynamics["phi"]), expected_phi, rtol=1.0e-14, atol=1.0e-14)
    assert np.allclose(dynamics["phidot"], expected_phidot, rtol=1.0e-14)


def test_circular_newtonian_polarizations_match_quadrupole_limit():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.0,
        m1_solar=0.7,
        m2_solar=0.5,
        inclination=0.6,
        phi0=0.3,
    )
    t = np.linspace(0.0, 10_000.0, 128)

    h_plus, h_cross = eccentric_polarizations(source, t, physics_mode="newtonian")

    gm = G_M3_KG_S2 * source.total_mass_kg
    orbital_velocity_squared = (gm * source.mean_motion) ** (2.0 / 3.0)
    strain_scale = G_M3_KG_S2 * source.total_mass_kg * source.symmetric_mass_ratio / (C_M_PER_S**4 * source.distance_m)
    phase = source.mean_motion * (t - source.t0) + source.phi0
    cos_i = np.cos(source.inclination)
    expected_h_plus = -2.0 * strain_scale * orbital_velocity_squared * (1.0 + cos_i * cos_i) * np.cos(2.0 * phase)
    expected_h_cross = -4.0 * strain_scale * orbital_velocity_squared * cos_i * np.sin(2.0 * phase)

    assert np.allclose(h_plus, expected_h_plus, rtol=1.0e-14, atol=1.0e-35)
    assert np.allclose(h_cross, expected_h_cross, rtol=1.0e-14, atol=1.0e-35)


def test_peters_mathews_partial_evolution_modes_are_finite_and_distinct():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 3.0e-3,
        eccentricity=0.3,
        m1_solar=0.8,
        m2_solar=0.6,
        inclination=0.8,
        phi0=0.2,
    )
    t = np.linspace(0.0, 365.25 * 24.0 * 3600.0, 256)

    fixed = eccentric_complex_strain(source, t, evolution_mode="fixed")
    orbital_only = eccentric_complex_strain(source, t, evolution_mode="peters_mathews_orbital_only")
    eccentricity_only = eccentric_complex_strain(source, t, evolution_mode="peters_mathews_eccentricity_only")
    full = eccentric_complex_strain(source, t, evolution_mode="peters_mathews")

    assert np.all(np.isfinite(orbital_only))
    assert np.all(np.isfinite(eccentricity_only))
    assert np.linalg.norm(orbital_only - fixed) > 0.0
    assert np.linalg.norm(eccentricity_only - fixed) > 0.0
    assert np.linalg.norm(full - orbital_only) > 0.0


def test_eccentric_harmonic_frequencies_follow_package_convention():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 3.0e-3,
        eccentricity=0.2,
        m1_solar=1.4,
        m2_solar=1.0,
    )

    radial, azimuthal = eccentric_fundamental_angular_frequencies(source, physics_mode="1pn")

    assert np.isclose(azimuthal, source.mean_motion)
    assert radial < azimuthal
    assert np.isclose(eccentric_harmonic_angular_frequency(source, 2, physics_mode="1pn"), 2.0 * azimuthal)
    assert np.isclose(eccentric_harmonic_angular_frequency(source, 1, physics_mode="1pn"), 2.0 * azimuthal - radial)
    assert np.isclose(eccentric_harmonic_angular_frequency(source, 3, physics_mode="1pn"), 2.0 * azimuthal + radial)
    assert np.isclose(eccentric_harmonic_projection_angular_frequency(source, physics_mode="1pn"), radial)

    circular = EccentricBinaryParams(**{**source.__dict__, "eccentricity": 0.0})
    circular_radial, circular_azimuthal = eccentric_fundamental_angular_frequencies(circular, physics_mode="1pn")
    assert circular_radial < circular_azimuthal
    assert np.isclose(eccentric_harmonic_projection_angular_frequency(circular, physics_mode="1pn"), circular_azimuthal)

    no_periastron_radial, no_periastron_azimuthal = eccentric_fundamental_angular_frequencies(
        source,
        physics_mode="1pn_no_periastron",
    )
    assert np.isclose(no_periastron_radial, source.mean_motion)
    assert np.isclose(no_periastron_azimuthal, source.mean_motion)
    for harmonic in (1, 2, 3, 4):
        assert np.isclose(
            eccentric_harmonic_angular_frequency(source, harmonic, physics_mode="1pn_no_periastron"),
            harmonic * source.mean_motion,
        )


def test_eccentric_harmonic_components_follow_order():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.1,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
    )

    assert eccentric_harmonic_indices(0) == (2,)
    assert eccentric_harmonic_indices(1) == (1, 2, 3)
    assert eccentric_harmonic_indices(2) == (1, 2, 3, 4)
    assert eccentric_harmonic_indices(5) == (1, 2, 3, 4, 5, 6, 7)
    assert recommended_eccentric_order(0.0) == 0
    assert recommended_eccentric_order(0.01) == 1
    assert recommended_eccentric_order(0.05) == 2
    assert recommended_eccentric_order(0.1) is None

    components = eccentric_harmonic_components(source, order=1, samples=512)

    assert [component.harmonic for component in components] == [1, 2, 3]
    for component in components:
        assert np.isclose(
            component.frequency,
            eccentric_harmonic_angular_frequency(source, component.harmonic) / (2.0 * np.pi),
        )
        assert component.tracefree_coefficients.shape == (5,)
        assert np.isclose(np.linalg.norm(component.propagation_direction), 1.0)
        assert np.all(np.isfinite(component.tracefree_coefficients))


def test_eccentric_harmonic_components_expose_fixed_physics_mode():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 3.0e-3,
        eccentricity=0.05,
        m1_solar=1.4,
        m2_solar=1.0,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
    )

    with pytest.raises(ValueError, match="unknown eccentric physics mode"):
        eccentric_harmonic_components(source, order=1, samples=512, physics_mode="bad")
    with pytest.raises(ValueError, match="evolution_mode='fixed'"):
        eccentric_harmonic_components(source, order=1, samples=512, evolution_mode="peters_mathews")


def _component_polarization_coefficients(source: EccentricBinaryParams, component):
    matrix = tracefree_matrix(component.tracefree_coefficients)
    eplus, ecross = source.polarization_basis()
    plus_weight = 0.5 * np.einsum("ij,ij", matrix, eplus)
    cross_weight = 0.5 * np.einsum("ij,ij", matrix, ecross)
    cosps = np.cos(2.0 * source.psi)
    sinps = np.sin(2.0 * source.psi)
    hplus_coeff = plus_weight * cosps - cross_weight * sinps
    hcross_coeff = plus_weight * sinps + cross_weight * cosps
    return hplus_coeff, hcross_coeff


def test_eccentric_sideband_specs_are_explicit_and_sorted():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.3,
        m1_solar=0.6,
        m2_solar=0.4,
    )

    specs = eccentric_sideband_specs(source, q_max=6)
    families = {spec.family for spec in specs}
    frequencies = np.array([spec.angular_frequency for spec in specs])

    assert {"quadrupole_plus", "quadrupole_minus", "radial"} <= families
    assert np.all(frequencies >= 0.0)
    assert np.all(np.diff(frequencies) >= 0.0)


def test_eccentric_sideband_components_reconstruct_high_eccentric_source():
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-3,
        eccentricity=0.3,
        m1_solar=0.6,
        m2_solar=0.4,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
        phi0=0.2,
    )
    projection_frequency = eccentric_harmonic_projection_angular_frequency(source)
    duration = 2.0 * np.pi / projection_frequency
    times = source.t0 + np.linspace(0.0, duration, 2048, endpoint=False)
    hplus, hcross = eccentric_polarizations(source, times)
    components = eccentric_sideband_components(source, q_max=16, samples=4096, fit_cycles=4.0)

    hplus_model = np.zeros_like(times, dtype=np.complex128)
    hcross_model = np.zeros_like(times, dtype=np.complex128)
    for component in components:
        hplus_coeff, hcross_coeff = _component_polarization_coefficients(source, component)
        carrier = np.exp(2.0j * np.pi * component.frequency * times)
        hplus_model += hplus_coeff * carrier
        hcross_model += hcross_coeff * carrier

    residual = np.sqrt(
        (
            np.linalg.norm(hplus - np.real(hplus_model)) ** 2
            + np.linalg.norm(hcross - np.real(hcross_model)) ** 2
        )
        / (np.linalg.norm(hplus) ** 2 + np.linalg.norm(hcross) ** 2)
    )

    assert residual < 1.0e-3


def test_fixed_eccentric_triliear_links_match_legacy_explicit_order():
    state = lisa_orbit(np.linspace(0.0, 10_000.0, 128))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-4,
        eccentricity=0.01,
        beta=0.2,
        lambda_=0.4,
        inclination=0.8,
    )

    legacy = eccentric_beta_link_responses(state, source, eccentric_order=1, harmonic_samples=128)
    fixed = fixed_eccentric_triliear_links(state, source, eccentric_order=1, harmonic_samples=128)

    assert set(fixed) == set(legacy)
    for label in legacy:
        assert np.allclose(fixed[label], legacy[label])


def test_fixed_eccentric_triliear_auto_order_rejects_high_eccentricity():
    state = lisa_orbit(np.linspace(0.0, 1_000.0, 64))
    source = EccentricBinaryParams(mean_motion=np.pi * 1.0e-4, eccentricity=0.2)

    with pytest.raises(ValueError, match="eccentricity < 0.1"):
        fixed_eccentric_triliear_links(state, source)


def test_eccentric_beta_link_responses_has_all_links():
    state = lisa_orbit(np.linspace(0.0, 10_000.0, 128))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-4,
        eccentricity=0.1,
        beta=0.2,
        lambda_=0.4,
        inclination=0.8,
    )

    links = eccentric_beta_link_responses(state, source, eccentric_order=1, harmonic_samples=512)

    assert set(links) == {"12", "23", "31", "13", "32", "21"}
    for value in links.values():
        assert value.shape == state.t.shape
        assert np.iscomplexobj(value)
        assert np.all(np.isfinite(value))


@pytest.mark.skipif(not has_package("pytdi"), reason="pytdi not installed")
def test_compute_triliear_xyz_eccentric_returns_channels():
    state = lisa_orbit(np.arange(512.0))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-4,
        eccentricity=0.1,
        beta=0.2,
        lambda_=0.4,
        inclination=0.8,
    )

    xyz = compute_triliear_xyz_eccentric(state, source, eccentric_order=1, harmonic_samples=512)

    assert set(xyz) == {"X", "Y", "Z"}
    for value in xyz.values():
        assert value.shape == state.t.shape
        assert np.iscomplexobj(value)
        assert np.all(np.isfinite(value))


@pytest.mark.skipif(not has_package("pytdi"), reason="pytdi not installed")
def test_compute_fixed_eccentric_triliear_xyz_returns_channels():
    state = lisa_orbit(np.arange(512.0))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-4,
        eccentricity=0.01,
        beta=0.2,
        lambda_=0.4,
        inclination=0.8,
    )

    xyz = compute_fixed_eccentric_triliear_xyz(state, source, harmonic_samples=128)

    assert set(xyz) == {"X", "Y", "Z"}
    for value in xyz.values():
        assert value.shape == state.t.shape
        assert np.iscomplexobj(value)
        assert np.all(np.isfinite(value))


@pytest.mark.skipif(not has_package("jax") or not has_package("pytdi"), reason="jax and pytdi are required")
def test_compute_fixed_eccentric_triliear_xyz_jax_returns_channels():
    state = lisa_orbit(np.arange(512.0))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-4,
        eccentricity=0.01,
        beta=0.2,
        lambda_=0.4,
        inclination=0.8,
    )

    xyz = compute_fixed_eccentric_triliear_xyz_jax(state, source, harmonic_samples=128, batch_size=1)

    assert set(xyz) == {"X", "Y", "Z"}
    for value in xyz.values():
        assert value.shape == state.t.shape
        assert np.iscomplexobj(value)
        assert np.all(np.isfinite(value))


@pytest.mark.skipif(not has_package("jax"), reason="jax not installed")
def test_fixed_eccentric_triliear_links_jax_accepts_sideband_basis():
    state = lisa_orbit(np.arange(512.0))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-4,
        eccentricity=0.2,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
    )

    links = fixed_eccentric_triliear_links_jax(
        state,
        source,
        basis="sideband",
        sideband_q_max=4,
        harmonic_samples=512,
        order="exact-transfer",
        batch_size=1,
    )

    assert set(links) == {"12", "23", "31", "13", "32", "21"}
    for value in links.values():
        assert value.shape == state.t.shape
        assert np.iscomplexobj(value)
        assert np.all(np.isfinite(value))


@pytest.mark.skipif(not has_package("jax") or not has_package("pytdi"), reason="jax and pytdi are required")
def test_compute_fixed_eccentric_sideband_xyz_jax_returns_channels():
    state = lisa_orbit(np.arange(512.0))
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-4,
        eccentricity=0.1,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
    )

    xyz = compute_fixed_eccentric_sideband_xyz_jax(state, source, q_max=4, harmonic_samples=512, batch_size=16)

    assert set(xyz) == {"X", "Y", "Z"}
    for value in xyz.values():
        assert value.shape == state.t.shape
        assert np.iscomplexobj(value)
        assert np.all(np.isfinite(value))
