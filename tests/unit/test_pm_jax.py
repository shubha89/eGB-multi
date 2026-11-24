"""Annual PM source tests: phase in radians, unfitted two-polarization residual."""
from dataclasses import replace

import jax
import numpy as np
import pytest

from egb_multi.constants import G_M3_KG_S2, C_M_PER_S
from egb_multi.eccentric import (
    EccentricBinaryParams, eccentric_fundamental_angular_frequencies,
    eccentric_polarizations, peters_mathews_evolution,
)
from egb_multi.pm_jax import prepare_pm, pm_evolution, pm_strain

YEAR = 31557600.
BASE = EccentricBinaryParams(np.pi*.003, .1, m1_solar=1.1, m2_solar=.4,
                             phi0=.7, inclination=.9, t0=1048576.25)


@pytest.mark.parametrize("frequency,eccentricity", [
    (.001, 0.), (.001, .19), (.003, .1), (.010, .19), (.010, .6),
])
@pytest.mark.parametrize("physics", ["1pn", "newtonian", "1pn_no_periastron"])
def test_annual_pm_matches_scipy_without_alignment(frequency, eccentricity, physics):
    src = replace(BASE, mean_motion=np.pi*frequency, eccentricity=eccentricity)
    rng = np.random.default_rng(56)
    t = src.t0 + np.r_[-5000., 0., np.sort(rng.uniform(0., YEAR, 1024)), YEAR+5000.]
    plan = prepare_pm(src, t.min(), t.max(), physics_mode=physics)
    n, e, radial, longitude = map(np.asarray, pm_evolution(plan, t))
    n0, _ = eccentric_fundamental_angular_frequencies(src, physics_mode=physics)
    ref = peters_mathews_evolution(replace(src, mean_motion=n0), t,
                                   phase_mode="peters_mathews" if physics=="1pn" else None,
                                   rtol=2e-13, phase_atol=1e-12)
    np.testing.assert_allclose(n, ref["mean_motion"], rtol=3e-13)
    np.testing.assert_allclose(e, ref["eccentricity"], rtol=3e-13, atol=1e-16)
    np.testing.assert_allclose(radial, ref["mean_anomaly"], rtol=0, atol=3e-9)
    np.testing.assert_allclose(longitude, ref.get("integrated_phase", ref["mean_anomaly"]),
                               rtol=0, atol=3e-9)
    hp, hc = eccentric_polarizations(src, t, physics_mode=physics, evolution_mode="peters_mathews",
                                     pm_ode_rtol=2e-13, pm_ode_phase_atol=1e-12)
    ref_h = hp-1j*hc
    h = np.asarray(pm_strain(plan, t))
    assert np.linalg.norm(h-ref_h)/np.linalg.norm(ref_h) < 1e-8
    # XLA can use different vector instructions for different block sizes.
    block = np.asarray(pm_strain(plan, t[-8:]))
    assert np.linalg.norm(block-h[-8:])/np.linalg.norm(h[-8:]) < 1e-12


def test_circular_pm_analytic_both_directions_and_shaped_queries():
    src = replace(BASE, mean_motion=np.pi*.01, eccentricity=0.)
    t = src.t0 + YEAR*np.array([[1., -.8], [0., .05]])
    plan = prepare_pm(src, t.min(), t.max(), physics_mode="newtonian")
    n, e, l, lam = pm_evolution(plan, t)
    n0 = src.mean_motion
    tau_m = G_M3_KG_S2*src.total_mass_kg/C_M_PER_S**3
    k = (96/5)*src.symmetric_mass_ratio*tau_m**(5/3)
    tc = 3/(8*k*n0**(8/3))
    log_t = np.log1p(-(t-src.t0)/tc)
    np.testing.assert_allclose(n, n0*np.exp(-3*log_t/8), rtol=3e-15)
    np.testing.assert_array_equal(e, np.zeros_like(t))
    np.testing.assert_allclose(l, -n0*tc*(8/5)*np.expm1(5*log_t/8), rtol=4e-15)
    np.testing.assert_array_equal(l, lam)
    assert np.isnan(pm_evolution(plan, t.max()+1.)[0])


def test_refines_strong_chirp_and_rejects_unresolved_evolution(monkeypatch):
    src = replace(BASE, m1_solar=14., m2_solar=12., eccentricity=.65, mean_motion=np.pi*.012)
    end = src.t0+YEAR
    with pytest.raises(ValueError, match="refinement"):
        prepare_pm(src, src.t0, end, max_steps=1)
    plan = prepare_pm(src, src.t0, end)
    assert len(plan.coefficients) > 1
    # An independent eighth-order solve with a capped step resolves this
    # much stronger chirp. A relative tolerance on million-radian unwrapped
    # phases alone can leave an appreciable dense-output error.
    from functools import partial
    from egb_multi import eccentric
    monkeypatch.setattr(eccentric, "solve_ivp", partial(eccentric.solve_ivp,
                                                        method="DOP853", max_step=YEAR/64))
    times = np.linspace(src.t0, end, 333)
    hp, hc = eccentric_polarizations(src, times, evolution_mode="peters_mathews",
                                     pm_ode_rtol=2e-13, pm_ode_phase_atol=1e-12)
    ref = hp-1j*hc
    assert np.linalg.norm(pm_strain(plan, times)-ref)/np.linalg.norm(ref) < 1e-8
    assert isinstance(pm_strain(plan, times), jax.Array)


def test_pm_batch_and_uniform_sampling_paths():
    from egb_multi.eccentric_jax import eccentric_complex_strain_batch_jax
    from egb_multi.pm_sampling import pm_uniform_polarizations
    dt = 2.5
    src = replace(BASE, mean_motion=np.pi*.01)
    size = 24579  # includes an incomplete final orbital block
    times = src.t0+np.arange(size)*dt
    other = replace(src, eccentricity=0.)
    direct, info = pm_uniform_polarizations(src, times[0], dt, size, return_info=True)
    assert info["method"] == "direct"
    sampled, info = pm_uniform_polarizations(src, times[0], dt, size,
                                             method="interpolate", return_info=True)
    assert info["method"] == "interpolate"
    assert np.max(np.abs(sampled-direct))/np.max(np.abs(direct)) < 1e-10
    from egb_multi.stride import sampled_polarizations
    adapter = sampled_polarizations(src, times, source_backend="jax", evolution_mode="peters_mathews",
                                    pm_sampling="interpolate")
    np.testing.assert_array_equal(adapter.values, sampled)
    with pytest.raises(ValueError, match="full PM"):
        sampled_polarizations(src, times, evolution_mode="fixed", pm_sampling="interpolate")
    batch = eccentric_complex_strain_batch_jax([src, other], times)
    np.testing.assert_allclose(batch[0], direct[0]-1j*direct[1], rtol=1e-12, atol=1e-35)
    for src2 in (src, replace(src, eccentricity=.68)):
        out, info = pm_uniform_polarizations(src2, times[0], dt, 57,
                                             method="interpolate", return_info=True)
        assert info["method"] == "direct"
        assert np.isfinite(out).all()


def test_noncommensurate_frequency_and_late_epoch_match_reference():
    from egb_multi.stride import sampled_pm_uniform
    src = replace(BASE, mean_motion=np.pi*.0031731, eccentricity=.274, t0=1_000_000_000.125)
    times = src.t0+YEAR-2000.+np.arange(4211)*.3125
    sampled = sampled_pm_uniform(src, times[0], .3125, len(times), method="interpolate")
    hp, hc = eccentric_polarizations(src, times, evolution_mode="peters_mathews",
                                     pm_ode_rtol=2e-13, pm_ode_phase_atol=1e-12)
    ref = np.asarray((hp, hc))
    assert np.linalg.norm(sampled.values-ref)/np.linalg.norm(ref) < 1e-8


@pytest.mark.parametrize("start,end", [(BASE.t0, BASE.t0), (BASE.t0-800., BASE.t0-50.)])
def test_epoch_and_past_only_plans(start, end):
    plan = prepare_pm(BASE, start, end)
    times = np.linspace(start, end, 9)[::-1]
    hp, hc = eccentric_polarizations(BASE, times, evolution_mode="peters_mathews")
    ref = hp-1j*hc
    assert np.linalg.norm(pm_strain(plan, times)-ref)/np.linalg.norm(ref) < 1e-12
