"""Output-partition independence and bounded-memory native response checks."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from egb_multi.eccentric import (EccentricBinaryParams,eccentric_dynamics,
    peters_mathews_evolution,peters_mathews_derivatives,
    _azimuthal_from_radial_frequency_1pn,G_M3_KG_S2)
from egb_multi.native_validation import ChunkedNativeResponse,normalized_overlap_change_bound
from egb_multi.comparison import TDIConvention


@pytest.mark.parametrize('mode',['peters_mathews','peters_mathews_orbital_only','peters_mathews_eccentricity_only'])
def test_evolving_phase_is_independent_of_requested_sample_partition(mode):
    source = EccentricBinaryParams(mean_motion=np.pi*.003,eccentricity=.5,
        m1_solar=1.4,m2_solar=1.4,t0=31557600/2,phi0=.7)
    t = np.linspace(0,31557600,1025)
    full = eccentric_dynamics(source,t,evolution_mode=mode)
    for indices in (np.arange(83),np.arange(940,1025),np.array([0,0,321,512,512,1024])):
        partial = eccentric_dynamics(source,t[indices],evolution_mode=mode)
        np.testing.assert_allclose(partial['phi'],full['phi'][indices],rtol=0,atol=2e-8)
        np.testing.assert_allclose(partial['mean_anomaly'],full['mean_anomaly'][indices],rtol=0,atol=2e-8)


def test_pm_phase_agrees_with_independent_high_accuracy_DOP853_integration():
    source = EccentricBinaryParams(mean_motion=np.pi*.003,eccentricity=.5,m1_solar=1.4,m2_solar=1.4)
    t = np.linspace(0,31557600,101)
    gm = G_M3_KG_S2*source.total_mass_kg
    def rhs(time,state):
        n,e = state[:2]
        dn,de = peters_mathews_derivatives(n,e,source.total_mass_kg,source.symmetric_mass_ratio)
        return [float(dn),float(de),n,float(_azimuthal_from_radial_frequency_1pn(gm,n,e))]
    independent = solve_ivp(rhs,(0,t[-1]),[source.mean_motion,source.eccentricity,0,0],
        method='DOP853',rtol=2e-13,atol=[1e-20,1e-16,1e-10,1e-10],t_eval=t)
    assert independent.success
    actual = peters_mathews_evolution(source,t,phase_mode='peters_mathews')
    np.testing.assert_allclose(actual['integrated_phase'],independent.y[3],rtol=0,atol=2e-8)
    np.testing.assert_allclose(actual['mean_anomaly'],independent.y[2],rtol=0,atol=2e-8)


@pytest.mark.parametrize('eccentricity,evolution',[(0.,'fixed'),(.5,'peters_mathews')])
def test_chunked_response_agrees_with_original_native_TDI(eccentricity,evolution):
    pytest.importorskip('pytdi');pytest.importorskip('lisaorbits');pytest.importorskip('jax')
    from egb_multi.external import default_lisaorbits
    from egb_multi.eccentric_jax import eccentric_links_jax
    from egb_multi.noise import aet_from_xyz,xyz_from_pytdi_data
    from egb_multi.tdi import pytdi_data_from_links
    from egb_multi.external import state_from_lisaorbits,link_delays_from_lisaorbits
    convention = TDIConvention(2,45,5,'native_esa_trailing','deterministic_retarded_TCB')
    orbit,t = default_lisaorbits('esa-trailing'),np.arange(1024)*10.
    source = EccentricBinaryParams(mean_motion=np.pi*.003,eccentricity=eccentricity,
        inclination=.9,beta=.3,lambda_=1.2,phi0=.7,psi=.4,
        fdot=1e-11 if evolution=='fixed' else 0.)
    chunked = ChunkedNativeResponse(orbit,t,convention,block_samples=257)
    actual = chunked(source,evolution)
    padded = t[0]+np.arange(-chunked.halo,t.size+chunked.halo)*10.
    state = state_from_lisaorbits(orbit,padded,frame='ecliptic')
    from egb_multi.eccentric_jax import precompute_jax_link_geometry
    geometry = precompute_jax_link_geometry(state,orbits=orbit,frame='ecliptic')
    links = eccentric_links_jax(source,geometry,physics_mode='1pn',evolution_mode=evolution,batch_size=1)
    data = pytdi_data_from_links(state,links,link_delays=link_delays_from_lisaorbits(orbit,padded))
    reference = aet_from_xyz(xyz_from_pytdi_data(data,generation=2,order=45,delay_order=5))
    np.testing.assert_array_equal(actual.t,t)
    for ch in ('A','E'):
        original = reference[ch][chunked.halo:-chunked.halo]
        assert np.linalg.norm(actual.channels[ch]-original)/np.linalg.norm(original) < 2e-8
    assert len(chunked.cache) <= 2
    with pytest.raises(ValueError,match='Resolve auto'):
        chunked(source,'auto')


def test_overlap_perturbation_bound_has_explicit_uniformity_condition():
    assert normalized_overlap_change_bound(0,0)==0
    assert normalized_overlap_change_bound(.001,.002) > .006
    with pytest.raises(ValueError):
        normalized_overlap_change_bound(1,0)


def test_source_ODE_tolerances_reach_the_retarded_native_response():
    pytest.importorskip('pytdi');pytest.importorskip('lisaorbits');pytest.importorskip('jax')
    from egb_multi.external import default_lisaorbits
    convention=TDIConvention(2,45,5,'native_esa_trailing','deterministic_retarded_TCB')
    engine=ChunkedNativeResponse(default_lisaorbits('esa-trailing'),np.arange(128)*10.,convention,block_samples=128)
    source=EccentricBinaryParams(mean_motion=np.pi*.003,eccentricity=.3)
    with pytest.raises(ValueError,match='tolerances'):
        engine(source,'peters_mathews',pm_ode_rtol=0.)
    default=engine(source,'peters_mathews')
    tight=engine(source,'peters_mathews',pm_ode_rtol=2e-13,pm_ode_phase_atol=1e-10)
    for ch in ('A','E'):
        assert np.linalg.norm(default.channels[ch]-tight.channels[ch])/np.linalg.norm(tight.channels[ch])<1e-7
