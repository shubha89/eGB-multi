"""Independent synthetic recovery and domain checks for the full family search."""

from dataclasses import replace

import numpy as np
import pytest

from egb_multi.circular_family_search import (CircularFamilySearch, FrequencyBasin,
    equal_area_sky, mass_coordinates, ordered_masses, select_frequency_basins,
    _unscale_optimizer_coordinates)
from egb_multi.comparison import AESeries, AEPSD, AEInnerProduct, TDIConvention
from egb_multi.eccentric import EccentricBinaryParams


def analytic_problem(covariance=False):
    t = np.arange(512)*20.
    duration, epoch = t.size*20., .5*(t[0]+t[-1])
    convention = TDIConvention(2,3,3,'synthetic_family_test','analytic_response')
    def response(source):
        tau = (t-source.t0)/duration
        cb = np.cos(source.beta)
        phase = 2*source.phi0 + 2*source.mean_motion*(t-source.t0) + np.pi*source.fdot*(t-source.t0)**2
        # Artificial independently specified modulation makes sky and masses
        # measurable. This tests search mechanics, not physical LISA accuracy.
        phase += 1.1*np.sin(source.beta)*np.sin(4*np.pi*tau)
        phase += .9*cb*np.cos(source.lambda_)*np.cos(6*np.pi*tau)
        phase += .8*cb*np.sin(source.lambda_)*np.sin(8*np.pi*tau)
        shape = 1.+.25*source.m1_solar*np.cos(10*np.pi*tau)+.3*source.m2_solar*np.sin(12*np.pi*tau)
        u = np.cos(source.inclination)
        hp = (1+u*u)*shape*np.cos(phase)+(1-u*u)*.02*np.sin(2*np.pi*.0015*t)
        hc = 2*u*shape*np.sin(phase)
        p,q = np.cos(2*source.psi),np.sin(2*source.psi)
        return AESeries(t,{ch:hp*(p*fp-q*fq)+hc*(q*fp+p*fq)
            for ch,fp,fq in (('A',1.,.22),('E',.35,1.))},convention)
    prototype = EccentricBinaryParams(mean_motion=np.pi*.004,eccentricity=0.,t0=epoch,
        beta=-.6,lambda_=.8,m1_solar=.5,m2_solar=.5)
    truth = replace(prototype,mean_motion=np.pi*(.004+.313/duration),fdot=-.717/duration**2,
        beta=.43,lambda_=6.08,m1_solar=1.12,m2_solar=.32,inclination=.91,phi0=.77,psi=.46)
    signal = response(truth)
    f = np.fft.rfftfreq(t.size,20.)
    psd = AEPSD(f,{'A':np.full(f.size,2.),'E':np.full(f.size,3.)},convention,
        {'sidedness':'one_sided','scaling':'density','noise_model':'synthetic_constant_covariance',
         'CSD_convention':'S_AE=E[tilde_A*conj(tilde_E)]'},
        np.full(f.size,.2+.3j) if covariance else None)
    metric = AEInnerProduct(signal,psd,(.0005,.012))
    search = CircularFamilySearch(metric,prototype,response,
        basins=[FrequencyBasin(.004,2.)],chirp_scaled_bounds=(-2.,2.),
        evaluation_time_bounds_s=(t[0],t[-1]))
    return search,truth,response,signal


@pytest.mark.parametrize('covariance',[False,True])
def test_unknown_sky_masses_frequency_and_chirp_are_recovered(covariance):
    search,truth,response,_ = analytic_problem(covariance)
    norm, timestamps, mask = search.metric.reference_norm,search.metric.t.copy(),search.metric.mask.copy()
    initial = search.evaluate(search.coordinates_of(search.prototype,0))['profile']['overlap']
    result = search.run(sky_points=24,coarse_shape=(3,3),starts_per_basin=3)
    refined = search.refine_best()
    best = refined['best']
    recovered = EccentricBinaryParams(**best['source'])
    assert best['profile']['overlap'] > 1.-2e-8
    assert best['profile']['overlap'] >= result['best']['profile']['overlap'] >= initial
    assert abs((recovered.mean_motion-truth.mean_motion)/np.pi*search.duration_s) < 5e-4
    assert abs((recovered.fdot-truth.fdot)*search.duration_s**2) < 2e-3
    assert np.linalg.norm(recovered.propagation_direction()-truth.propagation_direction()) < 5e-4
    assert abs(recovered.m1_solar-truth.m1_solar) < 2e-3
    assert abs(recovered.m2_solar-truth.m2_solar) < 2e-3
    assert search.metric.compare(response(recovered))['overlap'] > 1.-2e-8
    assert search.metric.reference_norm == norm
    np.testing.assert_array_equal(search.metric.t,timestamps)
    np.testing.assert_array_equal(search.metric.mask,mask)


@pytest.mark.parametrize('masses',[(.1,.1),(1.4,.1),(1.4,1.4),(.8,.6),(1.2,.2)])
def test_ordered_mass_mapping_covers_triangle_and_round_trips(masses):
    np.testing.assert_allclose(ordered_masses(mass_coordinates(*masses)),masses,rtol=0,atol=3e-16)


def test_periodic_longitude_and_previous_candidate_retention():
    search,truth,_,_ = analytic_problem()
    point = search.coordinates_of(truth,0)
    original = search.evaluate(point)
    # A zero/seam angle has an exactly representable modulo counterpart.
    point[3] = 0.
    seam = search.evaluate(point)
    point[3] = 2*np.pi
    assert search.evaluate(point) is seam
    assert search.retain(truth) is original
    assert original['profile']['overlap'] > 1.-1e-10
    with pytest.raises(ValueError,match='outside'):
        bad = point.copy();bad[4]=1.1;search.evaluate(bad)
    with pytest.raises(ValueError,match='ordered'):
        mass_coordinates(.2,.8)


def test_harmonic_seed_selection_uses_same_covariance_metric_and_quantifies_coverage():
    search,_,_,signal = analytic_problem(True)
    t = signal.t
    # Exactly resolved independent tones, with the higher harmonic strongest.
    f1,f2 = 30/search.duration_s,65/search.duration_s
    tones = AESeries(t,{ch:np.cos(2*np.pi*f1*t)+2*np.sin(2*np.pi*f2*t)
        for ch in ('A','E')},signal.convention)
    basins,record = select_frequency_basins(search.metric,tones,half_width_cycles=4,
        target_coverage=.999,extra_frequencies=(f1,))
    assert len(basins)==2
    assert any(abs(b.center_hz-f2)<search.metric.df for b in basins)
    assert record['covered_seed_spectrum_power_fraction']>.999
    assert record['fit_window']=='none'
    np.testing.assert_array_equal(search.metric.reference,search.metric.prepare(signal))


def test_equal_area_sky_includes_poles_and_invalid_basin_fails_before_generation():
    sky = equal_area_sky(16)
    assert set(sky[-2:,0])=={-1.,1.}
    assert np.all((sky[:,1]>=0)&(sky[:,1]<2*np.pi))
    search,_,response,_ = analytic_problem()
    with pytest.raises(ValueError,match='edge'):
        CircularFamilySearch(search.metric,search.prototype,response,
            basins=[FrequencyBasin(.00055,4.)],chirp_scaled_bounds=(-2.,2.),
            evaluation_time_bounds_s=(search.metric.t[0],search.metric.t[-1]))


def test_multibasin_search_selects_stronger_line_without_renormalizing_signal():
    original,truth,response,signal = analytic_problem()
    secondary = replace(truth,mean_motion=np.pi*.009)
    other = response(secondary)
    mixed = AESeries(signal.t,{ch:signal.channels[ch]+2*other.channels[ch] for ch in ('A','E')},signal.convention)
    f = np.fft.rfftfreq(signal.t.size,20.)
    psd = AEPSD(f,{'A':np.full(f.size,2.),'E':np.full(f.size,3.)},signal.convention,
        {'sidedness':'one_sided','scaling':'density','noise_model':'synthetic_two_lines'})
    metric = AEInnerProduct(mixed,psd,original.metric.band_hz)
    search = CircularFamilySearch(metric,original.prototype,response,
        basins=[FrequencyBasin(.004,2.),FrequencyBasin(.009,2.)],chirp_scaled_bounds=(-2.,2.),
        evaluation_time_bounds_s=(metric.t[0],metric.t[-1]))
    result = search.run(sky_points=12,coarse_shape=(3,3),starts_per_basin=2)
    assert len(result['basins_results'])==2
    assert result['best']['frequency_hz']>.008
    assert .8 < result['best']['profile']['overlap'] < .98
    assert result['basins_results'][1]['best']['profile']['overlap'] > result['basins_results'][0]['best']['profile']['overlap']
    assert all(item['best']['profile']['reference_norm']==metric.reference_norm for item in result['basins_results'])


def test_optimizer_chart_preserves_real_failed_chirp_endpoint_and_both_mass_endpoints():
    # Curvature from the saved high-PM failure. This scale makes -24*s/s
    # one ulp smaller than the legal lower bound, before any SLSQP step.
    scales = np.array([.465939803, np.sqrt(.007513868044359273), 6.24303601, 3.65284909, .001, .001])
    bounds = np.array([[-96.,96.],[-24.,24.],[-1.,1.],[2.,8.],[0.,1.],[0.,1.]])
    scaled = bounds*scales[:,None]
    assert scaled[1,0]/scales[1] < bounds[1,0]
    for side in (0,1):
        decoded,record = _unscale_optimizer_coordinates(scaled[:,side],scales,bounds,scaled)
        np.testing.assert_array_equal(decoded,bounds[:,side])
        assert record['maximum_correction_in_bound_ulps'] <= 1
    interior = np.array([.713,-.517,.43,6.08,.32,.77])
    y = interior*scales
    decoded,record = _unscale_optimizer_coordinates(y,scales,bounds,scaled)
    np.testing.assert_array_equal(decoded,y/scales)
    assert record['corrected_axes'] == []
    bad = scaled[:,0].copy();bad[1] -= 1e-8
    with pytest.raises(ValueError,match='scaled domain'):
        _unscale_optimizer_coordinates(bad,scales,bounds,scaled)


def test_public_domain_stays_strict_at_one_ulp_outside_bounds():
    search,truth,_,_ = analytic_problem()
    for axis,edge,direction in [(0,-2.,-np.inf),(0,2.,np.inf),(1,-2.,-np.inf),
                                (1,2.,np.inf),(2,1.,np.inf),(4,1.,np.inf),(5,0.,-np.inf)]:
        point = search.coordinates_of(truth,0)
        point[axis] = np.nextafter(edge,direction)
        with pytest.raises(ValueError,match='outside'):
            search.evaluate(point)


def test_bounded_local_fit_recovers_independent_quadratic_minimum_from_active_endpoint():
    # Independent analytic problem: every coordinate, including both masses,
    # contributes to a positive definite six-dimensional quadratic objective.
    search = CircularFamilySearch.__new__(CircularFamilySearch)
    search.basins = (FrequencyBasin(.01,96.),)
    search.chirp_scaled_bounds = (-24.,24.)
    start = np.array([-24.,-24.,-.5416666666666667,5.500226292038235,.30769230769230776,0.])
    target = np.array([-23.9,-23.5,-.50,5.55,.45,.30])
    curvature = np.array([.21709990043659866,.007513868044359273,38.975498625642494,13.343306494268264,.3,.4])
    observations = []
    def evaluate(point,index):
        bounds = np.array(search._bounds(index,start[3]))
        if np.any(point<bounds[:,0]) or np.any(point>bounds[:,1]):
            raise ValueError('outside quadratic domain')
        observations.append(point.copy())
        return {'profile':{'overlap':float(1-.5*np.sum(curvature*(point-target)**2))}}
    search.evaluate = evaluate
    fit,point,conditioning = search._local_fit(start,0,ftol=1e-12,maxiter=40,finite_difference_step=1e-6)
    assert fit.success
    np.testing.assert_allclose(point,target,rtol=0,atol=3e-5)
    assert len(observations)>10
    assert conditioning['optimizer_chart_endpoint_roundoff_control']['maximum_correction_in_bound_ulps']<=4
