from dataclasses import replace

import numpy as np
import pytest

from egb_multi.circular_refinement import refine_with_waveform_jacobian, doppler_transport_sky_seed
from test_circular_family_search import analytic_problem


@pytest.mark.parametrize('covariance',[False,True])
def test_correlated_waveform_refinement_recovers_all_intrinsic_coordinates(covariance):
    search,truth,response,signal=analytic_problem(covariance)
    start=replace(truth,mean_motion=truth.mean_motion+np.pi*.03/search.duration_s,
        fdot=truth.fdot+.05/search.duration_s**2,beta=truth.beta+.03,
        lambda_=truth.lambda_-.04,m1_solar=truth.m1_solar-.02,m2_solar=truth.m2_solar+.04)
    search.retain(start)
    original=search.result()['best']['profile']['overlap']
    result=refine_with_waveform_jacobian(search)
    assert result['best']['profile']['overlap']>1-1e-10
    assert result['best']['profile']['overlap']>=original
    recovered=result['best']['source']
    assert abs(recovered['m1_solar']-truth.m1_solar)<1e-4
    assert abs(recovered['m2_solar']-truth.m2_solar)<1e-4
    assert all(row['after_overlap']>=row['before_overlap'] for row in result['waveform_jacobian_refinement']['iterations'])


def test_invalid_SVD_tolerance_is_rejected():
    search,truth,_,_=analytic_problem()
    search.retain(truth)
    with pytest.raises(ValueError): refine_with_waveform_jacobian(search,relative_rank_tolerance=0)


def test_sky_transport_preserves_local_observed_phase_derivatives():
    _,truth,_,_=analytic_problem()
    r=np.array([400.,-250.,100.]);v=np.array([1e-4,3e-5,-2e-5]);a=np.array([2e-11,-1e-11,0.])
    def observed(s):
        k=s.propagation_direction();rate=1-k@v
        f=s.mean_motion/np.pi-s.fdot*(k@r)
        return np.array([f*rate,s.fdot*rate**2-f*(k@a)])
    candidate=doppler_transport_sky_seed(truth,-.7,8.,r,v,a)
    np.testing.assert_allclose(observed(candidate),observed(truth),rtol=2e-14,atol=1e-20)
    assert candidate.m1_solar==truth.m1_solar and candidate.m2_solar==truth.m2_solar
    assert candidate.phi0==truth.phi0 and candidate.t0==truth.t0
    assert 0<=candidate.lambda_<2*np.pi


def test_sky_transport_zero_motion_and_invalid_inputs():
    _,truth,_,_=analytic_problem()
    candidate=doppler_transport_sky_seed(truth,.4,.8,np.zeros(3),np.zeros(3),np.zeros(3))
    assert candidate.mean_motion==truth.mean_motion and candidate.fdot==truth.fdot
    with pytest.raises(ValueError): doppler_transport_sky_seed(truth,2.,.8,np.zeros(3),np.zeros(3),np.zeros(3))
    with pytest.raises(ValueError): doppler_transport_sky_seed(truth,.4,.8,np.zeros(2),np.zeros(3),np.zeros(3))
