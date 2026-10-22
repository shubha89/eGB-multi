from dataclasses import replace

import numpy as np
import pytest

from egb_multi.circular_seeds import rank_envelope_frequency_seeds,_batched_projection_scores
from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.circular_family_search import FrequencyBasin
from egb_multi.circular_fit import CircularAmplitudeBasis,profile_circular_orientation
from egb_multi.comparison import AEPSD,AEInnerProduct,TDIConvention
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.native_validation import ChunkedNativeResponse


@pytest.mark.parametrize('rank_deficient',[False,True])
def test_batched_projections_match_independent_real_svd_least_squares(rank_deficient):
    rng=np.random.default_rng(14615)
    reference=rng.normal(size=(2,37))+1j*rng.normal(size=(2,37))
    basis=rng.normal(size=(13,4,2,37))+1j*rng.normal(size=(13,4,2,37))
    if rank_deficient:
        basis[:,2]=basis[:,0]+2*basis[:,1]
        basis[:,3]=-3*basis[:,1]
    norm=float(np.vdot(reference,reference).real)+7.
    expected=[];target=np.r_[reference.real.ravel(),reference.imag.ravel()]
    for item in basis:
        design=np.concatenate((item.real.reshape(4,-1),item.imag.reshape(4,-1)),axis=1).T
        coefficients=np.linalg.lstsq(design,target,rcond=1e-6)[0]
        projected=design@coefficients
        expected.append(np.linalg.norm(projected)/np.sqrt(norm))
    np.testing.assert_allclose(_batched_projection_scores(basis,reference,norm),expected,rtol=2e-13,atol=2e-15)


def test_envelope_proposals_match_independent_native_offset_chirp_control():
    pytest.importorskip("pytdi");pytest.importorskip("lisaorbits");pytest.importorskip("jax")
    from egb_multi.external import default_lisaorbits
    t=np.arange(20480)*10.;duration=len(t)*10.;orbit=default_lisaorbits("esa-trailing")
    convention=TDIConvention(2,45,7,"native","retarded")
    prototype=EccentricBinaryParams(mean_motion=np.pi*641/duration,eccentricity=0,t0=t[-1]/2,
        beta=.43,lambda_=6.08,m1_solar=1.12,m2_solar=.32,phi0=0.)
    truth=replace(prototype,mean_motion=np.pi*(641+.713)/duration,fdot=-.517/duration**2,
        inclination=.91,phi0=.77,psi=.46)
    native=ChunkedNativeResponse(orbit,t,convention);signal=native(truth)
    f=np.linspace(0,.05,4097);sa=np.full(len(f),4e-42);se=1.2*sa
    psd=AEPSD(f,{"A":sa,"E":se},convention,dict(sidedness="one_sided",scaling="density",noise_model="seed_validation",
        CSD_convention="S_AE=E[tilde_A*conj(tilde_E)]"),.15*np.sqrt(sa*se)*np.exp(-2j*np.pi*f*20))
    metric=AEInnerProduct(signal,psd,(.002,.0045))
    response=InterpolatedCircularTDI(orbit,t,convention,envelope_intervals=256)
    seeds=rank_envelope_frequency_seeds(metric,response,prototype,FrequencyBasin(641/duration,4),
        sky=[(np.sin(prototype.beta),prototype.lambda_)],chirp_scaled_values=[-2,0,2],seed_samples=1024,window_half_width_cycles=48)
    assert round(seeds[0]["frequency_hz"]*duration)==642
    assert seeds[0]["chirp_scaled"]==0
    exact=profile_circular_orientation(metric,CircularAmplitudeBasis.generate(seeds[0]["source"],native))["overlap"]
    assert abs(exact-seeds[0]["seed_score"])<2e-3
    with pytest.raises(ValueError,match="FFT"):
        rank_envelope_frequency_seeds(metric,response,prototype,FrequencyBasin(641/duration,4),sky=[(0,0)],
            chirp_scaled_values=[0],seed_samples=63,window_half_width_cycles=48)
