import numpy as np
import pytest

from egb_multi.circular_spectral import CircularSpectralTDI
from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.circular_fit import CircularAmplitudeBasis,profile_circular_orientation
from egb_multi.comparison import AEPSD,AEInnerProduct,TDIConvention
from egb_multi.eccentric import EccentricBinaryParams


def test_spectral_six_response_profile_matches_sampled_nonzero_phase_basis():
    pytest.importorskip("pytdi");pytest.importorskip("lisaorbits")
    from egb_multi.external import default_lisaorbits
    t=np.arange(8192)*10.;duration=len(t)*10.
    convention=TDIConvention(2,81,7,"native","retarded")
    source=EccentricBinaryParams(mean_motion=np.pi*(.003+.713/duration),eccentricity=0.,
        fdot=-.517/duration**2,t0=t[-1]/2,beta=.43,lambda_=6.08,m1_solar=.8,m2_solar=.3,
        phi0=.77,psi=.46,inclination=.91)
    response=InterpolatedCircularTDI(default_lisaorbits("esa-trailing"),t,convention,envelope_intervals=256)
    signal=response(source);f=np.linspace(0,.05,4097);sa=np.full(len(f),7e-42);se=1.3*sa
    psd=AEPSD(f,{"A":sa,"E":se},convention,dict(sidedness="one_sided",scaling="density",noise_model="spectral_test",
        CSD_convention="S_AE=E[tilde_A*conj(tilde_E)]"),.2*np.sqrt(sa*se)*np.exp(-2j*np.pi*f*20))
    metric=AEInnerProduct(signal,psd,(.0006,.015));basis=CircularAmplitudeBasis.generate(source,response)
    direct=np.array([metric.prepare(row) for row in basis.responses])
    spectral=CircularSpectralTDI(response,metric);prepared=spectral.prepared_basis(source)
    assert prepared.source==source
    assert np.linalg.norm(prepared.prepared-direct)/np.linalg.norm(direct)<5e-10
    sampled=profile_circular_orientation(metric,basis)
    approximate=spectral.profile_source(metric,source)
    assert abs(sampled["overlap"]-approximate["overlap"])<1e-10
    assert approximate["overlap"]>1-2e-10
    with pytest.raises(ValueError,match="immutable"):
        spectral.profile_source(AEInnerProduct(signal,psd,(.0006,.015)),source)
