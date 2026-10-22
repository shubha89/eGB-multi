from dataclasses import replace

import numpy as np
import pytest

from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.circular_fit import CircularAmplitudeBasis
from egb_multi.native_validation import ChunkedNativeResponse
from egb_multi.comparison import TDIConvention
from egb_multi.eccentric import EccentricBinaryParams


@pytest.mark.parametrize('frequency,fdot',[(.001,0.),(.003,1e-12)])
def test_carrier_envelope_matches_native_physical_orientation_and_chirp(frequency,fdot):
    pytest.importorskip('pytdi');pytest.importorskip('lisaorbits');pytest.importorskip('jax')
    from egb_multi.external import default_lisaorbits
    t=np.arange(4096)*10.;orbit=default_lisaorbits('esa-trailing')
    conv=TDIConvention(2,45,5,'native_esa_trailing','deterministic_retarded_TCB')
    for evaluator in (InterpolatedCircularTDI,ChunkedNativeResponse):
        with pytest.raises(ValueError,match='fractional_frequency'):
            evaluator(orbit,t,replace(conv,unit='frequency_hz'))
    source=EccentricBinaryParams(mean_motion=np.pi*frequency,eccentricity=0,fdot=fdot,t0=t[-1]/2,
        inclination=.91,beta=.43,lambda_=6.08,phi0=.77,psi=.46,m1_solar=1.12,m2_solar=.32)
    approximated=InterpolatedCircularTDI(orbit,t,conv,envelope_intervals=256)
    actual=approximated(source);native=ChunkedNativeResponse(orbit,t,conv)(source)
    for channel in ('A','E'):
        error=np.linalg.norm(actual.channels[channel]-native.channels[channel])/np.linalg.norm(native.channels[channel])
        assert error<2e-7
    second=approximated(replace(source,inclination=.2,psi=.8,phi0=1.2))
    original=ChunkedNativeResponse(orbit,t,conv)(replace(source,inclination=.2,psi=.8,phi0=1.2))
    for channel in ('A','E'):
        assert np.linalg.norm(second.channels[channel]-original.channels[channel])/np.linalg.norm(original.channels[channel])<2e-7
    with pytest.raises(ValueError,match='only'):approximated(replace(source,eccentricity=.01))
    direct=CircularAmplitudeBasis.generate(source,approximated)
    sampled=CircularAmplitudeBasis.generate(source,lambda s:approximated(s))
    assert direct.source==source
    for a,b in zip(direct.responses,sampled.responses,strict=True):
        for channel in ('A','E'):
            # The small phase-independent term needs an absolute scale check.
            assert np.max(abs(a.channels[channel]-b.channels[channel]))<1e-12*np.max(abs(actual.channels[channel]))
    reconstructed=direct.reconstruct(inclination=.2,psi=.8,phi0=1.2)
    for channel in ('A','E'):
        assert np.linalg.norm(reconstructed.channels[channel]-second.channels[channel])/np.linalg.norm(second.channels[channel])<1e-12
