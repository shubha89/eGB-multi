import numpy as np
import pytest

pytest.importorskip("egb_multi")
pytest.importorskip("lisaorbits")
pytest.importorskip("pytdi")
from xtdi.adapters.egb import XTDIChunkedNativeResponse, XTDIGWTDIMichelson
from egb_multi.comparison import TDIConvention
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits
from egb_multi.native_validation import ChunkedNativeResponse


@pytest.mark.parametrize("backend", ["numba","jax"])
@pytest.mark.parametrize("evolution", ["fixed","peters_mathews"])
def test_egb_native_adapter_with_real_retarded_geometry(backend,evolution):
    t = 1e6+np.arange(1047)*2.5
    convention = TDIConvention(2,101,7,"analytic_esa_trailing_BME_J2000","native_analytic_retarded_delays")
    orbit = default_lisaorbits("esa-trailing")
    source = EccentricBinaryParams(mean_motion=np.pi*.003,eccentricity=.3,
        beta=.3,lambda_=1.2,inclination=.9,phi0=.7,psi=.4,t0=t[0])
    reference = ChunkedNativeResponse(orbit,t,convention,block_samples=257)(source,evolution)
    engine = XTDIChunkedNativeResponse(orbit,t,convention,block_samples=257,backend=backend)
    actual = engine(source,evolution)
    for channel in ("A","E"):
        a,b = actual.channels[channel],reference.channels[channel]
        assert np.linalg.norm(a-b)/np.linalg.norm(b) < 1e-9
    assert len(engine.cache) <= 2
