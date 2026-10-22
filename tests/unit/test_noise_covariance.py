import numpy as np
import pytest

from egb_multi.noise_covariance import frozen_raw_tdi_transfer,frozen_native_secondary_covariance,moving_native_secondary_symbol
from egb_multi.tdi import LINK_LABELS


def test_frozen_transfer_matches_native_pytdi_monochromatic_inputs():
    pytest.importorskip('pytdi')
    from pytdi import Data
    from pytdi.michelson import X2,Y2,Z2
    n,dt,f=8192,5.,.003127
    t=np.arange(n)*dt
    delays={label:8.1+.06*j for j,label in enumerate(LINK_LABELS)}
    transfer,measurements=frozen_raw_tdi_transfer(np.array([f]),delays)
    # Verify independent science and local metrology mappings, unequal delays
    # and Fourier sign, rather than only matching the covariance formula.
    for measurement in ('sci_12','tmi_23','ref_31'):
        streams={name:np.zeros(n) for name in Data.MEASUREMENTS}
        streams[measurement]=np.cos(2*np.pi*f*t)
        data=Data(streams,{f'd_{label}':np.full(n,value) for label,value in delays.items()},1/dt)
        data.compute_delay_derivatives()
        for channel,combination in enumerate((X2,Y2,Z2)):
            actual=combination.build(data.delays,data.fs,data.delay_derivatives,order=5)(data.measurements,order=45,unit='frequency')
            expected=np.real(transfer[0,channel,measurements.index(measurement)]*np.exp(2j*np.pi*f*t))
            np.testing.assert_allclose(actual[256:-256],expected[256:-256],rtol=1e-9,atol=2e-10)


def test_native_covariance_is_hermitian_positive_and_equal_arm_AE_orthogonal():
    Instrument=pytest.importorskip('lisainstrument').Instrument
    instrument=Instrument(size=4096,dt=5,physics_upsampling=1,aafilter=None,
        telemetry_downsampling=4096,noises_f_min_hz=1e-7)
    f=np.array([2e-5,1e-4,.001,.003,.02])
    covariance,components=frozen_native_secondary_covariance(f,dict.fromkeys(LINK_LABELS,8.3),instrument)
    np.testing.assert_allclose(covariance,covariance.swapaxes(1,2).conj(),rtol=1e-14,atol=0)
    assert np.all(np.linalg.eigvalsh(covariance)>0)
    np.testing.assert_allclose(covariance[:,0,0],covariance[:,1,1],rtol=1e-10)
    assert np.max(abs(covariance[:,0,1])/covariance[:,0,0].real)<1e-9
    np.testing.assert_allclose(sum(components.values()),covariance,rtol=1e-14)


def test_frozen_model_rejects_missing_or_invalid_delays():
    with pytest.raises(ValueError): frozen_raw_tdi_transfer([.001],{'12':8.})
    with pytest.raises(ValueError): frozen_raw_tdi_transfer([0.],dict.fromkeys(LINK_LABELS,8.))


def test_infrared_frozen_polynomial_against_high_precision_native_terms():
    mp=pytest.importorskip('mpmath');pytest.importorskip('pytdi')
    from pytdi.michelson import X2,Y2,Z2
    f=np.geomspace(3e-10,3e-3,23)
    delays={label:8.1+.06*j for j,label in enumerate(LINK_LABELS)}
    actual,measurements=frozen_raw_tdi_transfer(f,delays)
    with mp.workdps(70):
        for channel,combination in enumerate((X2,Y2,Z2)):
            for measurement,polynomial in combination.components.items():
                column=measurements.index(measurement)
                expected=[]
                for frequency in f:
                    terms=[]
                    for coefficient,path in polynomial:
                        tau=mp.fsum((1 if op[0]=='D' else -1)*mp.mpf(delays[op[2:]]) for op in path)
                        terms.append(coefficient*mp.exp(-2j*mp.pi*float(frequency)*tau))
                    expected.append(complex(mp.fsum(terms)))
                expected=np.array(expected)
                # Native polynomial evaluations with a nonzero component have
                # relative precision even at periods of a hundred years.
                np.testing.assert_allclose(actual[:,channel,column],expected,rtol=3e-12,atol=1e-29)


def test_moving_symbol_reduces_to_unequal_static_native_covariance():
    Instrument=pytest.importorskip('lisainstrument').Instrument
    instrument=Instrument(size=4096,dt=1,physics_upsampling=1,aafilter=None,
        telemetry_downsampling=4096,noises_f_min_hz=1e-7)
    t=np.arange(512)*600.;f=np.array([2e-5,.001,.02])
    constant={label:8.1+.06*j for j,label in enumerate(LINK_LABELS)}
    frozen,_=frozen_native_secondary_covariance(f,constant,instrument)
    actual=moving_native_secondary_symbol(f,t,{k:np.full(t.size,v) for k,v in constant.items()},instrument,[128,256,384])
    scale=np.max(frozen[:,[0,1],[0,1]].real)
    np.testing.assert_allclose(actual,np.broadcast_to(frozen,actual.shape),rtol=1e-9,atol=1e-12*scale)
