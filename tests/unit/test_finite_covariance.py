import numpy as np
from scipy.linalg import toeplitz

from egb_multi.comparison import AESeries,AEPSD,AEInnerProduct,TDIConvention
from egb_multi.finite_covariance import FiniteWindowAEInnerProduct


def test_finite_band_metric_matches_independent_dense_real_covariance():
    n,m,dt=48,768,5.
    f=np.fft.rfftfreq(m,dt);fs=1/dt
    two=np.zeros((len(f),2,2),complex)
    two[:,0,0]=2+np.cos(2*np.pi*f/fs)
    two[:,1,1]=3+.6*np.cos(4*np.pi*f/fs)
    two[:,0,1]=.3*np.exp(-2j*np.pi*f/fs)
    two[:,1,0]=two[:,0,1].conj()
    one=two.copy();one[1:-1]*=2
    convention=TDIConvention(2,45,5,"static","static")
    psd=AEPSD(f,{"A":one[:,0,0].real,"E":one[:,1,1].real},convention,
        dict(sidedness="one_sided",scaling="density",noise_model="colored_test",CSD_convention="S_AE=E[tilde_A*conj(tilde_E)]"),one[:,0,1])
    rng=np.random.default_rng(21);x=rng.normal(size=(2,n));y=rng.normal(size=(2,n));t=np.arange(n)*dt
    series=lambda a:AESeries(t,{"A":a[0],"E":a[1]},convention)
    metric=FiniteWindowAEInnerProduct(series(x),psd,(.015,.075),one,rtol=1e-12)
    actual=metric.compare(series(y))
    full=np.concatenate((two,two[-2:0:-1].conj()))
    lag=fs*np.fft.ifft(full,axis=0).real
    covariance=np.block([[toeplitz(lag[:n,a,b],np.r_[lag[0,a,b],lag[-1:-n:-1,a,b]]) for b in range(2)] for a in range(2)])
    frequencies=np.fft.rfftfreq(n,dt);mask=(frequencies>=.015)&(frequencies<=.075)
    modes=[]
    for channel in range(2):
        for frequency in frequencies[mask]:
            for trig in (np.cos,np.sin):
                v=np.zeros((2,n));v[channel]=trig(2*np.pi*frequency*t)*np.sqrt(2/n);modes.append(v.ravel())
    q=np.array(modes);restricted=q@covariance@q.T
    rx,ry=q@x.ravel(),q@y.ravel()
    expected=float(rx@np.linalg.solve(restricted,ry)/np.sqrt(rx@np.linalg.solve(restricted,rx)*(ry@np.linalg.solve(restricted,ry))))
    np.testing.assert_allclose(actual["overlap"],expected,atol=2e-12,rtol=0)
    assert all(row["status"]==0 for row in metric.solve_records)


def test_white_finite_covariance_recovers_standard_psd_metric():
    n,m,dt=64,512,5.
    f=np.fft.rfftfreq(m,dt);one=np.broadcast_to(np.eye(2),(len(f),2,2)).copy()
    one[[0,-1]]*=.5
    convention=TDIConvention(2,45,5,"static","static")
    psd=AEPSD(f,{"A":one[:,0,0],"E":one[:,1,1]},convention,dict(sidedness="one_sided",scaling="density",noise_model="white"))
    rng=np.random.default_rng(2);t=np.arange(n)*dt
    series=lambda:AESeries(t,{ch:rng.normal(size=n) for ch in ("A","E")},convention)
    source,candidate=series(),series()
    finite=FiniteWindowAEInnerProduct(source,psd,(.01,.08),one,rtol=1e-12)
    standard=AEInnerProduct(source,psd,(.01,.08))
    np.testing.assert_allclose(finite.compare(candidate)["overlap"],standard.compare(candidate)["overlap"],atol=1e-13)
    np.testing.assert_allclose(finite.reference_norm,standard.reference_norm,rtol=1e-13)
