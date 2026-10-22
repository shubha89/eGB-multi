import numpy as np
from scipy.interpolate import CubicSpline

from egb_multi.spline_fourier import CubicEnvelopeFourier


def test_endpoint_formula_matches_sampled_polynomial_carrier_at_every_bin():
    # A global cubic has no internal derivative jumps: the closed endpoint
    # correction must reproduce the sampled DFT, including fractional carrier
    # bins, endpoints, both frequency signs and non-dividing cubic knots.
    n,dt=768,5.;duration=n*dt;coarse=np.arange(-2,19)*duration/16
    t=np.arange(n)*dt;f=np.fft.fftfreq(n,dt);bins=np.rint(f*duration).astype(int)
    def envelope(q):
        u=q/duration
        return 1+.4*u-.7*u*u+.3j*u*u*u
    for carrier in (0.,17.713/duration,.075):
        spline=CubicSpline(coarse,envelope(coarse))
        epoch=.437*duration
        model=CubicEnvelopeFourier(coarse,duration,dt,bins,carrier_hz=carrier,epoch_s=epoch)
        actual=model.transform(spline)
        expected=dt*np.fft.fft(envelope(t)*np.exp(2j*np.pi*carrier*(t-epoch)))
        np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=2e-11)


def test_noninteger_knot_sampling_refines_to_actual_multichannel_fft():
    n,dt=4096,7.;duration=n*dt;t=np.arange(n)*dt;bins=np.arange(20,1200)
    expected=[];errors=[]
    for intervals in (127,255,511):
        coarse=np.arange(-2,intervals+3)*duration/intervals
        u=coarse/duration
        values=np.column_stack([np.exp(1.7j*np.sin(2*np.pi*u)),.7*np.exp(-.4j*np.cos(2*np.pi*u))])
        spline=CubicSpline(coarse,values,axis=0);carrier=231.713/duration
        model=CubicEnvelopeFourier(coarse,duration,dt,bins,carrier_hz=carrier)
        actual=model.transform(spline)
        sampled=spline(t)*np.exp(2j*np.pi*carrier*t)[:,None]
        reference=dt*np.fft.fft(sampled,axis=0)[bins]
        errors.append(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
    assert errors[-1]<2e-10
    assert errors[-1]<errors[0]
