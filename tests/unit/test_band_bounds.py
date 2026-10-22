"""Independent Euclidean two-band controls for the Cauchy-Schwarz bound."""

import numpy as np
import pytest

from egb_multi.band_bounds import expanded_band_overlap_upper_bound, expanded_band_maximum_increase_bound


def test_bound_matches_independent_angle_maximization():
    # s=(sqrt(1-eta),0,sqrt(eta)); in-band templates make a fixed angle
    # arccos(U) with s_in. Scan their arbitrary in/out amplitude ratio.
    for upper in (.05,.4,.9,1.):
        for fraction in (0.,1e-8,.01,.6,1.):
            angle=np.linspace(0,np.pi/2,100001)
            overlap=np.sqrt(1-fraction)*upper*np.cos(angle)+np.sqrt(fraction)*np.sin(angle)
            bound=expanded_band_overlap_upper_bound(upper,fraction)
            assert overlap.max() <= bound+2e-15
            assert bound-overlap.max() < 4e-11


def test_bound_controls_an_independent_random_template_family():
    rng=np.random.default_rng(264615)
    signal=rng.normal(size=9);signal/=np.linalg.norm(signal)
    templates=rng.normal(size=(1000,9))
    in_band=signal[:5];outside=signal[5:]
    overlaps_in=np.abs(templates[:,:5]@in_band)/(np.linalg.norm(templates[:,:5],axis=1)*np.linalg.norm(in_band))
    upper=overlaps_in.max()
    actual=np.abs(templates@signal)/np.linalg.norm(templates,axis=1)
    assert actual.max() <= expanded_band_overlap_upper_bound(upper,float(outside@outside))+2e-15


def test_invalid_bounds_are_rejected():
    for upper,fraction in ((-.1,.5),(1.1,.5),(.5,-.1),(.5,1.1),(np.nan,.5),(.5,np.inf)):
        with pytest.raises(ValueError):expanded_band_overlap_upper_bound(upper,fraction)


def test_maximum_increase_is_bounded_using_only_a_candidate_lower_bound():
    rng=np.random.default_rng(24546)
    for _ in range(20):
        signal=rng.normal(size=9);signal/=np.linalg.norm(signal)
        templates=rng.normal(size=(200,9))
        narrow=np.abs(templates[:,:5]@signal[:5])/(np.linalg.norm(templates[:,:5],axis=1)*np.linalg.norm(signal[:5]))
        broad=np.abs(templates@signal)/np.linalg.norm(templates,axis=1)
        lower=narrow[0]  # Intentionally NOT the maximum of this finite family.
        fraction=float(signal[5:]@signal[5:])
        assert broad.max()-narrow.max() <= expanded_band_maximum_increase_bound(lower,fraction)+2e-15


def test_single_template_family_saturates_the_increase_bound():
    for lower,fraction in ((.05,.1),(.4,.01),(.9,.6)):
        norm=np.sqrt((1-fraction)*lower**2+fraction)
        a=np.sqrt(1-fraction)*lower/norm;b=np.sqrt(fraction)/norm
        signal=np.array([np.sqrt(1-fraction),0.,np.sqrt(fraction)])
        template=np.array([a*lower,a*np.sqrt(1-lower**2),b])
        narrow=np.dot(signal[:2],template[:2])/(np.linalg.norm(signal[:2])*np.linalg.norm(template[:2]))
        broad=np.dot(signal,template)/np.linalg.norm(template)
        np.testing.assert_allclose(broad-narrow,expanded_band_maximum_increase_bound(lower,fraction),rtol=2e-13,atol=2e-15)


def test_increase_bound_retains_tiny_nonzero_power_and_rejects_invalid_inputs():
    from decimal import Decimal,localcontext
    with localcontext() as context:
        context.prec=80
        lower=Decimal('.9');eta=Decimal('1e-25')
        exact=((1-eta)*lower**2+eta).sqrt()-lower
    actual=expanded_band_maximum_increase_bound(.9,1e-25)
    assert actual > 0
    np.testing.assert_allclose(actual,float(exact),rtol=3e-15,atol=0)
    assert expanded_band_maximum_increase_bound(0,0)==0
    assert expanded_band_maximum_increase_bound(1,.4)==0
    for lower,eta in ((-.1,.5),(1.1,.5),(.5,-.1),(.5,1.1),(np.nan,.5),(.5,np.inf)):
        with pytest.raises(ValueError):expanded_band_maximum_increase_bound(lower,eta)
