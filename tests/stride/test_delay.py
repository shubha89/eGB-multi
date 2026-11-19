"""Differential and independent analytic checks of the delay primitive."""

import numpy as np
import pytest
from xtdi import FractionalDelay, timeshift

pytdi = pytest.importorskip("pytdi.dsp")


@pytest.mark.parametrize("order", [3, 7, 45, 81, 101, 121])
@pytest.mark.parametrize("kind", ["zero", "integer", "scalar", "varying", "extreme"])
def test_reference_including_edges(order, kind):
    n = 401
    data = np.random.default_rng(177).normal(size=n)
    shift = {"zero": 0., "integer": -3., "scalar": -3.25,
             "varying": -3. + .9*np.sin(np.arange(n)*.07),
             "extreme": np.linspace(-2*n, 2*n, n)}[kind]
    expected = pytdi.timeshift(data, shift, order)
    actual = timeshift(data, shift, order)
    assert np.max(np.abs(actual-expected)) / np.max(np.abs(data)) < 1e-12


@pytest.mark.parametrize("fraction", [0., 1e-12, .25, .5, 1-1e-12])
def test_near_integer_fraction_and_batch(fraction):
    data = np.random.default_rng(812).normal(size=(4, 512))
    shifts = np.full(512, -4+fraction)
    stencil = FractionalDelay.prepare(shifts, 512, 101)
    actual = stencil.apply(data)
    for i, values in enumerate(data):
        np.testing.assert_allclose(actual[i], pytdi.timeshift(values, shifts, 101), atol=1e-13, rtol=1e-12)
    assert stencil.nbytes < 40*512


def test_analytic_integer_and_fractional_tone():
    data = np.arange(512.)
    actual = timeshift(data, -3., 7)
    np.testing.assert_array_equal(actual[3:], data[:-3])
    np.testing.assert_array_equal(actual[:3], np.zeros(3))
    t = np.arange(1024.)
    # Low-frequency tone has a known phase delay independent of pyTDI.
    tone = np.cos(.03*t)
    expected = np.cos(.03*(t-3.25))
    assert np.max(np.abs(timeshift(tone, -3.25, 45)[80:-80]-expected[80:-80])) < 2e-13


def test_reuse_validation_and_singleton():
    plan = FractionalDelay.prepare(-3.2, 128, 7)
    x = np.arange(128.)
    original = x.copy()
    first = plan.apply(x)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_allclose(plan.apply(2*x), 2*first)
    assert timeshift(5., np.arange(9), 3) == 5.
    for invalid in (0, 4, 3.5, True):
        with pytest.raises(ValueError):
            FractionalDelay.prepare(-1, 128, invalid)
    with pytest.raises(ValueError):
        plan.apply(np.full(128, np.nan))
    with pytest.raises(ValueError):
        plan.apply(x.astype(complex))
    with pytest.raises(ValueError):
        plan.apply(np.zeros(127))
