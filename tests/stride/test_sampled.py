"""Source interpolation checks independent of waveform/TDI agreement."""
import numpy as np
import pytest

from xtdi.adapters.sampled import SampledPolarizations, UniformQueryPlan


def test_polynomial_exactness_unordered_queries_and_batches():
    origin, dt, n = 12000., .625, 300
    time = origin+np.arange(n)*dt
    coordinate = (time-origin)/100
    values = np.stack((coordinate**3-.2*coordinate, 2-coordinate**2))
    query = origin+dt*np.array([[180.25, 60.6, 80.], [90.01, 130.9, 60.6]])
    plan = UniformQueryPlan.prepare(origin, dt, n, query, order=7)
    expected_x = (query-origin)/100
    expected = np.stack((expected_x**3-.2*expected_x, 2-expected_x**2))
    actual = plan.apply(values)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)
    batch = plan.apply(np.stack((values, 2*values, -values)))
    np.testing.assert_allclose(batch, np.stack((expected, 2*expected, -expected)), atol=1e-13)
    # Q may be smaller or larger than the source record. No implicit Q=N.
    many = np.tile(query.ravel(), 80)
    assert UniformQueryPlan.prepare(origin,dt,n,many,order=7).apply(values).shape == (2,480)


@pytest.mark.parametrize('dt', [2.5, 1.25, .625])
@pytest.mark.parametrize('order', [25,39,101])
def test_analytic_tone_and_preservation(dt, order):
    n = 2048
    origin = 2.76129e7
    t = np.arange(n)*dt
    values = np.stack((np.cos(2*np.pi*.01*t), .7*np.sin(2*np.pi*.023*t+.3)))
    source = SampledPolarizations.from_arrays(*values,origin_s=origin,dt_s=dt,order=order)
    q = origin+dt*np.random.default_rng(47).uniform(100,n-101,800)
    tau = q-origin
    expected = np.stack((np.cos(2*np.pi*.01*tau), .7*np.sin(2*np.pi*.023*tau+.3)))
    actual = source(q)
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 2e-12
    plan = source.prepare_queries(q)
    np.testing.assert_array_equal(source.evaluate(plan),actual)
    np.testing.assert_array_equal(source.values,values)
    assert not source.values.flags.writeable
    assert plan.nbytes < 33*q.size+order*8


def test_independent_lagrange_product_and_grid_epoch():
    values=np.random.default_rng(191).normal(size=(2,400))
    queries=np.array([90.25,151.9,188.001])
    plan=UniformQueryPlan.prepare(0.,1.,400,queries,order=7)
    expected=[]
    for q in queries:
        nodes=np.floor(q).astype(int)+np.arange(-3,5)
        weight=np.array([np.prod([(q-b)/(a-b) for b in nodes if a!=b]) for a in nodes])
        expected.append(values[:,nodes]@weight)
    np.testing.assert_allclose(plan.apply(values),np.array(expected).T,rtol=3e-14,atol=3e-14)
    shifted=UniformQueryPlan.prepare(1000.,1.,400,1000+queries,order=7)
    np.testing.assert_allclose(shifted.apply(values),plan.apply(values),atol=2e-13)


def test_rejects_extrapolation_nonphysical_and_misaligned_inputs():
    values=np.arange(200.)
    source=SampledPolarizations.from_arrays(values,-values,origin_s=10.,dt_s=1.,order=25)
    for query in [[10.], [209.], [np.nan], []]:
        with pytest.raises(ValueError):source(query)
    for dt in [0.,-1.,np.nan]:
        with pytest.raises(ValueError):SampledPolarizations.from_arrays(values,values,origin_s=0.,dt_s=dt)
    with pytest.raises(ValueError):SampledPolarizations.from_arrays(values.astype(complex),values,origin_s=0.,dt_s=1.)
    plan=source.prepare_queries(np.array([60.25]))
    other=SampledPolarizations.from_arrays(values,values,origin_s=0.,dt_s=1.,order=25)
    with pytest.raises(ValueError):other.evaluate(plan)
    with pytest.raises(ValueError):plan.apply(np.zeros((2,201)))
    with pytest.raises(ValueError):UniformQueryPlan.prepare(0.,1.,True,[2.],order=1)


def test_fraction_precision_for_year_long_source_grid():
    origin, dt = -5000., 1.25
    query = np.array([3.0e7+.1, 3.0e7+.999, 2.0e7+1e-7])
    plan = UniformQueryPlan.prepare(origin,dt,26000000,query,order=101)
    physical_grid = origin+plan.centers*dt
    expected = (query-physical_grid)/dt
    np.testing.assert_array_equal(plan.fractions,expected)
    np.testing.assert_array_equal(physical_grid+plan.fractions*dt,query)


@pytest.mark.parametrize('order', [1, 7, 39, 101])
def test_fraction_rounding_near_an_exact_source_sample(order):
    # A physical query can straddle zero even though normalization by a large
    # source origin rounds to an integer. Rebase a fraction rounded to one;
    # otherwise preparing the stencil wraps the fraction without its center.
    origin, dt = -32., .5
    time = origin+np.arange(128)*dt
    query = np.array([-1e-17, 0., 1e-17])
    source = SampledPolarizations.from_arrays(time, np.zeros_like(time),
        origin_s=origin, dt_s=dt, order=order)
    plan = source.prepare_queries(query)
    assert np.all((plan.fractions >= 0) & (plan.fractions < 1))
    np.testing.assert_array_equal(plan.centers, [64, 64, 64])
    np.testing.assert_allclose(source.evaluate(plan)[0], query, rtol=0, atol=2e-16)
