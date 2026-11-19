import numpy as np
import pytest
from xtdi import LINK_LABELS,prepare_michelson

def plan_check(prepare,generation,measurement,delay,explicit_rates):
    rng=np.random.default_rng(782901);size=257;dt=1.25
    times=3.0e7+np.arange(size)*dt
    # Include zeros, integer and fractional delays, changing shifts, and
    # boundaries where the reference's out-of-record samples are zero.
    delays=np.stack([np.zeros(size),np.full(size,2.5),np.full(size,8.3),
        8.2+1e-3*np.sin(np.arange(size)/17.),
        8.4+1e-4*np.arange(size),np.full(size,1000.)])
    kwargs=dict(generation=generation,measurement_order=measurement,delay_order=delay)
    if explicit_rates:kwargs['delay_derivatives']=rng.normal(0,1e-8,size=delays.shape)
    reference=prepare_michelson(times,delays,**kwargs);candidate=prepare(times,delays,**kwargs)
    assert candidate.fingerprint==reference.fingerprint
    assert [op.chain for op in candidate.operators]==[op.chain for op in reference.operators]
    for a,b in zip(candidate.operators,reference.operators,strict=True):
        np.testing.assert_allclose(a.shift_s,b.shift_s,rtol=0,atol=2e-13)
        np.testing.assert_allclose(a.rate,b.rate,rtol=0,atol=2e-16)
        np.testing.assert_array_equal(a.stencil.centers,b.stencil.centers)
        np.testing.assert_allclose(a.stencil.fractions,b.stencil.fractions,rtol=0,atol=2e-13)
    values=rng.normal(size=(4,6,size))
    for unit in ['phase','frequency']:
        actual,stages=candidate.apply(values,unit=unit,return_stages=True)
        expected,reference_stages=reference.apply(values,unit=unit,return_stages=True)
        np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
        for name in stages:
            for x,y in zip(stages[name],reference_stages[name],strict=True):
                np.testing.assert_allclose(x,y,rtol=2e-12,atol=2e-12)

def validation_check(prepare):
    times=np.arange(64)*2.5;delays={k:np.full(64,8.3) for k in LINK_LABELS}
    for bad in [0,3,True]:
        with pytest.raises(ValueError):prepare(times,delays,generation=bad)
    with pytest.raises(ValueError):prepare(times,delays,measurement_order=2)
    with pytest.raises(ValueError):prepare(times,delays,delay_order=2)
    with pytest.raises(ValueError):prepare(times,np.full((6,64),-1.))
    with pytest.raises(ValueError):prepare(times,np.full((6,64),np.nan))
    with pytest.raises(ValueError):prepare(times,np.ones((6,64),dtype=complex))
    with pytest.raises(ValueError):prepare(times,delays,delay_derivatives=np.zeros((2,6,64)))
    with pytest.raises(ValueError):prepare(times+np.sin(times)*.1,delays)

import numpy as np
import pytest

from xtdi.optimized._stencils import prepare_delay,prepare_queries
from xtdi.optimized import prepare_michelson as prepare_compiled
from xtdi import LINK_LABELS
from xtdi.delay import FractionalDelay
from xtdi.adapters.sampled import UniformQueryPlan



@pytest.mark.parametrize('order',[1,7,25,39,101])
@pytest.mark.parametrize('mode',['zero','integer','fractional','varying','distant'])
def test_fused_delay_stencil(order,mode):
    size=257;rng=np.random.default_rng(7801)
    shifts={'zero':np.zeros(size),'integer':np.full(size,-4.),
        'fractional':np.full(size,-4.3),'varying':rng.uniform(-13,3,size),
        'distant':np.where(np.arange(size)%2,1e200,-1e200)}[mode]
    reference=FractionalDelay.prepare(shifts,size,order);candidate=prepare_delay(shifts,size,order)
    np.testing.assert_array_equal(candidate.centers,reference.centers)
    np.testing.assert_array_equal(candidate.fractions,reference.fractions)
    np.testing.assert_allclose(candidate.scales,reference.scales,rtol=3e-15,atol=0)
    values=rng.normal(size=(3,2,size))
    np.testing.assert_allclose(candidate.apply(values),reference.apply(values),rtol=5e-14,atol=5e-14)
    assert candidate.identity==reference.identity
    assert all(not a.flags.writeable for a in [candidate.centers,candidate.fractions,candidate.scales])


@pytest.mark.parametrize('order',[7,39,101])
@pytest.mark.parametrize('origin',[0.,-5000.,3.0e7])
@pytest.mark.parametrize('dt',[.15625,.625,2.5])
def test_fused_source_queries(order,origin,dt):
    rng=np.random.default_rng(99081);size=2048
    q=origin+dt*rng.uniform(100,size-101,(3,5))
    reference=UniformQueryPlan.prepare(origin,dt,size,q,order=order)
    candidate=prepare_queries(origin,dt,size,q,order=order)
    np.testing.assert_array_equal(candidate.centers,reference.centers)
    np.testing.assert_array_equal(candidate.fractions,reference.fractions)
    values=rng.normal(size=(4,2,size))
    np.testing.assert_allclose(candidate.apply(values),reference.apply(values),rtol=5e-14,atol=5e-14)


@pytest.mark.parametrize('order',[1,7,39,101])
def test_fused_queries_near_exact_sample_and_physical_support(order):
    grid=-32.+np.arange(128)*.5;q=np.array([-1e-17,0.,1e-17])
    plan=prepare_queries(-32.,.5,128,q,order=order)
    np.testing.assert_array_equal(plan.centers,[64,64,64])
    assert np.all((plan.fractions>=0)&(plan.fractions<1))
    np.testing.assert_allclose(plan.apply(grid),q,rtol=0,atol=2e-16)
    with pytest.raises(ValueError):prepare_queries(-32.,.5,128,[-33.],order=order)
    with pytest.raises(ValueError):prepare_queries(0.,0.,128,[25.],order=order)
    with pytest.raises(ValueError):prepare_queries(0.,1.,True,[25.],order=order)


@pytest.mark.parametrize('generation',[1,2])
@pytest.mark.parametrize('measurement,delay',[(1,1),(25,7),(101,15)])
@pytest.mark.parametrize('explicit_rates',[False,True])
def test_fused_plan_preserves_states_and_stage_graph(generation,measurement,delay,explicit_rates):
    plan_check(prepare_compiled,generation,measurement,delay,explicit_rates)


def test_fused_plan_validation():validation_check(prepare_compiled)
