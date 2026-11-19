"""Independent response, ordered-rate and late-epoch checks for optional JAX."""
import math
import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp
from xtdi import LINK_LABELS, xyz_to_aet
from xtdi.inference import DetectorResponse
from xtdi.jax_response import JAXDetectorResponse, _dot3, _round64, _fma_bounded
from xtdi._dot_policy import host_fma
from test_inference import fixture, basis, sampled
from test_michelson import reference_data, reference_xyz


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(): yield


def test_rounding_and_geometry_dot_against_host_numpy():
    rng=np.random.default_rng(20261116)
    vectors=rng.normal(size=(10003,3))*500.
    vectors=np.concatenate((vectors,fixture(epoch=0.)['p_sender'].reshape(-1,3)))
    k,*_=basis()
    expected=np.einsum('lti,i->lt',vectors.reshape(1,-1,3),k).ravel()
    actual=np.asarray(jax.jit(_dot3)(vectors,k))
    assert actual.tobytes()==expected.tobytes()
    special=np.array([0.,-0.,np.inf,-np.inf,np.nan,np.finfo(float).max,
                      -np.finfo(float).max,np.nextafter(0.,1.),np.nextafter(0.,-1.)])
    assert np.asarray(jax.jit(_round64)(special)).tobytes()==special.tobytes()


def test_bounded_fma_against_independent_libm():
    rng=np.random.default_rng(20261116)
    a,b,c=rng.normal(size=(3,10003))*500.
    fma=host_fma()
    expected=np.array([fma(float(x),float(y),float(z)) for x,y,z in zip(a,b,c)])
    actual=np.asarray(jax.jit(_fma_bounded)(a,b,c))
    assert actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('generation',[1,2])
@pytest.mark.parametrize('order',[1,25,101])
@pytest.mark.parametrize('explicit_rates',[False,True])
def test_ordered_detector_preparation_and_pytdi(generation,order,explicit_rates):
    g=fixture(size=513);t=g['times']-g['times'][0]
    g['length']+=np.stack([.02*np.sin(t/500+i) for i in range(6)])
    rates=np.stack([.02/500*np.cos(t/500+i) for i in range(6)]) if explicit_rates else None
    backend=JAXDetectorResponse.prepare(g,generation=generation,measurement_order=order,delay_derivatives=rates)
    cpu=DetectorResponse.prepare(g,generation=generation,measurement_order=order,delay_derivatives=rates)
    # Order and all retarded additive rates are compared, including endpoints.
    parameters=cpu.tdi.to_jax().parameters
    for actual,expected in zip(backend.parameters,parameters,strict=True):
        np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),rtol=5e-14,atol=2e-14)
    from xtdi.jax_backend import _compiled
    links=np.random.default_rng(33).normal(size=(6,513))
    actual=np.asarray(_compiled(generation,order,True)(jnp.asarray(links),backend.parameters))
    delays=dict(zip(LINK_LABELS,g['length']))
    oracle=reference_xyz(reference_data(g['times'],delays,None if rates is None else dict(zip(LINK_LABELS,rates))),
                         dict(zip(LINK_LABELS,links)),generation,order,7)
    assert np.max(abs(actual-oracle))/np.max(abs(oracle))<1e-12


@pytest.mark.parametrize('epoch',[0.,.125*31557600,3e7])
@pytest.mark.parametrize('sky',[(1.2,.3),(.01,1.5),(4.9,-1.4),(3.,0.)])
@pytest.mark.parametrize('retarded',[False,True])
def test_actual_physical_times_and_links(epoch,sky,retarded):
    g=fixture(epoch=epoch,size=257,retarded=retarded)
    cpu=DetectorResponse.prepare(g);backend=JAXDetectorResponse.prepare(g)
    k,p,c=basis(*sky);data=sampled(epoch=epoch,origin_offset=.2)
    projection,valid=backend.project_array(k,p,c,polarization_angle=.37)
    assert bool(valid)
    expected=cpu.sky(k,p,c,polarization_angle=.37)
    assert np.asarray(projection[0]).ravel().tobytes()==expected.query_times.tobytes()
    links,valid=backend.sampled_links_array(data.values,data.origin_s,data.dt_s,k,p,c,polarization_angle=.37)
    oracle=cpu.sampled_links(data,k,p,c,polarization_angle=.37)
    assert bool(valid)
    assert np.linalg.norm(np.asarray(links)-oracle)/np.linalg.norm(oracle)<1e-12
    output=np.asarray(backend.sampled_response(data,k,p,c,polarization_angle=.37))
    expected=cpu.apply_links(oracle)
    assert np.linalg.norm(output-expected)/np.linalg.norm(expected)<1e-12
    np.testing.assert_allclose(np.asarray(backend.xyz_to_aet(output)),xyz_to_aet(output),rtol=1e-12,atol=1e-15)


@pytest.mark.parametrize('order',[1,7,39,101])
@pytest.mark.parametrize('unit',['frequency','phase'])
def test_source_batches_changing_skies_jit_and_jvp(order,unit):
    cpu=DetectorResponse.prepare(fixture());backend=JAXDetectorResponse.from_detector(cpu)
    data=sampled(batch=3);k,p,c=basis()
    def apply(v):return backend.sampled_response_array(v,data.origin_s,data.dt_s,k,p,c,order=order,unit=unit)[0]
    values=jnp.asarray(data.values)
    actual=jax.jit(apply)(values)
    expected=cpu.apply_links(cpu.sampled_links(data,k,p,c,order=order),unit=unit)
    assert np.linalg.norm(np.asarray(actual)-expected)/np.linalg.norm(expected)<1e-12
    _,tangent=jax.jvp(apply,(values,),(values*.3,))
    np.testing.assert_allclose(np.asarray(tangent),np.asarray(actual)*.3,rtol=1e-11,atol=1e-13)
    skies=[basis(1.2,.3),basis(3.,-.5),basis(5.,.9)]
    directions,plus,cross=[jnp.asarray(np.stack(v)) for v in zip(*skies)]
    varying=jax.jit(jax.vmap(lambda v,k,p,c:backend.sampled_response_array(
        v,data.origin_s,data.dt_s,k,p,c,order=order,unit=unit)))(values,directions,plus,cross)
    assert np.all(varying[1])
    from xtdi.adapters.sampled import SampledPolarizations
    oracle=np.stack([cpu.apply_links(cpu.sampled_links(SampledPolarizations(data.origin_s,data.dt_s,data.values[i]),
        *skies[i],order=order),unit=unit) for i in range(3)])
    assert np.linalg.norm(np.asarray(varying[0])-oracle)/np.linalg.norm(oracle)<1e-12


@pytest.mark.parametrize('generation',[1,2])
def test_equal_arm_transfer_function_without_pytdi(generation):
    from xtdi.jax_response import _tdi
    g=fixture(epoch=0.,size=1024);g['length'][:]=8.3
    plan=JAXDetectorResponse.prepare(g,generation=generation,measurement_order=25)
    t=g['times'];omega=2*np.pi*.003;delay=np.exp(-1j*omega*8.3)
    rng=np.random.default_rng(99);amplitude=rng.normal(size=6)+1j*rng.normal(size=6)
    links=np.real(amplitude[:,None]*np.exp(1j*omega*t))
    actual=np.asarray(jax.jit(lambda x:_tdi(x,plan.parameters,generation,25,True))(links))
    expected=[]
    for a,b,c,d in ((5,0,2,3),(4,1,0,5),(3,2,1,4)):
        arms=amplitude[a]+delay*amplitude[b]-amplitude[c]-delay*amplitude[d]
        transfer=(delay**2-1.)*(1.-delay**4 if generation==2 else 1.)
        expected.append(np.real(arms*transfer*np.exp(1j*omega*t)))
    keep=slice(150,-150);expected=np.asarray(expected)
    assert np.linalg.norm(actual[:,keep]-expected[:,keep])/np.linalg.norm(expected[:,keep])<1e-11


def test_invalid_support_and_inputs_are_rejected():
    backend=JAXDetectorResponse.prepare(fixture());data=sampled();k,p,c=basis()
    for origin,dt,direction in [(data.origin_s+1e5,data.dt_s,k),(data.origin_s,0.,k),
                               (np.nan,data.dt_s,k),(data.origin_s,data.dt_s,k*2)]:
        result,valid=jax.jit(lambda:backend.sampled_response_array(data.values,origin,dt,direction,p,c))()
        assert not bool(valid) and np.all(np.isnan(result))
    from xtdi.adapters.sampled import SampledPolarizations
    with pytest.raises(ValueError,match='support'):
        backend.sampled_response(SampledPolarizations(data.origin_s+1e5,data.dt_s,data.values),k,p,c)
    with pytest.raises(ValueError,match='float64'):
        backend.sampled_response_array(data.values.astype('f4'),data.origin_s,data.dt_s,k,p,c)
    with pytest.raises(ValueError):backend.sampled_response(data,k,p,c,order=2)
    with jax.enable_x64(False):
        with pytest.raises(ValueError,match='x64'):JAXDetectorResponse.prepare(fixture())


def test_preparation_ownership_and_explicit_alignment_failure():
    g=fixture();backend=JAXDetectorResponse.prepare(g);data=sampled();k,p,c=basis()
    before=np.asarray(backend.project_array(k,p,c)[0][0]).copy()
    g['p_sender'][:]+=100.;g['length'][:]+=1.
    np.testing.assert_array_equal(np.asarray(backend.project_array(k,p,c)[0][0]),before)
    k=np.asarray(backend.geometry[1][0,0]);u=np.cross(k,[0.,0.,1.]);u/=np.linalg.norm(u);v=np.cross(k,u)
    _,valid=backend.sampled_response_array(data.values,data.origin_s,data.dt_s,k,np.outer(u,u)-np.outer(v,v),
                                          np.outer(u,v)+np.outer(v,u))
    assert not bool(valid)


@pytest.mark.parametrize('budget',[0,8*1024**2])
@pytest.mark.parametrize('schedule',['staged','fused_unrolled'])
def test_streamed_blocks_cache_and_changing_sky(budget,schedule):
    from xtdi.jax_response import JAXStreamedDetectorResponse
    from xtdi.inference import StreamedDetectorResponse
    epoch=3e7;times=epoch+np.arange(769)*2.5
    def provider(t):return fixture(epoch=epoch,times=t,retarded=True)
    kwargs=dict(block_samples=256,cache_bytes=budget,padding_order=121)
    cpu=StreamedDetectorResponse(times,provider,**kwargs)
    device=JAXStreamedDetectorResponse(times,provider,execution_schedule=schedule,**kwargs)
    data=sampled(epoch=epoch,batch=2,origin_offset=.2)
    for sky,angle in [((1.2,.3),0.),((4.9,-1.4),.37)]:
        k,p,c=basis(*sky)
        expected=cpu.sampled_response(data,k,p,c,polarization_angle=angle)
        actual=device.sampled_response(data,k,p,c,polarization_angle=angle)
        parts=[]
        for _,block,valid in device.iter_arrays(data,k,p,c,polarization_angle=angle,max_inflight=3):
            assert bool(valid)
            parts.append(np.asarray(block))
        streamed=np.concatenate(parts,axis=-1)
        assert np.linalg.norm(streamed-expected)/np.linalg.norm(expected)<1e-10
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<1e-10
        assert device.cached_bytes<=budget
        assert device.last_stages['returned_samples']==times.size
    if budget:assert device.last_stages['cache_hits']==4
    else:assert device.last_stages['cache_misses']==4
    with pytest.raises(ValueError,match='float64'):
        list(device.iter_arrays(data,k.astype('f4'),p,c))
    with pytest.raises(ValueError,match='memory'):
        device.sampled_response(data,k,p,c,max_output_bytes=1)
    device.clear_cache();assert device.cached_bytes==0


def test_rounding_identity_on_raw_ieee_patterns():
    rng=np.random.default_rng(161116)
    bits=rng.integers(0,2**64,size=1000003,dtype=np.uint64)
    # Includes random normal/subnormal, signed zero, NaN/Inf and extreme finite
    # values. Bit equality also checks payloads rather than float comparisons.
    expected=bits.view(np.float64)
    actual=np.asarray(jax.jit(_round64)(expected))
    assert actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('schedule',['staged','fused','fused_unrolled'])
def test_execution_schedule_outer_jit_and_reverse_derivative(schedule):
    cpu=DetectorResponse.prepare(fixture())
    backend=JAXDetectorResponse.from_detector(cpu,execution_schedule=schedule)
    data=sampled();k,p,c=basis(4.9,-1.4)
    apply=lambda v:backend.sampled_response_array(v,data.origin_s,data.dt_s,k,p,c)[0]
    direct=np.asarray(apply(data.values));outer=np.asarray(jax.jit(apply)(data.values))
    expected=cpu.sampled_response(data,k,p,c)
    for out in [direct,outer]:
        assert np.linalg.norm(out-expected)/np.linalg.norm(expected)<1e-12
    values=jnp.asarray(data.values)
    energy=lambda amplitude:jnp.sum(apply(values*amplitude)**2)
    derivative=jax.jit(jax.grad(energy))(jnp.float64(1.))
    np.testing.assert_allclose(np.asarray(derivative),2*np.sum(direct**2),rtol=1e-11)
    with pytest.raises(ValueError,match='execution_schedule'):
        JAXDetectorResponse.from_detector(cpu,execution_schedule='unknown')
    with pytest.raises(ValueError,match='execution_schedule'):
        JAXDetectorResponse.prepare(fixture(),execution_schedule='unknown')
