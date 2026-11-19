"""Changing-sky correctness, epoch precision and detector/source invalidation."""
import numpy as np
import pytest
from numba import set_num_threads, get_num_threads
from xtdi import LINK_LABELS
from xtdi.inference import DetectorResponse
from xtdi.optimized import LinkProjection, prepare_queries, evaluate_samples
from xtdi.adapters.sampled import SampledPolarizations


def assert_identical(actual, expected, **kwargs):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes(), kwargs.get('err_msg', 'different array bytes')


def fixture(epoch=3e7, size=257, retarded=False, times=None):
    times=epoch+np.arange(size)*2.5 if times is None else np.asarray(times)
    local=times-epoch
    center=np.stack([500*np.cos(local/31557600),500*np.sin(local/31557600),.1*np.sin(local/1e6)],axis=-1)
    vertices=np.array([[0.,0.,0.],[8.3,0.,.1],[4.,7.1,-.2]])
    senders=[];receivers=[];directions=[];lengths=[]
    for label in LINK_LABELS:
        r,e=int(label[0])-1,int(label[1])-1
        receiver=center+vertices[r]
        sender=center+vertices[e]
        if retarded:sender=sender-np.array([0.,500*8.3/31557600,0.])
        delta=receiver-sender;length=np.linalg.norm(delta,axis=-1)
        senders.append(sender);receivers.append(receiver)
        directions.append(delta/length[:,None]);lengths.append(length)
    return dict(times=times,n=np.array(directions),length=np.array(lengths),p_sender=np.array(senders),p_receiver=np.array(receivers))


def basis(longitude=1.2,latitude=.3):
    k=-np.array([np.cos(latitude)*np.cos(longitude),np.cos(latitude)*np.sin(longitude),np.sin(latitude)])
    u=np.array([np.sin(longitude),-np.cos(longitude),0.])
    v=np.array([-np.sin(latitude)*np.cos(longitude),-np.sin(latitude)*np.sin(longitude),np.cos(latitude)])
    return k,np.outer(u,u)-np.outer(v,v),np.outer(u,v)+np.outer(v,u)


def sampled(epoch=3e7,batch=1,origin_offset=0.):
    origin=epoch-5000.+origin_offset;dt=.625
    t=origin+np.arange(16384)*dt
    values=np.stack([np.cos(.02*(t-epoch)),.7*np.sin(.031*(t-epoch)+.3)])
    if batch>1:values=np.arange(1,batch+1)[:,None,None]*values
    values=np.ascontiguousarray(values);values.flags.writeable=False
    return SampledPolarizations(origin,dt,values,101)


@pytest.mark.parametrize('epoch',[0.,.125*31557600,3.0e7])
@pytest.mark.parametrize('sky',[(1.2,.3),(.01,1.5),(4.9,-1.4),(3.0,0.)])
@pytest.mark.parametrize('retarded',[False,True])
def test_prepared_sky_preserves_physical_queries_and_projections(epoch,sky,retarded):
    g=fixture(epoch,retarded=retarded);detector=DetectorResponse.prepare(g)
    k,p,c=basis(*sky)
    reference=LinkProjection.prepare(k,p,c,g,polarization_angle=.37)
    fingerprint=detector.tdi.fingerprint
    actual=detector.sky(k,p,c,polarization_angle=.37)
    for name in ['query_times','sender_times','receiver_times','plus_projection','cross_projection','denominator']:
        assert_identical(getattr(actual,name),getattr(reference,name),err_msg=name)
    assert detector.tdi.fingerprint==fingerprint
    assert all(not a.flags.writeable for a in [detector.times,detector.directions,detector.lengths,detector.senders,detector.receivers])


@pytest.mark.parametrize('order',[1,7,39,101])
@pytest.mark.parametrize('batch',[1,4])
@pytest.mark.parametrize('parallel',[False,True])
def test_fused_sampled_links_and_staged_tdi(order,batch,parallel):
    g=fixture();detector=DetectorResponse.prepare(g,measurement_order=25)
    k,p,c=basis(4.9,-1.4);data=sampled(batch=batch)
    projection=LinkProjection.prepare(k,p,c,g,polarization_angle=.37)
    queries=prepare_queries(data.origin_s,data.dt_s,data.values.shape[-1],projection.query_times,order=order)
    expected=projection.apply(lambda q:evaluate_samples(data,queries))
    old_threads=get_num_threads();set_num_threads(2)
    try:actual=detector.sampled_links(data,k,p,c,order=order,polarization_angle=.37,parallel=parallel)
    finally:set_num_threads(old_threads)
    assert_identical(actual,expected)
    assert_identical(detector.apply_links(actual),detector.apply_links(expected))


def test_new_source_grid_and_geometry_ownership():
    g=fixture();detector=DetectorResponse.prepare(g);k,p,c=basis()
    for offset in [0.,.2,-.1]:
        data=sampled(origin_offset=offset)
        projection=LinkProjection.prepare(k,p,c,g)
        queries=prepare_queries(data.origin_s,data.dt_s,data.values.shape[-1],projection.query_times,order=39)
        expected=projection.apply(lambda q:evaluate_samples(data,queries))
        assert_identical(detector.sampled_links(data,k,p,c),expected)
    before=detector.sky(k,p,c).query_times.copy()
    g['p_sender'][:]+=10.;g['length'][:]+=1.
    assert_identical(detector.sky(k,p,c).query_times,before)


def test_invalid_inputs_support_output_and_alignment():
    g=fixture();detector=DetectorResponse.prepare(g);k,p,c=basis();data=sampled()
    with pytest.raises(ValueError):detector.sky(k*2,p,c)
    with pytest.raises(ValueError):detector.sampled_links(data,k,p,c,order=2)
    mutable=data.values.copy()
    with pytest.raises(ValueError):detector.sampled_links(SampledPolarizations(data.origin_s,data.dt_s,mutable),k,p,c)
    bad=SampledPolarizations(3e7+1e5,data.dt_s,data.values)
    for parallel in [False,True]:
        with pytest.raises(ValueError):detector.sampled_links(bad,k,p,c,parallel=parallel)
    output=np.empty((6,detector.size));assert detector.sampled_links(data,k,p,c,output=output) is output
    with pytest.raises(ValueError):detector.sampled_links(data,k,p,c,output=output.astype(np.float32))
    # Exact alignment is an existing explicit limitation, not clipped away.
    aligned=detector.directions[0,0].copy()
    u=np.cross(aligned,[0.,0.,1.]);u/=np.linalg.norm(u);v=np.cross(aligned,u)
    with pytest.raises(ValueError):detector.sampled_links(data,aligned,np.outer(u,u)-np.outer(v,v),np.outer(u,v)+np.outer(v,u))


@pytest.mark.parametrize('order',[25,101])
@pytest.mark.parametrize('generation',[1,2])
def test_parallel_tdi_preserves_all_intermediate_stages(order,generation):
    g=fixture(size=4099);detector=DetectorResponse.prepare(g,generation=generation,measurement_order=order)
    rng=np.random.default_rng(44103);links=rng.normal(size=(2,6,4099))
    expected,reference=detector.apply_links(links,return_stages=True)
    old_threads=get_num_threads();set_num_threads(4)
    try:actual,stages=detector.apply_links(links,return_stages=True,parallel=True)
    finally:set_num_threads(old_threads)
    assert_identical(actual,expected)
    for name in stages:
        for a,b in zip(stages[name],reference[name],strict=True):assert_identical(a,b)



@pytest.mark.parametrize('generation',[1,2])
@pytest.mark.parametrize('order',[25,101])
def test_parallel_detector_preparation_retains_all_states(generation,order):
    g=fixture(size=4099);reference=DetectorResponse.prepare(g,generation=generation,measurement_order=order)
    old_threads=get_num_threads();set_num_threads(4)
    try:candidate=DetectorResponse.prepare(g,generation=generation,measurement_order=order,parallel=True)
    finally:set_num_threads(old_threads)
    assert candidate.tdi.fingerprint==reference.tdi.fingerprint
    for a,b in zip(candidate.tdi.operators,reference.tdi.operators,strict=True):
        for name in ['shift_s','rate']:assert_identical(getattr(a,name),getattr(b,name))
        for name in ['centers','fractions','scales']:assert_identical(getattr(a.stencil,name),getattr(b.stencil,name))


def test_detector_only_stream_cache_survives_changing_sky_and_source_grid():
    from xtdi.inference import StreamedDetectorResponse
    from xtdi.optimized import apply_michelson
    epoch=3e7;times=epoch+np.arange(1003)*2.5;calls=[]
    def provider(q):
        calls.append(q.size)
        return fixture(epoch=epoch,size=q.size,times=q)
    stream=StreamedDetectorResponse(times,provider,block_samples=257,cache_bytes=64*1024**2,padding_order=121,parallel=True)
    old_threads=get_num_threads();set_num_threads(4)
    try:
        for index,(lam,beta) in enumerate([(1.2,.3),(4.9,-1.4)]):
            data=sampled(epoch=epoch,origin_offset=.2*index);k,p,c=basis(lam,beta)
            actual=stream.sampled_response(data,k,p,c,polarization_angle=.37)
            q=epoch+np.arange(-stream.halo,times.size+stream.halo)*2.5
            g=fixture(epoch=epoch,size=q.size,times=q)
            reference=DetectorResponse.prepare(g)
            projection=LinkProjection.prepare(k,p,c,g,polarization_angle=.37)
            queries=prepare_queries(data.origin_s,data.dt_s,data.values.shape[-1],projection.query_times,order=39)
            links=projection.apply(lambda _:evaluate_samples(data,queries))
            expected=apply_michelson(reference.tdi,links)[:,stream.halo:stream.halo+times.size]
            assert_identical(actual,expected)
            assert stream.last_stages['cache_misses']==(4 if index==0 else 0)
            assert stream.last_stages['cache_hits']==(0 if index==0 else 4)
        assert len(calls)==4
        with pytest.raises(ValueError):stream.sampled_response(data,k,p,c,max_output_bytes=10)
        stream.clear_cache();assert stream.cached_bytes==0
    finally:set_num_threads(old_threads)


def test_reference_fallback_matches_the_fused_host_result():
    from dataclasses import replace
    detector=DetectorResponse.prepare(fixture());fallback=replace(detector,compiled_sky_compatible=False)
    k,p,c=basis(.01,1.5);data=sampled()
    assert_identical(detector.sampled_links(data,k,p,c,polarization_angle=.37),fallback.sampled_links(data,k,p,c,polarization_angle=.37))


@pytest.mark.parametrize('settings', [dict(block_samples=2.5),dict(cache_bytes=1.5),
    dict(block_samples=True),dict(generation=True)])
def test_stream_rejects_noninteger_configuration(settings):
    from xtdi import StreamedDetectorResponse
    with pytest.raises(ValueError):
        StreamedDetectorResponse(np.arange(10)*2.5,lambda q:fixture(times=q),**settings)
