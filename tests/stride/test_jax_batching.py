"""Independent-sky ordering, partial blocks and invalid-source propagation."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from lisa_stride import StreamedDetectorResponse
from lisa_stride.batching import sampled_response_batch_array
from test_jax_public import fixture,basis

@pytest.mark.parametrize('generation',[1,2])
@pytest.mark.parametrize('unit',['frequency','phase'])
def test_distinct_sky_batch_preserves_sources_and_partial_blocks(generation,unit):
    with jax.enable_x64():
        epoch=3e7;origin=epoch-5000.;dt=.625;t=np.arange(16384)*dt
        values=jnp.asarray(np.array([np.stack([np.cos(.02*t+.2*i),(.7+.1*i)*np.sin(.031*t+.3)]) for i in range(3)]))
        ks,ps,cs=[jnp.asarray(np.stack(a)) for a in zip(*[basis(1.2+.2*i,.3-.1*i) for i in range(3)])]
        angles=jnp.array([.1,-.2,.3],dtype=jnp.float64);calls=[]
        def geometry(q):calls.append(len(q));return fixture(times=q,retarded=True)
        stream=StreamedDetectorResponse(epoch+np.arange(769)*2.5,geometry,device='cpu',generation=generation,
            measurement_order=7,block_samples=256,padding_order=121,cache_bytes=16*1024**2,execution_schedule='fused_unrolled')
        # Existing unary API supplies an independent source-by-source collector.
        expected=[]
        for i in range(3):
            parts=[]
            for start in range(0,769,256):
                count=min(256,769-start);det,*_=stream._entry(start,count)
                out,valid=det.sampled_response_array(values[i],origin,dt,ks[i],ps[i],cs[i],order=39,polarization_angle=angles[i],unit=unit)
                assert bool(valid);parts.append(np.asarray(out[...,stream.halo:stream.halo+count]))
            expected.append(np.concatenate(parts,axis=-1))
        before=len(calls)
        for group,loop in [(1,'blocks'),(2,'sources'),(3,'blocks')]:
            out,valid=sampled_response_batch_array(stream,values,origin,dt,ks,ps,cs,angles,
                microbatch=group,max_inflight=3,loop_order=loop,unit=unit)
            assert out.shape==(3,3,769) and np.all(valid)
            expected_array=np.array(expected)
            assert np.linalg.norm(np.asarray(out)-expected_array)/np.linalg.norm(expected_array)<1e-12
            assert len(calls)==before
        # Invalid sky support affects exactly the corresponding output source.
        out,valid=sampled_response_batch_array(stream,values,origin,dt,ks.at[1].set(ks[1]*2),ps,cs,angles)
        assert np.array_equal(valid,[True,False,True]) and np.all(np.isnan(out[1]))
        out,valid=sampled_response_batch_array(stream,values,origin+1e6,dt,ks,ps,cs,angles)
        assert not np.any(valid) and np.all(np.isnan(out))


def test_batch_requires_device_fp64_inputs_before_geometry_work():
    with jax.enable_x64():
        def unused_geometry(q):raise AssertionError('invalid input reached preparation')
        stream=StreamedDetectorResponse(3e7+np.arange(128)*2.5,unused_geometry,device='cpu')
        values=jnp.ones((1,2,8192),dtype=jnp.float64);k,p,c=basis()
        ks=jnp.asarray(k[None]);ps=jnp.asarray(p[None]);cs=jnp.asarray(c[None]);angles=jnp.zeros(1,dtype=jnp.float64)
        with pytest.raises(TypeError,match='device'):
            sampled_response_batch_array(stream,np.asarray(values),3e7-5000,.625,ks,ps,cs,angles)
        with pytest.raises(ValueError,match='float64'):
            sampled_response_batch_array(stream,values.astype(jnp.float32),3e7-5000,.625,ks,ps,cs,angles)
