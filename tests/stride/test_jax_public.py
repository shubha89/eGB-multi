"""JAX-only public contract tests; no Numba, pyTDI or waveform model imports."""
import sys
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import lisa_stride as stride
from lisa_stride import LINK_LABELS

@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64():yield

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



def data(epoch=3e7):
    origin=epoch-5000.;dt=.625;t=np.arange(16384)*dt-5000.
    return stride.SampledPolarizations.from_arrays(np.cos(.02*t),.7*np.sin(.031*t+.3),
        origin_s=origin,dt_s=dt)


def test_jax_only_import_contract_in_subprocess():
    import subprocess,os
    script="""
import importlib.abc,sys
class NoCPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('numba','pytdi','egb_multi','phenomxpy'):
   raise ModuleNotFoundError(fullname,name=fullname)
sys.meta_path.insert(0,NoCPU())
import lisa_stride
assert not any(n in sys.modules for n in ('numba','pytdi','egb_multi','phenomxpy'))
try:lisa_stride.TDIPlan.prepare(None,None,backend='numba')
except ModuleNotFoundError as e:assert 'egb-multi[cpu]' in str(e)
else:raise AssertionError('Numba unexpectedly imported')
"""
    subprocess.run([sys.executable,'-c',script],check=True,env=dict(os.environ,PYTHONPATH=str(__import__('pathlib').Path(__file__).resolve().parents[2]/'src')))


@pytest.mark.parametrize('generation',[1,2])
@pytest.mark.parametrize('unit',['frequency','phase'])
def test_public_raw_plan_matches_detector_and_analytic_transfer(generation,unit):
    g=fixture(epoch=0.,size=1024);g['length'][:]=8.3
    plan=stride.TDIPlan.prepare(g['times'],g['length'],generation=generation,device='cpu')
    detector=stride.DetectorResponse.prepare(g,generation=generation,device='cpu')
    assert isinstance(plan,stride.JAXTDIPlan) and isinstance(detector,stride.JAXDetectorResponse)
    rng=np.random.default_rng(9);a=rng.normal(size=6)+1j*rng.normal(size=6)
    omega=2*np.pi*.003;d=np.exp(-1j*omega*8.3)
    links=np.real(a[:,None]*np.exp(1j*omega*g['times']))
    actual=jax.jit(lambda p,x:p.apply_array(x,unit=unit))(plan,links)
    np.testing.assert_array_equal(np.asarray(actual),np.asarray(detector.apply_links_array(links,unit=unit)))
    expected=[]
    for i,j,k,l in ((5,0,2,3),(4,1,0,5),(3,2,1,4)):
        amplitude=(a[i]+d*a[j]-a[k]-d*a[l])*(d**2-1.)*(1.-d**4 if generation==2 else 1.)
        expected.append(np.real(amplitude*np.exp(1j*omega*g['times'])))
    keep=slice(150,-150);expected=np.array(expected)
    assert np.linalg.norm(actual[:,keep]-expected[:,keep])/np.linalg.norm(expected[:,keep])<1e-11
    with pytest.raises(ValueError):plan.apply_array(links.astype('f4'))
    with pytest.raises(ValueError):plan.apply_array(links[:5])
    with pytest.raises(ValueError):plan.apply(links*np.nan)


def test_dynamic_plan_and_source_grid_reuse_outer_compilation():
    first=stride.DetectorResponse.prepare(fixture())
    g=fixture();g['length']+=.01
    second=stride.DetectorResponse.prepare(g)
    source=data();k,p,c=basis()
    traces=[]
    def call(detector,sampled,k,p,c):
        traces.append(1)
        return detector.sampled_response_array(sampled.values,sampled.origin_s,sampled.dt_s,k,p,c,order=sampled.order)
    compiled=jax.jit(call)
    out,valid=compiled(first,source,k,p,c);assert bool(valid)
    other=stride.SampledPolarizations(source.origin_s+.2,source.dt_s,source.values,source.order)
    out2,valid=compiled(second,other,*basis(2.1,-.4));assert bool(valid)
    assert len(traces)==1
    assert np.linalg.norm(out2-out)>0
    leaves=jax.tree.leaves(first)
    assert len(leaves)>5 and all(not isinstance(v,str) for v in leaves)
    assert isinstance(source.values,jax.Array)
    with pytest.raises(ValueError):stride.DetectorResponse.prepare(g,backend='unknown')


@pytest.mark.parametrize('budget',[0,8*1024**2])
def test_device_stream_validity_and_memory_bound(budget):
    epoch=3e7;times=epoch+np.arange(769)*2.5
    stream=stride.StreamedDetectorResponse(times,lambda t:fixture(times=t),
        block_samples=256,cache_bytes=budget,padding_order=121,device='cpu')
    sampled=data();k,p,c=basis()
    expected=stream.sampled_response(sampled,k,p,c)
    parts=[]
    for index,block,valid in stream.iter_arrays(sampled,k,p,c,max_inflight=2):
        assert bool(valid) and isinstance(block,jax.Array)
        parts.append(np.asarray(block))
        assert stream.cached_bytes<=budget
    np.testing.assert_array_equal(np.concatenate(parts,axis=-1),expected)
    bad=stride.SampledPolarizations(sampled.origin_s+1e5,sampled.dt_s,sampled.values)
    for _,block,valid in stream.iter_arrays(bad,k,p,c,max_inflight=1):
        assert not bool(valid) and np.all(np.isnan(block))
    with pytest.raises(ValueError):list(stream.iter_arrays(sampled,k,p,c,max_inflight=0))
    with pytest.raises(ValueError):stream.sampled_response(bad,k,p,c)


def test_checked_source_and_float64_contract():
    with pytest.raises(ValueError):stride.SampledPolarizations.from_arrays([1.],[2.],origin_s=0.,dt_s=1.)
    with pytest.raises(ValueError):stride.TDIPlan.prepare(np.arange(100.),np.ones((6,100)),generation=True)
    with jax.enable_x64(False):
        with pytest.raises(ValueError,match='x64'):stride.DetectorResponse.prepare(fixture())


def test_optional_numba_backend_uses_explicit_host_source():
    pytest.importorskip('numba')
    source=data();host=source.to_host();k,p,c=basis()
    cpu=stride.DetectorResponse.prepare(fixture(),backend='numba')
    device=stride.DetectorResponse.prepare(fixture())
    expected=cpu.sampled_response(host,k,p,c)
    actual=np.asarray(device.sampled_response(source,k,p,c))
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<1e-12
    plan=stride.TDIPlan.prepare(fixture()['times'],fixture()['length'],backend='numba')
    assert plan.measurement_order==25
    with jax.enable_x64(False):
        ready=stride.SampledPolarizations.from_arrays(np.ones(100),np.zeros(100),origin_s=0.,dt_s=1.,backend='numba')
        assert not ready.values.flags.writeable


def test_jax_only_geometry_rounding_against_explicit_fma():
    import math
    from xtdi.jax_response import _dot3
    rng=np.random.default_rng(20261116)
    vectors=np.concatenate((rng.normal(size=(10003,3))*500.,fixture(epoch=0.)['p_sender'].reshape(-1,3)))
    k,*_=basis()
    expected=np.array([math.fma(float(v[2]),float(k[2]),float(v[0]*k[0]))+v[1]*k[1] for v in vectors])
    actual=np.asarray(jax.jit(_dot3)(vectors,k))
    assert actual.tobytes()==expected.tobytes()


def test_jax_only_rounding_preserves_ieee_bits():
    from xtdi.jax_response import _round64
    rng=np.random.default_rng(161116)
    bits=rng.integers(0,2**64,size=1000003,dtype=np.uint64)
    values=bits.view(np.float64)
    actual=np.asarray(jax.jit(_round64)(values))
    assert actual.tobytes()==values.tobytes()
