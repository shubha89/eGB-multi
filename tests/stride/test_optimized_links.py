"""Independent Cartesian link formula and source-batch linearity."""
import numpy as np
import pytest
from xtdi import LINK_LABELS
from xtdi.optimized import LinkProjection,prepare_queries,apply_queries,evaluate_samples
from xtdi.adapters.sampled import SampledPolarizations


def geometry():
    time=3e7+np.arange(257)*2.5
    positions=np.array([[0.,0.,0.],[8.,0.,0.],[4.,7.,.2]])
    n=[];length=[];senders=[];receivers=[]
    for label in LINK_LABELS:
        r,e=int(label[0])-1,int(label[1])-1
        delta=positions[r]-positions[e];arm=np.linalg.norm(delta)
        n.append(np.tile(delta/arm,(time.size,1)));length.append(np.full(time.size,arm))
        senders.append(np.tile(positions[e],(time.size,1)));receivers.append(np.tile(positions[r],(time.size,1)))
    return dict(times=time,n=np.array(n),length=np.array(length),p_sender=np.array(senders),p_receiver=np.array(receivers))


@pytest.mark.parametrize('batch',[1,4])
@pytest.mark.parametrize('psi',[0.,.4])
def test_model_independent_link_formula_and_receiver_reuse(batch,psi):
    g=geometry();k=np.array([-.3,-.4,-np.sqrt(.75)])
    u=np.cross(k,[0.,0.,1.]);u/=np.linalg.norm(u);v=np.cross(k,u)
    p=np.outer(u,u)-np.outer(v,v);c=np.outer(u,v)+np.outer(v,u)
    plan=LinkProjection.prepare(k,p,c,g,polarization_angle=psi)
    scale=np.arange(1,batch+1)[:,None,None]
    def callback(q):return scale*np.stack([np.cos(.02*q),.7*np.sin(.03*q+.3)])
    actual=plan.apply(callback)
    pp=np.cos(2*psi)*p-np.sin(2*psi)*c;cc=np.sin(2*psi)*p+np.cos(2*psi)*c
    sender=g['times'][None,:]-g['length']-np.einsum('lti,i->lt',g['p_sender'],k)
    receiver=g['times'][None,:]-np.einsum('lti,i->lt',g['p_receiver'],k)
    a=np.einsum('lti,ij,ltj->lt',g['n'],pp,g['n']);b=np.einsum('lti,ij,ltj->lt',g['n'],cc,g['n'])
    expected=.5*(a*(np.cos(.02*sender)-np.cos(.02*receiver))
                 +b*.7*(np.sin(.03*sender+.3)-np.sin(.03*receiver+.3)))/(1-np.einsum('lti,i->lt',g['n'],k))
    np.testing.assert_allclose(actual,scale*expected,rtol=5e-13,atol=1e-14)
    assert plan.query_times.size==9*g['times'].size
    assert all(not a.flags.writeable for a in vars(plan).values())
    # Unbatched and batched calls implement the same linear response.
    np.testing.assert_array_equal(actual[0],plan.apply(lambda q:callback(q)[0]))


def test_query_application_grid_validation_and_source_batches():
    times=-5000.+np.arange(2048)*2.5
    sampled=SampledPolarizations.from_arrays(np.sin(.02*times),np.cos(.02*times),origin_s=-5000.,dt_s=2.5,order=39)
    plan=prepare_queries(-5000.,2.5,2048,np.array([-4000.1,-4002.5,-4000.1]),order=39)
    np.testing.assert_array_equal(evaluate_samples(sampled,plan),plan.apply(sampled.values))
    batch=np.stack([sampled.values,2*sampled.values]);actual=apply_queries(plan,batch)
    np.testing.assert_array_equal(actual[0],plan.apply(sampled.values))
    np.testing.assert_array_equal(actual[1],2*actual[0])
    with pytest.raises(ValueError):apply_queries(plan,batch[:,:,:-1])
    with pytest.raises(ValueError):apply_queries(plan,np.full_like(batch,np.nan))
    with pytest.raises(ValueError):evaluate_samples(SampledPolarizations(-5001.,2.5,sampled.values),plan)


def test_projection_rejects_wrong_geometry_and_callback():
    g=geometry();k=np.array([0.,0.,1.]);p=np.diag([1.,-1.,0.]);c=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])
    plan=LinkProjection.prepare(k,p,c,g)
    with pytest.raises(ValueError):plan.apply(lambda q:np.zeros((q.size,2)))
    with pytest.raises(ValueError):LinkProjection.prepare(k*2,p,c,g)
    bad=dict(g);bad['length']=-g['length']
    with pytest.raises(ValueError):LinkProjection.prepare(k,p,c,bad)
    bad=dict(g);bad['p_receiver']=g['p_receiver'].copy();bad['p_receiver'][0,:,2]+=1.
    with pytest.raises(ValueError):LinkProjection.prepare(k,p,c,bad)
