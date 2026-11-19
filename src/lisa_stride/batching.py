"""Independent-sky source batching with the existing FP64 response kernels.

Inputs are real FP64 arrays on one device with a common original source grid.
Only the grouping/dispatch of independent sources changes. All sky-dependent
quantities are recomputed by the public detector API on every invocation.
"""
from collections import deque
from functools import lru_cache
from xtdi.jax_response import JAXStreamedDetectorResponse
import jax
import jax.numpy as jnp

@lru_cache(maxsize=16)
def _group_kernel(order, unit):
    @jax.jit
    def evaluate(detector, values, origin, step, ks, ps, cs, angles):
        return jax.vmap(lambda v,k,p,c,a: detector.sampled_response_array(
            v,origin,step,k,p,c,order=order,polarization_angle=a,unit=unit))(
                values,ks,ps,cs,angles)
    return evaluate

def _integer(value, name):
    if isinstance(value,bool) or not isinstance(value,int) or value<1:
        raise ValueError(name+' must be a positive integer')

def sampled_response_batch_array(stream, values, origin, step, ks, ps, cs, angles, *,
                     microbatch=1, max_inflight=16, loop_order='blocks',
                     order=39, unit='frequency'):
    """Return independent (B,3,N) device outputs and per-source validity.

    Source slices and sky scalars are hoisted outside the detector-block loop.
    microbatch=1 uses the existing single-source kernel, with no outer vmap.
    A full warm detector cache is needed to avoid extra provider work with
    source-major iteration; cold-cache costs must be measured separately.
    Inputs have shapes (B,2,M), (B,3), (B,3,3), (B,3,3), (B,), respectively;
    the common original source origin and cadence are in seconds. Put source
    and sky arrays on the stream's device once before calling. Returned arrays
    remain on device; callers must reject false validity entries. The full
    output and compiler workspaces are additional to the detector-cache budget.
    This is host orchestration of compiled kernels, not a single JIT function.
    """
    if not isinstance(stream,JAXStreamedDetectorResponse):
        raise TypeError('a JAX StreamedDetectorResponse is required')
    if not all(isinstance(x,jax.Array) for x in (values,ks,ps,cs,angles)):
        raise TypeError('put source and sky arrays on the stream device before batching')
    if any(x.dtype!=jnp.float64 for x in (values,ks,ps,cs,angles)):
        raise ValueError('source and sky arrays must be float64')
    _integer(microbatch,'microbatch');_integer(max_inflight,'max_inflight')
    if loop_order not in ('sources','blocks'):raise ValueError('unknown loop order')
    if values.ndim!=3 or values.shape[1]!=2:raise ValueError('expected (B,2,M) source arrays')
    batch=values.shape[0]
    if batch<1:raise ValueError('at least one independent source required')
    if ks.shape!=(batch,3) or ps.shape!=(batch,3,3) or cs.shape!=(batch,3,3) or angles.shape!=(batch,):
        raise ValueError('one sky and polarization angle per independent source required')
    groups=[]
    for first in range(0,batch,microbatch):
        stop=min(first+microbatch,batch)
        # These slices may dispatch device work, but only once per group/call.
        if stop-first==1:
            args=(values[first],ks[first],ps[first],cs[first],angles[first]);scalar=True
        else:
            args=tuple(x[first:stop] for x in (values,ks,ps,cs,angles));scalar=False
        groups.append((scalar,args))
    parts=[[] for _ in groups];flags=[[] for _ in groups];pending=deque()
    blocks=[(s,min(stream.block_samples,stream.times.size-s))
            for s in range(0,stream.times.size,stream.block_samples)]
    tasks=((gi,s,n) for gi in range(len(groups)) for s,n in blocks) if loop_order=='sources' else (
        (gi,s,n) for s,n in blocks for gi in range(len(groups)))
    try:
        for gi,start,count in tasks:
            if len(pending)>=max_inflight:jax.block_until_ready(pending.popleft())
            detector,*_=stream._entry(start,count)
            scalar,(v,k,p,c,a)=groups[gi]
            if scalar:
                out,valid=detector.sampled_response_array(v,origin,step,k,p,c,
                    order=order,polarization_angle=a,unit=unit)
                out=out[None];valid=valid[None]
            else:out,valid=_group_kernel(order,unit)(detector,v,origin,step,k,p,c,a)
            out=out[...,stream.halo:stream.halo+count]
            parts[gi].append(out);flags[gi].append(valid);pending.append((out,valid))
    finally:
        for task in pending:jax.block_until_ready(task)
    return (jnp.concatenate([jnp.concatenate(p,axis=-1) for p in parts],axis=0),
            jnp.concatenate([jnp.all(jnp.stack(f),axis=0) for f in flags]))


__all__=['sampled_response_batch_array']
