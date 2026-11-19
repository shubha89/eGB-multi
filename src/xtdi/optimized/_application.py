"""Exact tiled evaluation of the reference sampled Michelson stages."""
import numpy as np
from ..delay import real_array
from ..michelson import LINK_LABELS,_CHANNEL_LINKS,link_array
from ._interpolation import apply_tiled
from ._parallel import apply_parallel

def apply_michelson(plan,links,*,unit='frequency',return_stages=False,parallel=False):
    if unit not in ['phase','frequency']:raise ValueError('invalid measurement unit')
    array=link_array(links,plan.times.size,name='eta/GW links')
    values={label:array[...,i,:] for i,label in enumerate(LINK_LABELS)}
    def delay(data,chain):
        op=plan.operator(chain);stencil=op.stencil;data=np.ascontiguousarray(data)
        if stencil.identity:shifted=data.copy()
        else:
            kernel=apply_parallel if parallel and plan.times.size>=2048 else apply_tiled
            shifted=kernel(data.reshape((-1,plan.times.size)),stencil.centers,
                stencil.fractions,stencil.scales,stencil.side_coefficients).reshape(data.shape)
        return shifted if unit=='phase' else (1.+op.rate)*shifted
    outputs=[];stages={}
    for channel,(a,b,c,d) in zip('XYZ',_CHANNEL_LINKS,strict=True):
        arm_a=values[a]+delay(values[b],(a,));arm_b=values[c]+delay(values[d],(c,))
        if return_stages:stages[channel+'.arms']=(arm_a,arm_b)
        if plan.generation==1:
            result=-arm_a+delay(arm_a,(c,d));result=result+arm_b
            result=result+(-delay(arm_b,(a,b)))
        else:
            round_a=arm_a+delay(arm_b,(a,b));round_b=delay(arm_a,(c,d))+arm_b
            if return_stages:stages[channel+'.rounds']=(round_a,round_b)
            result=-round_a+delay(round_a,(c,d,a,b));result=result+round_b
            result=result+(-delay(round_b,(a,b,c,d)))
        outputs.append(result)
    result=np.stack(outputs,axis=-2)
    return (result,stages) if return_stages else result


def apply_queries(plan,values):
    """Evaluate a prepared source query plan, retaining any leading batch axes.

    This checked array interface validates finite real source values. For an
    immutable SampledPolarizations object, evaluate_samples avoids rescanning
    a validated annual source table for every detector block.
    """
    values=real_array(values,name='source samples')
    if values.ndim<1 or values.shape[-1]!=plan.source_size:
        raise ValueError('source sample length differs from the prepared grid')
    return _query_arrays(plan,values)


def _query_arrays(plan,values):
    result=apply_tiled(values.reshape((-1,plan.source_size)),plan.centers,
        plan.fractions,plan.scales,plan.side_coefficients)
    return result.reshape(values.shape[:-1]+plan.query_shape)


def evaluate_samples(sampled,plan):
    """Query a validated immutable SampledPolarizations table on its own grid."""
    if (sampled.origin_s,sampled.dt_s,sampled.values.shape[-1])!=(plan.origin_s,plan.dt_s,plan.source_size):
        raise ValueError('query plan belongs to a different source time grid')
    return _query_arrays(plan,sampled.values)
