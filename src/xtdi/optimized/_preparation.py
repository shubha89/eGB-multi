"""Compile common ordered-path prefixes without changing sampled physics."""
import hashlib
import numpy as np
from numba import njit
from ..delay import check_order,readonly,FractionalDelay
from ..michelson import LINK_LABELS,_CHANNEL_LINKS,DelayOperator,MichelsonPlan,link_array,validate_times
from ._stencils import prepare_delay,coefficients,frozen_owned

def chains_for(generation):
    required=[]
    for a,b,c,d in _CHANNEL_LINKS:
        required.extend([(a,),(c,),(a,b),(c,d)])
        if generation==2:required.extend([(c,d,a,b),(a,b,c,d)])
    return tuple(dict.fromkeys(required))

def inputs(times,delays_s,delay_derivatives,generation,measurement_order,delay_order):
    times,dt=validate_times(times)
    if generation not in (1,2) or isinstance(generation,bool):raise ValueError('generation must be 1 or 2')
    measurement_order,delay_order=check_order(measurement_order),check_order(delay_order)
    delays=link_array(delays_s,times.size,name='delays')
    if delays.ndim!=2 or np.any(delays<0):raise ValueError('delays must be nonnegative and unbatched')
    rates=np.gradient(delays,dt,axis=-1) if delay_derivatives is None else link_array(
        delay_derivatives,times.size,name='delay derivatives')
    if rates.ndim!=2:raise ValueError('delay derivatives must be unbatched')
    return times,dt,delays,rates,measurement_order,delay_order

@njit(cache=True,fastmath=False,inline='always')
def _interpolate_pair(first,second,sample,shift,side_coefficients,weights):
    """The existing centered recurrence, same ascending accumulation order."""
    size=first.size;half=side_coefficients.size
    integer=np.floor(shift);fraction=shift-integer
    center=min(max(float(sample)+integer,-half-1.),size+half-1.)
    center=int(center);scale=1.
    for j in range(2,half):scale*=1.-(fraction/j)**2
    if half>1:scale*=(1.+fraction)*(1.-fraction/half)
    side=fraction*(1.-fraction)
    for j in range(1,half):
        side*=side_coefficients[j]
        weights[half-1-j]=side/(j+fraction)*scale
        weights[half+j]=side/(j+1.-fraction)*scale
    weights[half-1]=(1.-fraction)*scale;weights[half]=fraction*scale
    low=center-(half-1);a=0.;b=0.
    for k in range(2*half):
        index=low+k
        if 0<=index<size:
            a+=weights[k]*first[index];b+=weights[k]*second[index]
    return a,b

@njit(cache=True,fastmath=False)
def _prefix_states(delays,rates,parents,atoms,depths,fs,side_coefficients):
    nodes=parents.size;size=delays.shape[-1]
    shifts=np.zeros((nodes,size));retarded_rates=np.zeros((nodes,size))
    negative_rates=-rates;negative_delays=-delays
    weights=np.empty(2*side_coefficients.size,dtype=np.float64)
    for node in range(1,nodes):
        parent=parents[node];atom=atoms[node]
        atom_rates=negative_rates[atom];atom_delays=negative_delays[atom]
        for i in range(size):
            # The first atomic operation is an identity interpolation over the
            # full grid, exactly the special case used by the original code.
            if depths[node]==1:
                shifts[node,i]=0.+(-delays[atom,i])
                retarded_rates[node,i]=0.+(-rates[atom,i])
            else:
                scaled=shifts[parent,i]*fs
                if not np.isfinite(scaled):raise ValueError('Nonfinite compound shift')
                rate_value,shift_value=_interpolate_pair(atom_rates,atom_delays,i,scaled,side_coefficients,weights)
                retarded_rates[node,i]=retarded_rates[parent,i]+rate_value
                shifts[node,i]=shifts[parent,i]+shift_value
    return shifts,retarded_rates

def prepare_michelson(times,delays_s,*,delay_derivatives=None,generation=2,
                     measurement_order=101,delay_order=7,parallel=False):
    times,dt,delays,rates,measurement_order,delay_order=inputs(
        times,delays_s,delay_derivatives,generation,measurement_order,delay_order)
    chains=chains_for(generation)
    prefixes=sorted({c[:length] for c in chains for length in range(len(c)+1)},key=lambda c:(len(c),c))
    lookup={chain:i for i,chain in enumerate(prefixes)}
    parents=np.array([0 if not c else lookup[c[:-1]] for c in prefixes],dtype=np.int64)
    atoms=np.array([0 if not c else LINK_LABELS.index(c[-1]) for c in prefixes],dtype=np.int64)
    depths=np.array([len(c) for c in prefixes],dtype=np.int64)
    if parallel:
        from ._prepare_parallel import prefix_states_parallel
        shifts,retarded_rates,errors=prefix_states_parallel(delays,rates,parents,atoms,depths,1./dt,coefficients(delay_order))
        if np.any(errors):raise ValueError('Nonfinite compound shift')
    else:
        shifts,retarded_rates=_prefix_states(delays,rates,parents,atoms,depths,1./dt,coefficients(delay_order))
    terminal_arrays=None
    if parallel:
        from ._prepare_parallel import terminal_stencils_parallel
        terminal_arrays=terminal_stencils_parallel(shifts,np.array([lookup[c] for c in chains]),
                                                    1./dt,(measurement_order+1)//2)
        if np.any(terminal_arrays[4]):raise ValueError('Nonfinite terminal stencil shift')
        for array in terminal_arrays[:3]:array.flags.writeable=False
    operators=[]
    for operator_index,chain in enumerate(chains):
        shift=shifts[lookup[chain]];rate=retarded_rates[lookup[chain]]
        if terminal_arrays is None:
            stencil=prepare_delay(shift*(1./dt),times.size,measurement_order)
        else:
            stencil=FractionalDelay(*(frozen_owned(a[operator_index]) for a in terminal_arrays[:3]),
                coefficients(measurement_order),measurement_order,bool(np.all(terminal_arrays[3][operator_index])))
        operators.append(DelayOperator(chain,readonly(shift),readonly(rate),stencil))
    digest=hashlib.sha256()
    for value in (times,delays,rates):digest.update(value.tobytes())
    digest.update(repr((generation,measurement_order,delay_order,LINK_LABELS,
                       'native-additive-retarded-rates-v1')).encode())
    return MichelsonPlan(readonly(times),dt,generation,measurement_order,delay_order,
                         tuple(operators),digest.hexdigest())
