"""Fused preparation of the unchanged centered Lagrange parameters.

The existing recurrence and physical-time fraction calculation are retained.
Each output array is newly allocated and frozen; source data are never read
or changed during preparation. No coefficient table or reduced precision.
"""
from functools import lru_cache
import numpy as np
from numba import njit
from ..delay import FractionalDelay,check_order,real_array
from ..adapters.sampled import UniformQueryPlan


def frozen_owned(array):
    array.flags.writeable=False
    return array


@lru_cache(maxsize=32)
def coefficients(order):
    half=(order+1)//2;values=np.zeros(half)
    for j in range(1,half):values[j]=-(1.-j/half)/(1.+j/half)
    return frozen_owned(values)


@njit(cache=True,fastmath=False,inline='always')
def _scale(fraction,half):
    value=1.
    for j in range(2,half):value*=1.-(fraction/j)**2
    if half>1:value*=(1.+fraction)*(1.-fraction/half)
    return value


@njit(cache=True,fastmath=False)
def _delay_arrays(shifts,half):
    size=shifts.size
    centers=np.empty(size,dtype=np.int64);fractions=np.empty(size);scales=np.empty(size)
    identity=True
    for i in range(size):
        shift=shifts[i]
        if shift!=0:identity=False
        integer=np.floor(shift);fraction=shift-integer
        center=min(max(float(i)+integer,-half-1.),size+half-1.)
        centers[i]=int(center);fractions[i]=fraction;scales[i]=_scale(fraction,half)
    return centers,fractions,scales,identity


def prepare_delay(shifts,size,order=101):
    order=check_order(order)
    if not isinstance(size,(int,np.integer)) or size<1:raise ValueError('size must be a positive integer')
    shifts=real_array(shifts,name='shifts')
    if shifts.size==1:shifts=np.full(size,shifts.item())
    if shifts.shape!=(size,):raise ValueError('shifts must be scalar or have one value per sample')
    arrays=_delay_arrays(shifts,(order+1)//2)
    return FractionalDelay(*(frozen_owned(a) for a in arrays[:3]),coefficients(order),order,bool(arrays[3]))


@njit(cache=True,fastmath=False)
def _query_arrays(queries,origin,dt,size,half):
    count=queries.size
    centers=np.empty(count,dtype=np.int64);fractions=np.empty(count);scales=np.empty(count)
    for i in range(count):
        query=queries[i];normalized=(query-origin)/dt
        if not np.isfinite(normalized):raise ValueError('nonfinite normalized source times')
        integer=np.floor(normalized)
        if integer-(half-1)<0 or integer+half>=size:raise ValueError('source samples do not cover the query stencil')
        fraction=(query-(origin+integer*dt))/dt
        if fraction<0:integer-=1.;fraction+=1.
        elif fraction>=1:integer+=1.;fraction-=1.
        if fraction>=1:integer+=1.;fraction-=1.
        if integer-(half-1)<0 or integer+half>=size:raise ValueError('source samples do not cover the corrected query stencil')
        centers[i]=int(integer);fractions[i]=fraction;scales[i]=_scale(fraction,half)
    return centers,fractions,scales


def prepare_queries(origin_s,dt_s,source_size,query_times,*,order=101):
    order=check_order(order);origin_s,dt_s=float(origin_s),float(dt_s)
    if not np.isfinite(origin_s) or not np.isfinite(dt_s) or dt_s<=0:raise ValueError('finite origin and positive finite cadence required')
    if isinstance(source_size,(bool,np.bool_)) or not isinstance(source_size,(int,np.integer)) or source_size<order+1:
        raise ValueError('source_size must cover the stencil')
    queries=real_array(query_times,name='query times')
    if queries.size==0:raise ValueError('at least one source query required')
    arrays=_query_arrays(queries.ravel(),origin_s,dt_s,source_size,(order+1)//2)
    return UniformQueryPlan(origin_s,dt_s,int(source_size),queries.shape,
        *(frozen_owned(a) for a in arrays),coefficients(order),order)
