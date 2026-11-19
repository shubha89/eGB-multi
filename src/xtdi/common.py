"""Backend-independent numerical contracts; NumPy only, no compiled backend imports.

Receiver/emitter labels, seconds and Cartesian light-seconds follow the
legacy xtdi API. The recurrences are unchanged from that implementation.
"""
from collections.abc import Mapping
from functools import lru_cache
import math
import numpy as np

def check_order(order):
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or order < 1 or order % 2 != 1:
        raise ValueError("interpolation order must be a positive odd integer")
    return int(order)


def real_array(value, *, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real; complex streams are not supported")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


LINK_LABELS = ("21", "32", "13", "31", "23", "12")
_CHANNEL_LINKS = (("12", "21", "13", "31"),
                  ("23", "32", "21", "12"),
                  ("31", "13", "32", "23"))


def validate_times(times):
    times = real_array(times, name="times")
    if times.ndim != 1 or times.size < 4:
        raise ValueError("at least four uniformly spaced times in seconds are required")
    steps = np.diff(times)
    if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-9, atol=1e-10):
        raise ValueError("times must be increasing and uniformly sampled")
    return times, float(steps[0])


def link_array(values, size, *, name):
    if isinstance(values, Mapping):
        if set(values) != set(LINK_LABELS):
            raise ValueError(f"{name} must have exactly these link labels: {LINK_LABELS}")
        values = np.stack([real_array(values[label], name=name) for label in LINK_LABELS], axis=-2)
    array = real_array(values, name=name)
    if array.ndim < 2 or array.shape[-2:] != (6, size):
        raise ValueError(f"{name} shape must be (..., 6, {size}) in LINK_LABELS order")
    return array


def chains_for(generation):
    required=[]
    for a,b,c,d in _CHANNEL_LINKS:
        required.extend([(a,),(c,),(a,b),(c,d)])
        if generation==2:required.extend([(c,d,a,b),(a,b,c,d)])
    return tuple(dict.fromkeys(required))

RECEIVERS=np.array([int(label[0])-1 for label in LINK_LABELS])
FIRST=np.array([np.flatnonzero(RECEIVERS==i)[0] for i in range(3)])
RECEIVERS.flags.writeable=False
FIRST.flags.writeable=False

@lru_cache(maxsize=32)
def coefficients(order):
    half=(order+1)//2;values=np.zeros(half)
    for j in range(1,half):values[j]=-(1.-j/half)/(1.+j/half)
    values.flags.writeable=False
    return values

def conservative_halo(dt, delay_bound_s, measurement_order, delay_order, generation=2):
    # A retained TDI2 sample uses <=8 atomic light times and <=3 staged
    # measurement stencils. Four complete stencil widths additionally bound
    # the retarded delay-plan interpolation and endpoint derivative support.
    # This intentionally exceeds the minimum; tests compare contiguous data.
    return math.ceil((8 if generation == 2 else 4)*delay_bound_s/dt) + \
        4*(measurement_order+1)+4*(delay_order+1)+2
