"""Compensated periodic phases for long, uniformly modelled source evolution.

The source law is unchanged: angle = rate*(t-t0) + phase0 + rate_dot*(t-t0)^2/2.
Products, sums and reduction by 2*pi retain their floating-point residuals
until the final bounded angle. This avoids putting the rounding of a large
unwrapped orbital phase into small retarded-time waveform differences.
"""
from __future__ import annotations

import numpy as np

# Dekker's split for IEEE binary64 (53 significant bits). The low part of 2*pi
# is the difference between mathematical 2*pi and its nearest binary64 value.
_SPLITTER=134217729.0
_TWO_PI_HI=6.283185307179586
_TWO_PI_LO=2.4492935982947064e-16


def _operations(barrier):
    return (lambda a,b:barrier(a+b),lambda a,b:barrier(a-b),lambda a,b:barrier(a*b))


def _two_sum(a,b,add,sub):
    high=add(a,b);part=sub(high,a)
    low=add(sub(a,sub(high,part)),sub(b,part))
    return high,low


def _two_product(a,b,add,sub,mul):
    high=mul(a,b)
    a_split=mul(_SPLITTER,a);a_high=sub(a_split,sub(a_split,a));a_low=sub(a,a_high)
    b_split=mul(_SPLITTER,b);b_high=sub(b_split,sub(b_split,b));b_low=sub(b,b_high)
    low=add(add(add(sub(mul(a_high,b_high),high),mul(a_high,b_low)),mul(a_low,b_high)),mul(a_low,b_low))
    return high,low


def _reduce(high,low,xp,add,sub,mul):
    turns=xp.rint(high/_TWO_PI_HI)
    period_high,period_low=_two_product(turns,_TWO_PI_HI,add,sub,mul)
    reduced,residual=_two_sum(high,-period_high,add,sub)
    tail=sub(add(sub(low,period_low),residual),mul(turns,_TWO_PI_LO))
    angle=add(reduced,tail)
    angle=xp.where(angle>np.pi,sub(angle,_TWO_PI_HI),angle)
    return xp.where(angle < -np.pi,add(angle,_TWO_PI_HI),angle)


def _phase_backend(rate,times,t0,phase0,rate_dot,xp,barrier,*,quadratic):
    """Backend-neutral arithmetic; JAX supplies optimization barriers.

    Error-free transformations require each operation to be rounded in its
    written order. Barriers keep a compiled backend from algebraically removing
    residuals or contracting those operations into a different calculation.
    """
    add,sub,mul=_operations(barrier)
    delta,delta_low=_two_sum(times,-t0,add,sub)
    high,low=_two_product(rate,delta,add,sub,mul)
    low=add(low,mul(rate,delta_low))
    if quadratic:
        square,square_low=_two_product(delta,delta,add,sub,mul)
        square_low=add(square_low,add(mul(2*delta,delta_low),mul(delta_low,delta_low)))
        q_high,q_low=_two_product(.5*rate_dot,square,add,sub,mul)
        q_low=add(q_low,mul(.5*rate_dot,square_low))
        high,tail=_two_sum(high,q_high,add,sub)
        low=add(low,add(q_low,tail))
    high,tail=_two_sum(high,phase0,add,sub)
    return _reduce(high,add(low,tail),xp,add,sub,mul)


def periodic_linear_phase(rate,times,*,t0=0.,phase0=0.):
    """Return rate*(times-t0)+phase0 modulo 2*pi, in radians.

    Inputs are finite binary64 quantities in consistent time units. Intended
    LISA phases are many orders below the range where splitter overflow or
    non-representable turn counts would require a general large-argument
    trigonometric reduction. No physical frequency/epoch is adjusted.
    """
    return _phase_backend(np.asarray(rate,dtype=np.float64),np.asarray(times,dtype=np.float64),
        np.asarray(t0,dtype=np.float64),np.asarray(phase0,dtype=np.float64),0.,np,lambda x:x,quadratic=False)


def periodic_phase(angle,*,offset=0.):
    """Reduce an already integrated binary64 phase without losing its offset."""
    add,sub,mul=_operations(lambda x:x)
    high,low=_two_sum(np.asarray(angle,dtype=np.float64),np.asarray(offset,dtype=np.float64),add,sub)
    return _reduce(high,low,np,add,sub,mul)
