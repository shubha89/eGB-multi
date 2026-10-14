"""Primary-laser covariance control for ideal N1-12 locking and moving TDI2.

Laser-only local fluctuations are p12=p13=p, p21=p23=D21 p and
p31=p32=D31 p. The raw SCI fluctuation is Dij pji - pij, and the
laser-only TMI/REF fluctuations vanish for these adjacent pairs. Substituting
these paths into the native raw pyTDI polynomial gives a primary-laser
polynomial with the correct nested order and frequency-delay rates.

The local transfer H(f,t) is evaluated with arbitrary-precision complex
arithmetic to control catastrophic cancellation. Geometry and native delay
interpolation remain float64 inputs. H S_p H^H is an adiabatic covariance
symbol, not a complete finite-observation covariance or a numerical model of
the sampled instrument's interpolation artifacts. Only N1-12 is supported.
"""

import numpy as np

from .tdi import LINK_LABELS


def primary_laser_polynomials(*, locking="N1-12"):
    """Native XYZ polynomials in the single ideal primary frequency noise."""
    if locking != "N1-12":
        raise ValueError("This diagnostic is derived only for N1-12 locking")
    from pytdi import TDICombination
    from pytdi.michelson import X2, Y2, Z2
    local = {"12": (), "13": (), "21": ("D_21",), "23": ("D_21",),
        "31": ("D_31",), "32": ("D_31",)}
    combinations = []
    for raw in (X2, Y2, Z2):
        terms = {}
        for measurement, polynomial in raw.components.items():
            if not measurement.startswith("sci_"):
                continue
            label = measurement[4:]
            remote = label[::-1]
            for coefficient, chain in polynomial:
                received = tuple(chain) + (f"D_{label}",) + local[remote]
                emitted = tuple(chain) + local[label]
                terms[received] = terms.get(received, 0.) + coefficient
                terms[emitted] = terms.get(emitted, 0.) - coefficient
        combinations.append(TDICombination({"primary": [(value,list(path)) for path,value in terms.items() if value != 0]}))
    return tuple(combinations)


def moving_primary_laser_transfer(frequency_hz, times, delays, epoch_indices, *, delay_order=7, precision_digits=60,
    delay_derivatives=None, rate_mode="native_first_order", summation="mpmath"):
    """Return H_A,H_E per unit primary frequency fluctuation, shape (t,f,2).

    Delay-rate factors retain frequency units. Divide by optical carrier to
    convert Hz-valued transfer outputs to fractional-frequency outputs.
    Arbitrary precision refines transfer summation, not delay/orbit inputs.
    The native rate sums retarded derivatives to first order. The independent
    ``chain_product`` control instead multiplies every retarded (1-Ldot), the
    exact chain rule for a composed frequency-delay operator. Both use native
    sampled/interpolated shifts. Supplied derivatives allow an independent
    smooth orbit-derivative control without changing the underlying delays.
    ``grouped_numpy`` groups paths with equal commuting delay counts. Their
    coefficients sum to zero, so expm1 of their small shift differences and
    high-precision rate differences avoid subtracting unit exponentials.
    This faster evaluation must be checked against the direct mpmath sum.
    """
    import mpmath as mp
    f, t = np.asarray(frequency_hz,dtype=float), np.asarray(times,dtype=float)
    indices = np.asarray(epoch_indices,dtype=int)
    if f.ndim != 1 or not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("Positive finite frequencies required")
    if t.ndim != 1 or len(t) < 32 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0) or not np.allclose(np.diff(t),t[1]-t[0]):
        raise ValueError("Uniform finite time grid required")
    if set(delays) != set(LINK_LABELS) or any(np.shape(v) != t.shape or np.any(np.asarray(v)<=0) or not np.all(np.isfinite(v)) for v in delays.values()):
        raise ValueError("Six aligned finite positive delays required")
    if precision_digits < 30 or indices.ndim != 1 or indices.size == 0 or np.any(indices < 8*(delay_order+1)) or np.any(indices >= len(t)-8*(delay_order+1)):
        raise ValueError("High-precision arithmetic and complete interpolation halos required")
    raw = {f"d_{label}":np.asarray(value) for label,value in delays.items()}
    if rate_mode not in ("native_first_order", "chain_product"):
        raise ValueError("Unknown delay-rate control")
    if summation not in ("mpmath","grouped_numpy"):
        raise ValueError("Unknown transfer summation control")
    derivatives = {label:np.gradient(value,t[1]-t[0],edge_order=2) for label,value in raw.items()} if delay_derivatives is None else {f"d_{label}":np.asarray(value) for label,value in delay_derivatives.items()}
    if set(derivatives)!=set(raw) or any(value.shape!=t.shape or not np.all(np.isfinite(value)) for value in derivatives.values()):
        raise ValueError("Six aligned finite delay derivatives required")
    from pytdi import dsp
    polynomials = primary_laser_polynomials()
    geometry = []
    for polynomial in polynomials:
        shifts, rates = polynomial.build_shifts(raw,1/(t[1]-t[0]),derivatives,order=delay_order)
        channel=[]
        for coefficient,path in polynomial.components["primary"]:
            if rate_mode=="native_first_order":
                deltas=[np.broadcast_to(rates[tuple(path)],t.shape)[indices]]
            else:
                partial=np.zeros_like(t);deltas=[]
                for atom in path:
                    if not atom.startswith("D_"):
                        raise ValueError("Primary diagnostic supports delay paths only")
                    key="d_"+atom[2:]
                    deltas.append(-dsp.timeshift(derivatives[key],partial/(t[1]-t[0]),delay_order)[indices])
                    partial+=dsp.timeshift(-raw[key],partial/(t[1]-t[0]),delay_order)
            channel.append((coefficient,np.broadcast_to(shifts[tuple(path)],t.shape)[indices],deltas))
        geometry.append(channel)
    result = np.empty((len(indices),len(f),2),complex)
    with mp.workdps(precision_digits):
        for row in range(len(indices)):
            # Keep the small rate correction separate from unity until inside
            # the high-precision calculation; rounding 1+delta in float64
            # would reintroduce a cancellation floor in this control.
            terms = [[(mp.mpf(float(c)),mp.mpf(float(s[row])),mp.fprod(1+mp.mpf(float(delta[row])) for delta in deltas)) for c,s,deltas in channel] for channel in geometry]
            if summation=="grouped_numpy":
                xyz=[]
                omega=2*np.pi*f
                for channel,polynomial in zip(terms,polynomials):
                    groups={}
                    for term,(_,path) in zip(channel,polynomial.components["primary"]):
                        counts=tuple(path.count("D_"+label) for label in LINK_LABELS)
                        groups.setdefault(counts,[]).append(term)
                    value=np.zeros(len(f),complex)
                    for group in groups.values():
                        if mp.fsum(c for c,_,_ in group)!=0:
                            raise ValueError("Primary paths must cancel in commuting limit")
                        _,shift0,rate0=group[0]
                        small=np.zeros(len(f),complex)
                        for coefficient,shift,rate in group:
                            small+=float(coefficient)*(float(rate-rate0)+float(rate)*np.expm1(1j*omega*float(shift-shift0)))
                        value+=np.exp(1j*omega*float(shift0))*small
                    xyz.append(value)
                result[row,:,0]=(xyz[2]-xyz[0])/np.sqrt(2)
                result[row,:,1]=(xyz[0]-2*xyz[1]+xyz[2])/np.sqrt(6)
                continue
            for column,frequency in enumerate(f):
                omega = 2*mp.pi*mp.mpf(float(frequency))
                xyz = [mp.fsum(c*r*mp.exp(1j*omega*s) for c,s,r in channel) for channel in terms]
                result[row,column,0] = complex((xyz[2]-xyz[0])/mp.sqrt(2))
                result[row,column,1] = complex((xyz[0]-2*xyz[1]+xyz[2])/mp.sqrt(6))
    return result
