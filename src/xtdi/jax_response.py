"""Primary float64 JAX response from caller-supplied Cartesian geometry.

Detector preparation, actual sky projection, sampled strain interpolation,
links and TDI have device-neutral array kernels. Orbit/waveform providers and
cache management are external. CPU and A100 qualification is configuration-specific.
Numba remains an optional CPU backend. Other GPU settings require validation.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .common import (check_order, real_array, LINK_LABELS, validate_times,
                     link_array, FIRST, RECEIVERS, chains_for, coefficients)
from ._host_stream import HostStreamBase
from ._dot_policy import numpy_dot_uses_fma

_HOST_DOT_USES_FMA = numpy_dot_uses_fma()


def _device(device):
    """Resolve an explicit JAX device without changing global configuration."""
    if device is None or not isinstance(device,str): return device
    platform, _, index=device.partition(':')
    devices=jax.devices(platform)
    number=int(index) if index else 0
    if number < 0 or number >= len(devices): raise ValueError('requested device is unavailable')
    return devices[number]


def _require_x64():
    if not jax.config.x64_enabled:
        raise ValueError('STRIDE JAX response requires explicitly enabled JAX x64')


@jax.custom_jvp
def _round64(x):
    """Preserve a float64 rounding boundary through XLA contraction.

    Each ordinary normal value is reconstructed as its mantissa with the
    lowest bit cleared plus that exact signed bit value. No information or
    precision is discarded. Exceptional and very small values retain their
    original bits. This enforces the measured CPU rounding boundaries with
    fewer operations than a nextafter round trip. The JVP is identity.
    Other devices still require independent numerical validation.
    """
    # Exact bit-preserving reconstruction through a low-bit split. The
    # explicit mantissa bit operation forces the preceding arithmetic to
    # produce its float64 value before a subsequent multiply/add can contract.
    bits=lax.bitcast_convert_type(x,jnp.uint64)
    sign=bits & jnp.uint64(0x8000000000000000)
    exponent=(bits>>jnp.uint64(52)) & jnp.uint64(2047)
    high=lax.bitcast_convert_type(bits & jnp.uint64(0xfffffffffffffffe),jnp.float64)
    normal_ulp=(exponent-jnp.uint64(52))<<jnp.uint64(52)
    low=lax.bitcast_convert_type(sign|jnp.where((bits & jnp.uint64(1))!=0,normal_ulp,jnp.uint64(0)),jnp.float64)
    result=high+low
    # Preserve zeros/subnormals/very small normals and all NaN/Inf bit patterns.
    return jnp.where((exponent<=52)|(exponent==2047),x,result)


@_round64.defjvp
def _round64_jvp(primals, tangents):
    return _round64(primals[0]), tangents[0]


def _add(a, b): return _round64(a+b)
def _sub(a, b): return _round64(a-b)
def _mul(a, b): return _round64(a*b)
def _div(a, b): return _round64(a/b)


def _fma_bounded(a, b, c):
    """Compensated product-plus-sum for bounded normal orbital quantities.

    Dekker splitting and Knuth two-sum reproduce the explicitly fused host
    geometry contraction on the checked domain, without a CPU callback. This
    is not a general replacement for IEEE fma at overflow/subnormal extremes.
    """
    split = 134217729.  # 2**27+1 for a binary64 53-bit significand.
    sa, sb = _mul(split, a), _mul(split, b)
    ah, bh = _sub(sa, _sub(sa, a)), _sub(sb, _sub(sb, b))
    al, bl = _sub(a, ah), _sub(b, bh)
    product = _mul(a, b)
    error = _add(_add(_add(_sub(_mul(ah, bh), product), _mul(ah, bl)),
                      _mul(al, bh)), _mul(al, bl))
    total = _add(product, c)
    z = _sub(total, product)
    residual = _add(_sub(product, _sub(total, z)), _sub(c, z))
    return _add(total, _add(error, residual))


def _dot3(vector, direction):
    # Select once from a checked host probe, then compile only device arithmetic.
    # No host callback occurs during a sky, link or response evaluation.
    first = _mul(vector[..., 0], direction[0])
    if _HOST_DOT_USES_FMA:
        first = _fma_bounded(vector[..., 2], direction[2], first)
    else:
        first = _add(first, _mul(vector[..., 2], direction[2]))
    return _add(first, _mul(vector[..., 1], direction[1]))


# Interpolation weights and sums retain float64 and the same recurrence,
# allowing normal XLA contraction. Physical source seconds above/below retain
# explicit rounding boundaries. Independent physical/refinement gates apply.
def _filter_add(a,b): return lax.optimization_barrier(a+b)
def _filter_sub(a,b): return lax.optimization_barrier(a-b)
def _filter_mul(a,b): return lax.optimization_barrier(a*b)
def _filter_div(a,b): return lax.optimization_barrier(a/b)


def _scale(fraction, half, *, unroll=False):
    def step(j, value):
        divided = _filter_div(fraction, j)
        return _filter_mul(value, _filter_sub(1., _filter_mul(divided, divided)))
    value = lax.fori_loop(2, half, step, jnp.ones_like(fraction), unroll=unroll)
    if half > 1:
        value = _filter_mul(value, _filter_mul(_filter_add(1., fraction), _filter_sub(1., _filter_div(fraction, half))))
    return value


def _interpolate(values, centers, fraction, scale, coeff, *, zero_boundary):
    """Original side recurrence and ascending tap sum, not a new filter."""
    half = coeff.shape[0]
    def weights(previous, item):
        j, coefficient = item
        side = _filter_mul(previous, coefficient)
        return side, (_filter_mul(_filter_div(side, _filter_add(j, fraction)), scale),
                      _filter_mul(_filter_div(side, _filter_sub(j+1., fraction)), scale))
    _, (left, right) = lax.scan(weights, _filter_mul(fraction, _filter_sub(1., fraction)),
        (jnp.arange(1, half, dtype=jnp.float64), coeff[1:]))
    taps = jnp.concatenate((left[::-1], _filter_mul(_filter_sub(1., fraction), scale)[None],
                            _filter_mul(fraction, scale)[None], right), axis=0)
    def step(tap, total):
        index = centers+tap-(half-1)
        data = jnp.take(values, jnp.clip(index, 0, values.shape[-1]-1), axis=-1)
        if zero_boundary:
            data = jnp.where((index >= 0) & (index < values.shape[-1]), data, 0.)
        return _filter_add(total, _filter_mul(taps[tap], data))
    return lax.fori_loop(0, 2*half, step,
        jnp.zeros(values.shape[:-1]+centers.shape, dtype=jnp.float64))


def _paired_interpolate(values, centers, fraction, scale, order, *, unroll=False):
    """Same Lagrange polynomial, paired taps with explicit float64 checking.

    Reassociation changes final rounding; numerical equivalence, not byte
    equality of filter outputs, is checked against independent references.
    Physical source-query times are not changed by this schedule.
    """
    half=(order+1)//2
    ratios=jnp.asarray(np.cumprod(np.r_[1.,coefficients(order)[1:]]))
    def sample(offset):
        return jnp.take(values,jnp.clip(centers+offset,0,values.shape[-1]-1),axis=-1)
    total=(1.-fraction)*sample(0)+fraction*sample(1)
    def step(j,total):
        factor=fraction*(1.-fraction)*ratios[j]
        return total+factor*(sample(-j)/(j+fraction)+sample(1+j)/(j+1.-fraction))
    return lax.fori_loop(1,half,step,total,unroll=unroll)*scale


def _tdi(values, parameters, generation, order, frequency, *, unroll=False):
    """Map independent XYZ channels, with no CPU device pin or callback."""
    centers,fractions,scales,rates,ratios=parameters
    operators=6 if generation==2 else 4
    size=values.shape[-1];half=(order+1)//2
    centers=centers.reshape(3,operators,size)
    fractions=fractions.reshape(3,operators,size)
    scales=scales.reshape(3,operators,size)
    rates=rates.reshape(3,operators,size)
    channel_links=np.array(((5,0,2,3),(4,1,0,5),(3,2,1,4)))
    streams=jnp.moveaxis(values[...,channel_links,:],-3,0)
    def channel(streams,cc,ff,ss,rr):
        def delay(data,index):
            center,fraction,scale=cc[index],ff[index],ss[index]
            def sample(offset):
                indices=center+offset
                return jnp.where((indices>=0)&(indices<size),
                    jnp.take(data,jnp.clip(indices,0,size-1),axis=-1),0.)
            shifted=((1.-fraction)*sample(0)+fraction*sample(1))*scale
            def step(j,total):
                factor=fraction*(1.-fraction)*ratios[j]*scale
                return total+factor*(sample(-j)/(j+fraction)+sample(1+j)/(j+1.-fraction))
            shifted=lax.fori_loop(1,half,step,shifted,unroll=unroll)
            return (1.+rr[index])*shifted if frequency else shifted
        arm_a=streams[...,0,:]+delay(streams[...,1,:],0)
        arm_b=streams[...,2,:]+delay(streams[...,3,:],1)
        if generation==1:
            result=-arm_a+delay(arm_a,3)
            result=result+arm_b
            result=result+(-delay(arm_b,2))
        else:
            round_a=arm_a+delay(arm_b,2)
            round_b=delay(arm_a,3)+arm_b
            result=-round_a+delay(round_a,4)
            result=result+round_b
            result=result+(-delay(round_b,5))
        return result
    result=jax.vmap(channel)(streams,centers,fractions,scales,rates)
    return jnp.moveaxis(result,0,-2)


@lru_cache(maxsize=32)
def _prepare_kernel(generation, measurement_order, delay_order):
    chains = chains_for(generation)
    prefixes = sorted({c[:i] for c in chains for i in range(len(c)+1)}, key=lambda c:(len(c), c))
    lookup = {c:i for i,c in enumerate(prefixes)}
    groups = [[c for c in prefixes if len(c) == depth] for depth in range(1, max(map(len, prefixes))+1)]
    @jax.jit
    def prepare(delays, rates, dt):
        size = delays.shape[-1]
        coeff = jnp.asarray(coefficients(delay_order))
        half = (delay_order+1)//2
        shifts = [jnp.zeros((1, size), dtype=jnp.float64)]
        retarded = [jnp.zeros((1, size), dtype=jnp.float64)]
        previous = [()]
        for group in groups:
            parents = np.array([previous.index(c[:-1]) for c in group])
            atoms = np.array([LINK_LABELS.index(c[-1]) for c in group])
            inherited, rate = shifts[-1][parents], retarded[-1][parents]
            if len(group[0]) == 1:
                shift, value = _add(0., -delays[atoms]), _add(0., -rates[atoms])
            else:
                scaled = _mul(inherited, _div(1., dt))
                integer = jnp.floor(scaled)
                fraction = _sub(scaled, integer)
                centers = jnp.clip(jnp.arange(size, dtype=jnp.float64)[None]+integer,
                                   -half-1., size+half-1.).astype(jnp.int64)
                scale = _scale(fraction, half)
                atomic = jnp.stack((-rates[atoms], -delays[atoms]), axis=1)
                interpolated = jax.vmap(lambda x,c,f,s:_interpolate(x,c,f,s,coeff,
                    zero_boundary=True))(atomic, centers, fraction, scale)
                shift, value = _add(inherited, interpolated[:,1]), _add(rate, interpolated[:,0])
            shifts.append(shift); retarded.append(value); previous = group
        indices = np.array([lookup[c] for c in chains])
        selected = jnp.concatenate(shifts)[indices]
        rates = jnp.concatenate(retarded)[indices]
        scaled = _mul(selected, _div(1., dt))
        integer = jnp.floor(scaled)
        fraction = _sub(scaled, integer)
        half = (measurement_order+1)//2
        centers = jnp.clip(jnp.arange(size, dtype=jnp.float64)[None]+integer,
                           -half-1., size+half-1.).astype(jnp.int64)
        # Small immutable coefficients are compile-time constants, not host callbacks.
        ratios = np.cumprod(np.r_[1., coefficients(measurement_order)[1:]])
        parameters = (centers, fraction, _scale(fraction, half), rates, jnp.asarray(ratios))
        return parameters, selected
    return prepare


@jax.jit
def _rates(delays, dt):
    middle = _div(_sub(delays[:,2:], delays[:,:-2]), _mul(2., dt))
    return jnp.concatenate((_div(_sub(delays[:,1:2], delays[:,:1]), dt), middle,
                            _div(_sub(delays[:,-1:], delays[:,-2:-1]), dt)), axis=-1)


def _project(geometry, direction, plus, cross, angle):
    times, n, lengths, senders, receivers = geometry
    cosine, sine = jnp.cos(2.*angle), jnp.sin(2.*angle)
    plus, cross = (_sub(_mul(cosine,plus), _mul(sine,cross)),
                   _add(_mul(sine,plus), _mul(cosine,cross)))
    a = jnp.zeros(n.shape[:2], dtype=jnp.float64); b = jnp.zeros_like(a)
    for i in range(3):
        for j in range(3):
            a = _filter_add(a, _filter_mul(_filter_mul(n[...,i], plus[i,j]), n[...,j]))
            b = _filter_add(b, _filter_mul(_filter_mul(n[...,i], cross[i,j]), n[...,j]))
    emitted = _sub(_sub(times[None], lengths), _dot3(senders, direction))
    received = _sub(times[None], _dot3(receivers[FIRST], direction))
    denominator = _sub(1., _dot3(n, direction))
    projection = (jnp.concatenate((emitted, received)), a, b, denominator)
    valid = ((jnp.abs(jnp.sum(direction*direction)-1.) <= 1e-12)
             & jnp.all(jnp.isfinite(direction)) & jnp.all(jnp.isfinite(plus))
             & jnp.all(jnp.isfinite(cross)) & jnp.isfinite(angle)
             & jnp.all(denominator != 0.))
    return projection, valid


_project_kernel = jax.jit(_project)


def _links(projection, values, origin, dt, source_order, *, unroll=False):
    queries, a, b, denominator = projection
    half = (source_order+1)//2
    normalized = _div(_sub(queries, origin), dt)
    integer = jnp.floor(normalized)
    supported = (jnp.isfinite(normalized) & (integer-(half-1) >= 0)
                 & (integer+half < values.shape[-1]))
    fraction = _div(_sub(queries, _add(origin, _mul(integer, dt))), dt)
    lower, upper = fraction < 0., fraction >= 1.
    integer = jnp.where(lower, integer-1., jnp.where(upper, integer+1., integer))
    fraction = jnp.where(lower, fraction+1., jnp.where(upper, fraction-1., fraction))
    upper = fraction >= 1.
    integer, fraction = jnp.where(upper, integer+1., integer), jnp.where(upper, fraction-1., fraction)
    supported &= (integer-(half-1) >= 0) & (integer+half < values.shape[-1])
    polarized = _paired_interpolate(values, integer.astype(jnp.int64), fraction,
        _scale(fraction, half, unroll=unroll), source_order, unroll=unroll)
    hp, hc = polarized[...,0,:,:], polarized[...,1,:,:]
    sender = _add(_mul(a, hp[...,:6,:]), _mul(b, hc[...,:6,:]))
    receiver = _add(_mul(a, hp[...,6+RECEIVERS,:]), _mul(b, hc[...,6+RECEIVERS,:]))
    links = _div(_mul(.5, _sub(sender, receiver)), denominator)
    valid = (jnp.all(supported) & jnp.isfinite(origin) & jnp.isfinite(dt) & (dt > 0.)
             & jnp.all(jnp.isfinite(links)))
    return links, valid


@lru_cache(maxsize=64)
def _response_kernel(generation, measurement_order, source_order, frequency, links_only=False, *, unroll=False):
    @jax.jit
    def response(geometry, parameters, values, origin, dt, direction, plus, cross, angle):
        projection, valid = _project(geometry, direction, plus, cross, angle)
        links, supported = _links(projection, values, origin, dt, source_order, unroll=unroll)
        valid &= supported
        result = links if links_only else _tdi(links, parameters, generation, measurement_order, frequency, unroll=unroll)
        valid &= jnp.all(jnp.isfinite(result))
        # Invalid support never silently returns a clipped/zero-extrapolated waveform.
        return jnp.where(valid, result, jnp.nan), valid
    return response


@lru_cache(maxsize=64)
def _staged_response_kernel(generation, measurement_order, source_order, frequency, links_only=False):
    """Separate device calls avoid costly CPU fusion across response stages.

    Arrays remain in JAX; there is no intermediate host conversion or wait.
    An enclosing JIT may compose these stages into one executable again.
    """
    project=jax.jit(_project)
    links=jax.jit(lambda q,v,o,d:_links(q,v,o,d,source_order))
    tdi=jax.jit(lambda v,p:_tdi(v,p,generation,measurement_order,frequency))
    @jax.jit
    def finish(result, valid):
        valid &= jnp.all(jnp.isfinite(result))
        return jnp.where(valid,result,jnp.nan),valid
    def response(geometry,parameters,values,origin,dt,direction,plus,cross,angle):
        projection,valid=project(geometry,direction,plus,cross,angle)
        link_values,supported=links(projection,values,origin,dt)
        result=link_values if links_only else tdi(link_values,parameters)
        return finish(result,valid & supported)
    return response


def _execution_schedule(schedule):
    if schedule not in ('staged','fused','fused_unrolled'):
        raise ValueError('execution_schedule must be staged, fused or fused_unrolled')
    return schedule


@dataclass(frozen=True, eq=False)
class JAXDetectorResponse:
    """One padded detector block; no implicit whole-year cache allocation.

    Setup validates host geometry once, then retains only immutable device
    arrays. It uses the active/default JAX device, with no explicit CPU pin.
    For a NumPy-prepared detector, ``from_detector`` uploads its validated plan.
    """
    geometry: tuple
    parameters: tuple
    dt: float
    generation: int
    measurement_order: int
    delay_order: int
    geometry_prescription: str
    preparation_backend: str
    execution_schedule: str = 'staged'

    @classmethod
    def prepare(cls, geometry, *, generation=2, measurement_order=25, delay_order=7,
                delay_derivatives=None, geometry_prescription='caller-supplied',
                execution_schedule='staged', device=None):
        _require_x64()
        execution_schedule=_execution_schedule(execution_schedule)
        times, dt = validate_times(geometry['times'])
        n, lengths, senders, receivers = [real_array(geometry[k], name=k)
            for k in ('n', 'length', 'p_sender', 'p_receiver')]
        if generation not in (1,2) or isinstance(generation,bool):
            raise ValueError('generation must be 1 or 2')
        measurement_order, delay_order = check_order(measurement_order), check_order(delay_order)
        shape = (6,times.size,3)
        if n.shape != shape or senders.shape != shape or receivers.shape != shape:
            raise ValueError('geometry vectors must have shape (6,N,3) in LINK_LABELS order')
        if lengths.shape != (6,times.size) or np.any(lengths < 0.):
            raise ValueError('nonnegative light times with shape (6,N) required')
        if not np.allclose(np.sum(n*n,axis=-1),1.,rtol=0,atol=1e-12):
            raise ValueError('arm directions must be unit vectors')
        for link,r in enumerate(RECEIVERS):
            if not np.array_equal(receivers[link], receivers[FIRST[r]]):
                raise ValueError('receiver reuse requires identical physical receiver positions')
        # Bounded inputs are needed by the compensated normal-range product.
        for a in (n,senders,receivers):
            if np.any(np.abs(a)>1e12) or np.any((a!=0.) & (np.abs(a)<1e-140)):
                raise ValueError('geometry is outside the validated normal-range dot-product domain')
        target=_device(device)
        arrays = tuple(jax.device_put(np.array(a,copy=True),target) for a in (times,n,lengths,senders,receivers))
        if delay_derivatives is None:
            rates = _rates(arrays[2], jnp.float64(dt))
        else:
            rates = real_array(delay_derivatives, name='delay derivatives')
            if rates.shape != lengths.shape: raise ValueError('rate shape must be (6,N)')
            rates = jax.device_put(np.array(rates,copy=True),target)
        parameters, shifts = _prepare_kernel(generation,measurement_order,delay_order)(
            arrays[2], rates, jnp.float64(dt))
        if not bool(jax.device_get(jnp.all(jnp.isfinite(shifts)))):
            raise ValueError('nonfinite compound detector shift')
        return cls(arrays,parameters,dt,generation,measurement_order,delay_order,
                   str(geometry_prescription),'jax',execution_schedule)

    def apply_links_array(self, links, *, unit='frequency'):
        """Six real link streams (...,6,N) to XYZ; no host copy or wait.

        Caller guarantees finite values. Nonfinite input propagates. The
        checked JAXTDIPlan.apply API validates values on the host.
        """
        return JAXTDIPlan(self.parameters,self.dt,self.generation,
            self.measurement_order,self.delay_order,self.size).apply_array(links,unit=unit)

    @classmethod
    def from_detector(cls, detector, *, execution_schedule='staged'):
        _require_x64()
        execution_schedule=_execution_schedule(execution_schedule)
        plan = detector.tdi
        device_plan = plan.to_jax()
        geometry = tuple(jnp.array(a, copy=True) for a in (detector.times,
            detector.directions,detector.lengths,detector.senders,detector.receivers))
        return cls(geometry,device_plan.parameters,plan.dt,plan.generation,
            plan.measurement_order,plan.delay_order,detector.geometry_prescription,'numba',execution_schedule)

    @property
    def size(self): return self.geometry[0].shape[0]

    @property
    def nbytes(self):
        """Resident array bytes; excludes compiler/interpolation workspaces."""
        return sum(a.nbytes for a in self.geometry+self.parameters)

    def project_array(self, direction, plus, cross, *, polarization_angle=0.):
        """Device arrays and validity scalar; no host synchronization."""
        _require_x64()
        direction, plus, cross = [jnp.asarray(a) for a in (direction,plus,cross)]
        if direction.shape != (3,) or plus.shape != (3,3) or cross.shape != (3,3):
            raise ValueError('one direction and two Cartesian (3,3) tensors required')
        if any(a.dtype != jnp.float64 for a in (direction,plus,cross)):
            raise ValueError('sky arrays must be float64')
        return _project_kernel(self.geometry,direction,plus,cross,jnp.float64(polarization_angle))

    def _sampled_array(self, values, origin_s, dt_s, direction, plus, cross, *,
                       order, polarization_angle, unit, links_only):
        _require_x64()
        order = check_order(order)
        if unit not in ('frequency','phase'): raise ValueError('unit must be frequency or phase')
        values, direction, plus, cross = [jnp.asarray(a) for a in (values,direction,plus,cross)]
        if values.ndim < 2 or values.shape[-2] != 2 or values.shape[-1] < order+1:
            raise ValueError('source values must have shape (...,2,M), covering the stencil')
        if direction.shape != (3,) or plus.shape != (3,3) or cross.shape != (3,3):
            raise ValueError('one direction and two Cartesian (3,3) tensors required')
        if any(a.dtype != jnp.float64 for a in (values,direction,plus,cross)):
            raise ValueError('source and sky arrays must be float64')
        if self.execution_schedule=='staged':
            kernel=_staged_response_kernel(self.generation,self.measurement_order,order,unit=='frequency',links_only)
        else:
            kernel=_response_kernel(self.generation,self.measurement_order,order,unit=='frequency',links_only,
                                    unroll=self.execution_schedule=='fused_unrolled')
        return kernel(
            self.geometry,self.parameters,values,jnp.float64(origin_s),jnp.float64(dt_s),
            direction,plus,cross,jnp.float64(polarization_angle))

    def sampled_response_array(self, values, origin_s, dt_s, direction, plus, cross, *,
                               order=39, polarization_angle=0., unit='frequency'):
        """JIT/vmap compatible response; returns ``(XYZ, valid)`` on device.

        Leading source batches share one sky/grid. Use vmap for changing skies.
        A false validity scalar accompanies all-NaN output. Callers composing
        likelihoods must propagate/reject that status, never ignore it.
        """
        return self._sampled_array(values,origin_s,dt_s,direction,plus,cross,order=order,
            polarization_angle=polarization_angle,unit=unit,links_only=False)

    def sampled_links_array(self, values, origin_s, dt_s, direction, plus, cross, *,
                            order=39, polarization_angle=0.):
        """As above, returning six GW fractional-frequency links."""
        return self._sampled_array(values,origin_s,dt_s,direction,plus,cross,order=order,
            polarization_angle=polarization_angle,unit='frequency',links_only=True)

    def sampled_response(self, sampled, direction, plus, cross, **kwargs):
        """Checked convenience API; synchronizes validity and raises on failure."""
        result, valid = self.sampled_response_array(sampled.values,sampled.origin_s,sampled.dt_s,
            direction,plus,cross,**kwargs)
        if not bool(jax.device_get(valid)):
            raise ValueError('invalid sky/alignment or insufficient/nonfinite source interpolation support')
        return result

    @staticmethod
    def xyz_to_aet(xyz):
        """Orthonormal A/E/T, matching the CPU convention, staying on device."""
        x,y,z = xyz[...,0,:],xyz[...,1,:],xyz[...,2,:]
        return jnp.stack(((z-x)/jnp.sqrt(2.),(x-2*y+z)/jnp.sqrt(6.),
                          (x+y+z)/jnp.sqrt(3.)),axis=-2)


@dataclass(frozen=True, eq=False)
class JAXStreamedDetectorResponse(HostStreamBase):
    """Host block/cache orchestration with JAX detector/response kernels.

    The supplied geometry provider can be an ordinary CPU ephemeris adapter.
    Cache bytes count device detector arrays; source arrays and compiler
    workspaces are additional. ``parallel`` must be false: JAX controls its
    own device scheduling, rather than a specified Numba worker count.
    Inherited ``sampled_response`` collects NumPy output under a byte limit;
    ``iter_sampled`` yields retained JAX blocks for a device-side consumer.
    """
    device: object = None
    execution_schedule: str = 'staged'

    def __post_init__(self):
        _require_x64()
        _execution_schedule(self.execution_schedule)
        if self.parallel:
            raise ValueError('JAX schedules its own device work; parallel=True is a Numba setting')
        super().__post_init__()

    def _entry(self, start, count):
        import time
        key=start,count
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key],True,0.,0.
        grid=self.times[0]+(start+np.arange(-self.halo,count+self.halo))*self.dt_s
        tick=time.perf_counter();geometry=self.geometry_provider(grid)
        if not np.array_equal(geometry['times'],grid):
            raise ValueError('geometry provider changed the requested absolute time grid')
        if np.any(np.asarray(geometry['length'])>self.delay_bound_s):
            raise ValueError('actual light time exceeds the declared padding bound')
        provider_s=time.perf_counter()-tick;tick=time.perf_counter()
        detector=JAXDetectorResponse.prepare(geometry,generation=self.generation,
            measurement_order=self.measurement_order,delay_order=self.delay_order,
            geometry_prescription=self.geometry_prescription,device=self.device,
            execution_schedule=self.execution_schedule)
        jax.block_until_ready(detector.parameters)
        plan_s=time.perf_counter()-tick
        if detector.nbytes<=self.cache_bytes:
            while self._cache and self.cached_bytes+detector.nbytes>self.cache_bytes:
                self._cache.popitem(last=False)
            self._cache[key]=detector
        return detector,False,provider_s,plan_s

    def _call_inputs(self, sampled, direction, plus, cross, polarization_angle):
        # Upload a fresh proposal once per call, rather than once per block.
        # No sky/polarization cache is retained. Preserve source/sky dtypes so
        # the checked array API still rejects float32 inputs.
        target=_device(self.device)
        values=jax.device_put(sampled.values,target)
        sky=tuple(jax.device_put(jnp.asarray(a),target) for a in (direction,plus,cross))
        grid=tuple(jax.device_put(jnp.float64(a),target)
                   for a in (sampled.origin_s,sampled.dt_s,polarization_angle))
        return (values,*grid,*sky)

    def iter_sampled(self, sampled, direction, plus, cross, *, order=39, polarization_angle=0.):
        import time
        stages=dict(detector_prepare_s=0.,provider_s=0.,tdi_plan_s=0.,response_s=0.,
                    source_transfer_s=0.,cache_hits=0,cache_misses=0,returned_samples=0,
                    cached_array_bytes=0)
        object.__setattr__(self,'last_stages',stages)
        tick=time.perf_counter()
        values,origin,dt,angle,direction,plus,cross=self._call_inputs(
            sampled,direction,plus,cross,polarization_angle)
        jax.block_until_ready((values,origin,dt,angle,direction,plus,cross))
        stages['source_transfer_s']=time.perf_counter()-tick
        for start in range(0,self.times.size,self.block_samples):
            count=min(self.block_samples,self.times.size-start);tick=time.perf_counter()
            detector,hit,provider_s,plan_s=self._entry(start,count)
            stages['detector_prepare_s']+=time.perf_counter()-tick
            stages['provider_s']+=provider_s;stages['tdi_plan_s']+=plan_s
            stages['cache_hits' if hit else 'cache_misses']+=1
            tick=time.perf_counter()
            result,valid=detector.sampled_response_array(values,origin,dt,
                direction,plus,cross,order=order,polarization_angle=angle)
            jax.block_until_ready((result,valid))
            if not bool(jax.device_get(valid)):
                raise ValueError('invalid sky/alignment or insufficient/nonfinite source interpolation support')
            result=result[...,self.halo:self.halo+count]
            result.block_until_ready()
            stages['response_s']+=time.perf_counter()-tick
            stages['returned_samples']+=count;stages['cached_array_bytes']=self.cached_bytes
            yield slice(start,start+count),result


    def iter_arrays(self, sampled, direction, plus, cross, *, order=39,
                    polarization_angle=0., max_inflight=2):
        """Yield (slice, device XYZ, validity) with bounded pending work.

        No per-block validity readback. Consumers must reject false validity;
        invalid responses are all NaN. Synchronization at the in-flight bound
        prevents unbounded queued work. Blocks retained by a consumer are the
        consumer's memory responsibility. Provider/cache orchestration is host
        code; sampled values and the response remain on the selected device.
        """
        from collections import deque
        if isinstance(max_inflight,bool) or not isinstance(max_inflight,int) or max_inflight<1:
            raise ValueError('max_inflight must be a positive integer')
        pending=deque()
        values,origin,dt,angle,direction,plus,cross=self._call_inputs(
            sampled,direction,plus,cross,polarization_angle)
        try:
            for start in range(0,self.times.size,self.block_samples):
                if len(pending)>=max_inflight:
                    jax.block_until_ready(pending.popleft())
                count=min(self.block_samples,self.times.size-start)
                detector,*_=self._entry(start,count)
                value,valid=detector.sampled_response_array(values,origin,dt,
                    direction,plus,cross,order=order,polarization_angle=angle)
                value=value[...,self.halo:self.halo+count]
                pending.append((value,valid))
                yield slice(start,start+count),value,valid
        finally:
            for task in pending: jax.block_until_ready(task)


@dataclass(frozen=True,eq=False)
class JAXTDIPlan:
    """Device-resident ordered TDI from caller-supplied six-link time series.

    Input times and light times are seconds. Labels are receiver/emitter in
    LINK_LABELS order. Frequency includes additive retarded-rate factors;
    phase omits them. Out-of-record link samples use the reference zero rule.
    Host validation occurs once at preparation, not in apply_array.
    """
    parameters: tuple
    dt: object
    generation: int
    measurement_order: int
    delay_order: int
    size: int

    @classmethod
    def prepare(cls,times,delays_s,*,delay_derivatives=None,generation=2,
                measurement_order=25,delay_order=7,device=None):
        _require_x64()
        times,dt=validate_times(times)
        if isinstance(generation,bool) or generation not in (1,2):
            raise ValueError('generation must be 1 or 2')
        mo,do=check_order(measurement_order),check_order(delay_order)
        delays=link_array(delays_s,times.size,name='delays')
        if delays.ndim!=2 or np.any(delays<0):
            raise ValueError('delays must be nonnegative and unbatched')
        target=_device(device);delays=jax.device_put(np.array(delays,copy=True),target)
        if delay_derivatives is None: rates=_rates(delays,jnp.float64(dt))
        else:
            rates=link_array(delay_derivatives,times.size,name='delay derivatives')
            if rates.ndim!=2: raise ValueError('delay derivatives must be unbatched')
            rates=jax.device_put(np.array(rates,copy=True),target)
        parameters,shifts=_prepare_kernel(generation,mo,do)(delays,rates,jnp.float64(dt))
        if not bool(jax.device_get(jnp.all(jnp.isfinite(shifts)))):
            raise ValueError('nonfinite compound detector shift')
        return cls(parameters,dt,generation,mo,do,times.size)

    @property
    def nbytes(self):return sum(a.nbytes for a in self.parameters)

    def apply_array(self,links,*,unit='frequency'):
        _require_x64();values=jnp.asarray(links)
        if values.ndim<2 or values.shape[-2:]!=(6,self.size):
            raise ValueError('link shape must be (...,6,N) in LINK_LABELS order')
        if values.dtype!=jnp.float64:raise ValueError('links must be float64')
        if unit not in ('frequency','phase'):raise ValueError('unit must be frequency or phase')
        return _apply_tdi_kernel(self.generation,self.measurement_order,unit=='frequency')(values,self.parameters)

    def apply(self,links,*,unit='frequency'):
        return self.apply_array(link_array(links,self.size,name='links'),unit=unit)


@lru_cache(maxsize=16)
def _apply_tdi_kernel(generation,order,frequency):
    return jax.jit(lambda values,parameters:_tdi(values,parameters,generation,order,frequency))


# Large detector arrays are dynamic leaves, never hidden outer-JIT constants.
jax.tree_util.register_dataclass(JAXDetectorResponse,
    data_fields=['geometry','parameters','dt'],
    meta_fields=['generation','measurement_order','delay_order','geometry_prescription',
                 'preparation_backend','execution_schedule'])
jax.tree_util.register_dataclass(JAXTDIPlan,data_fields=['parameters','dt'],
    meta_fields=['generation','measurement_order','delay_order','size'])
