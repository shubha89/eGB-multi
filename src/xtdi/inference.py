"""Independent detector, sky and waveform layers for repeated inference calls.

Geometry is Cartesian light-seconds; times/light times are seconds in one
common coordinate. The caller supplies the emitter prescription. Preparation
retains the existing ordered/additive-rate Michelson graph and float64.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

from .delay import real_array, readonly, check_order
from .michelson import LINK_LABELS
from .optimized import prepare_michelson, apply_michelson, LinkProjection
from .optimized.links import FIRST, RECEIVERS
from .optimized._stencils import coefficients, prepare_queries
from .optimized._application import evaluate_samples
from .optimized._sky import (TILE, sky_serial, sky_parallel,
                             links_serial, links_parallel, dot_probe)


@lru_cache(maxsize=1)
def _numpy_dot_compatible():
    """Enable the fused arithmetic only after matching the host NumPy kernel."""
    rng = np.random.default_rng(20261104)
    vectors = np.ascontiguousarray(rng.normal(size=(1024, 3))*500.)
    for direction in rng.normal(size=(12, 3)):
        direction = np.ascontiguousarray(direction/np.linalg.norm(direction))
        expected = np.einsum('lti,i->lt', vectors.reshape(4, 256, 3), direction).ravel()
        if not np.array_equal(dot_probe(vectors, direction).view(np.uint64), expected.view(np.uint64)):
            return False
    return True


def _basis(direction, plus, cross, angle):
    direction = real_array(direction, name='propagation direction')
    plus = real_array(plus, name='plus tensor')
    cross = real_array(cross, name='cross tensor')
    if direction.shape != (3,) or not np.isclose(direction@direction, 1., rtol=0, atol=1e-12):
        raise ValueError('propagation direction must be a Cartesian unit vector')
    if plus.shape != (3, 3) or cross.shape != (3, 3) or not np.isfinite(angle):
        raise ValueError('finite (3,3) transverse tensors and polarization angle required')
    c, s = np.cos(2*angle), np.sin(2*angle)
    plus, cross = c*plus-s*cross, s*plus+c*cross
    return direction, plus, cross


@dataclass(frozen=True, eq=False)
class DetectorResponse:
    """Immutable detector-only state, reusable for arbitrary source skies.

    The retained arrays are owned/read-only so updates to caller geometry do
    not silently invalidate the prepared TDI. A fixed source-grid origin is
    deliberately NOT part of detector state: each sampled call supplies it.
    For annual use, create one instance per padded block under an explicit
    storage budget. This object does not silently cache a whole observation.
    """
    times: np.ndarray
    directions: np.ndarray
    lengths: np.ndarray
    senders: np.ndarray
    receivers: np.ndarray
    tdi: object
    geometry_prescription: str
    compiled_sky_compatible: bool

    @classmethod
    def prepare(cls, geometry, *, generation=2, measurement_order=25,
                delay_order=7, delay_derivatives=None,
                geometry_prescription='caller-supplied', parallel=False):
        times = real_array(geometry['times'], name='reception times')
        n = real_array(geometry['n'], name='arm directions')
        lengths = real_array(geometry['length'], name='light times')
        senders = real_array(geometry['p_sender'], name='emitter positions')
        receivers = real_array(geometry['p_receiver'], name='receiver positions')
        if times.ndim != 1 or times.size < 2:
            raise ValueError('at least two uniform detector samples required')
        shape = (6, times.size, 3)
        if n.shape != shape or senders.shape != shape or receivers.shape != shape:
            raise ValueError('geometry vectors must have shape (6,N,3) in LINK_LABELS order')
        if lengths.shape != (6, times.size) or np.any(lengths < 0):
            raise ValueError('nonnegative light times with shape (6,N) required')
        if not np.allclose(np.sum(n*n, axis=-1), 1., rtol=0, atol=1e-12):
            raise ValueError('arm directions must be unit vectors')
        for link, r in enumerate(RECEIVERS):
            if not np.array_equal(receivers[link], receivers[FIRST[r]]):
                raise ValueError('receiver reuse requires identical physical receiver positions')
        plan = prepare_michelson(times, lengths, generation=generation,
                                 measurement_order=measurement_order,
                                 delay_order=delay_order,
                                 delay_derivatives=delay_derivatives, parallel=parallel)
        return cls(plan.times, *(readonly(a) for a in (n, lengths, senders, receivers)),
                   plan, str(geometry_prescription), _numpy_dot_compatible())

    @property
    def size(self):
        return self.times.size

    @property
    def nbytes(self):
        return self.tdi.nbytes + sum(a.nbytes for a in
            (self.directions, self.lengths, self.senders, self.receivers))

    def sky(self, direction, plus, cross, *, polarization_angle=0., parallel=False):
        """Prepare the actual new sky without evaluating detector providers.

        Returned LinkProjection is compatible with direct polarization
        callbacks and with prepared source-query interpolation. The fused
        sampled path below avoids allocating this intermediate object.
        """
        direction, plus, cross = _basis(direction, plus, cross, polarization_angle)
        if not self.compiled_sky_compatible:
            geometry = dict(times=self.times, n=self.directions, length=self.lengths,
                            p_sender=self.senders, p_receiver=self.receivers)
            return LinkProjection.prepare(direction, plus, cross, geometry)
        queries = np.empty((9, self.size))
        a, b, d = (np.empty((6, self.size)) for _ in range(3))
        errors = np.empty((self.size+TILE-1)//TILE, dtype=np.uint8)
        kernel = sky_parallel if parallel else sky_serial
        kernel(self.times, self.directions, self.lengths, self.senders,
               self.receivers, FIRST, direction, plus, cross, queries, a, b, d, errors)
        if np.any(errors):
            raise ValueError('exact arm/propagation alignment requires a separately evaluated limiting response')
        for array in (queries, a, b, d):
            array.flags.writeable = False
        receiver = queries[6:][RECEIVERS]
        receiver.flags.writeable = False
        return LinkProjection(queries[:6], receiver, a, b, d, queries.ravel())

    def sampled_links(self, sampled, direction, plus, cross, *, order=39,
                      polarization_angle=0., parallel=False, output=None):
        """New sky -> interpolation -> six links, with only tile-local scratch.

        The sampled values must be float64, C-contiguous and read-only, as
        returned by SampledPolarizations.from_arrays. Values can have leading
        batch dimensions (...,2,N); each member gets its own six link streams.
        A new source grid is checked on every call; no sky/query cache is used.
        """
        order = check_order(order)
        direction, plus, cross = _basis(direction, plus, cross, polarization_angle)
        values = sampled.values
        if (not isinstance(values, np.ndarray) or values.dtype != np.float64
                or values.ndim < 2 or values.shape[-2] != 2
                or not values.flags.c_contiguous or values.flags.writeable):
            raise ValueError('validated immutable contiguous float64 polarizations (...,2,N) required')
        origin, dt = float(sampled.origin_s), float(sampled.dt_s)
        if not np.isfinite(origin) or not np.isfinite(dt) or dt <= 0 or values.shape[-1] < order+1:
            raise ValueError('finite source origin, positive cadence and covered source grid required')
        shape = values.shape[:-2]+(6, self.size)
        if output is None:
            output = np.empty(shape, dtype=np.float64)
        elif (not isinstance(output, np.ndarray) or output.shape != shape
              or output.dtype != np.float64 or not output.flags.c_contiguous
              or not output.flags.writeable or np.shares_memory(output, values)):
            raise ValueError('output must be a separate writable contiguous float64 (...,6,N) array')
        if not self.compiled_sky_compatible:
            projection = self.sky(direction, plus, cross)
            query = prepare_queries(origin, dt, values.shape[-1], projection.query_times, order=order)
            output[...] = projection.apply(lambda q: evaluate_samples(sampled, query))
            return output
        errors = np.empty((self.size+TILE-1)//TILE, dtype=np.uint8)
        kernel = links_parallel if parallel else links_serial
        kernel(self.times, self.directions, self.lengths, self.senders,
               self.receivers, FIRST, RECEIVERS, direction, plus, cross,
               origin, dt, values.reshape((-1, 2, values.shape[-1])),
               coefficients(order), output.reshape((-1, 6, self.size)), errors)
        if np.any(errors == 1):
            raise ValueError('exact arm/propagation alignment requires a separately evaluated limiting response')
        if np.any(errors):
            raise ValueError('source samples do not cover the physical query interpolation stencil')
        if not np.all(np.isfinite(output)):
            raise ValueError('nonfinite waveform/link response')
        return output

    def apply_links(self, links, *, unit='frequency', return_stages=False, parallel=False):
        """Apply the unchanged ordered sampled Michelson graph."""
        return apply_michelson(self.tdi, links, unit=unit, return_stages=return_stages, parallel=parallel)

    def sampled_response(self, sampled, direction, plus, cross, *, order=39,
                         polarization_angle=0., parallel=False):
        links = self.sampled_links(sampled, direction, plus, cross, order=order,
                                   polarization_angle=polarization_angle,
                                   parallel=parallel)
        return self.apply_links(links, parallel=parallel)


@dataclass(frozen=True, eq=False)
class StreamedDetectorResponse:
    """Bounded detector-only cache with a new sky on every sampled call.

    The geometry provider receives absolute padded times and returns the
    Cartesian geometry contract of DetectorResponse.prepare. It is fixed for
    this instance; construct a new instance when orbit/ranging settings change.
    One mutable cache/work log belongs to one inference worker. Internal
    parallelism is over independent samples, not concurrent cache mutation.
    """
    times: object
    geometry_provider: object
    measurement_order: int = 25
    delay_order: int = 7
    generation: int = 2
    block_samples: int = 32768
    cache_bytes: int = 256*1024**2
    delay_bound_s: float = 12.
    padding_order: object = None
    parallel: bool = False
    geometry_prescription: str = 'caller-supplied'

    def __post_init__(self):
        from collections import OrderedDict
        from .michelson import validate_times
        from .streaming import conservative_halo
        times, dt = validate_times(self.times)
        measurement = check_order(self.measurement_order)
        delay = check_order(self.delay_order)
        padding = measurement if self.padding_order is None else check_order(self.padding_order)
        integer_fields = (self.block_samples, self.cache_bytes, self.generation)
        if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in integer_fields)
                or not callable(self.geometry_provider)
                or padding < measurement or self.block_samples < 1 or self.cache_bytes < 0
                or not np.isfinite(self.delay_bound_s) or self.delay_bound_s <= 0
                or self.generation not in (1, 2)):
            raise ValueError('invalid block, storage, delay bound or interpolation configuration')
        object.__setattr__(self, 'times', readonly(times))
        object.__setattr__(self, 'dt_s', dt)
        object.__setattr__(self, 'halo', conservative_halo(dt, self.delay_bound_s, padding, delay, self.generation))
        object.__setattr__(self, '_cache', OrderedDict())
        object.__setattr__(self, 'last_stages', {})

    @property
    def cached_bytes(self):
        return sum(value.nbytes for value in self._cache.values())

    def clear_cache(self):
        self._cache.clear()

    def _entry(self, start, count):
        import time
        key = start, count
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key], True, 0., 0.
        grid = self.times[0]+(start+np.arange(-self.halo, count+self.halo))*self.dt_s
        tick = time.perf_counter()
        geometry = self.geometry_provider(grid)
        if not np.array_equal(geometry['times'], grid):
            raise ValueError('geometry provider changed the requested absolute time grid')
        lengths = np.asarray(geometry['length'])
        if np.any(lengths > self.delay_bound_s):
            raise ValueError('actual light time exceeds the declared padding bound')
        provider_s = time.perf_counter()-tick
        tick = time.perf_counter()
        detector = DetectorResponse.prepare(geometry, generation=self.generation,
            measurement_order=self.measurement_order, delay_order=self.delay_order,
            geometry_prescription=self.geometry_prescription, parallel=self.parallel)
        plan_s = time.perf_counter()-tick
        if detector.nbytes <= self.cache_bytes:
            while self._cache and self.cached_bytes+detector.nbytes > self.cache_bytes:
                self._cache.popitem(last=False)
            self._cache[key] = detector
        return detector, False, provider_s, plan_s

    def iter_sampled(self, sampled, direction, plus, cross, *, order=39, polarization_angle=0.):
        """Yield retained (slice, XYZ) blocks with explicit complete stage clocks."""
        import time
        stages = dict(detector_prepare_s=0., provider_s=0., tdi_plan_s=0.,
                      sky_interpolation_links_s=0., tdi_s=0., cache_hits=0,
                      cache_misses=0, returned_samples=0, cached_array_bytes=0)
        object.__setattr__(self, 'last_stages', stages)
        for start in range(0, self.times.size, self.block_samples):
            count = min(self.block_samples, self.times.size-start)
            tick = time.perf_counter()
            detector, hit, provider_s, plan_s = self._entry(start, count)
            stages['detector_prepare_s'] += time.perf_counter()-tick
            stages['provider_s'] += provider_s
            stages['tdi_plan_s'] += plan_s
            stages['cache_hits' if hit else 'cache_misses'] += 1
            tick = time.perf_counter()
            links = detector.sampled_links(sampled, direction, plus, cross,
                order=order, polarization_angle=polarization_angle, parallel=self.parallel)
            stages['sky_interpolation_links_s'] += time.perf_counter()-tick
            tick = time.perf_counter()
            value = detector.apply_links(links, parallel=self.parallel)
            stages['tdi_s'] += time.perf_counter()-tick
            stages['returned_samples'] += count
            stages['cached_array_bytes'] = self.cached_bytes
            yield slice(start, start+count), value[..., self.halo:self.halo+count]

    def sampled_response(self, sampled, direction, plus, cross, *, order=39,
                         polarization_angle=0., output=None, max_output_bytes=512*1024**2):
        import math
        shape = sampled.values.shape[:-2]+(3, self.times.size)
        if output is None:
            if math.prod(shape)*8 > max_output_bytes:
                raise ValueError('full output exceeds memory limit; consume iter_sampled instead')
            output = np.empty(shape, dtype=np.float64)
        elif (not isinstance(output, np.ndarray) or output.shape != shape
                or output.dtype != np.float64 or not output.flags.writeable
                or np.shares_memory(output, sampled.values)):
            raise ValueError('output must be a separate writable float64 XYZ array')
        for index, block in self.iter_sampled(sampled, direction, plus, cross,
                order=order, polarization_angle=polarization_angle):
            output[..., index] = block
        return output
