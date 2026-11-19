"""Bounded detector-plan caching and padded time-domain block execution."""

from collections import OrderedDict
from dataclasses import dataclass
import math

import numpy as np

from .michelson import prepare_michelson, validate_times


def conservative_halo(dt, delay_bound_s, measurement_order, delay_order, generation=2):
    # A retained TDI2 sample uses <=8 atomic light times and <=3 staged
    # measurement stencils. Four complete stencil widths additionally bound
    # the retarded delay-plan interpolation and endpoint derivative support.
    # This intentionally exceeds the minimum; tests compare contiguous data.
    return math.ceil((8 if generation == 2 else 4)*delay_bound_s/dt) + \
        4*(measurement_order+1)+4*(delay_order+1)+2


@dataclass(frozen=True, eq=False)
class StreamedMichelson:
    """Providers receive absolute padded times and return six aligned arrays.

    `delay_provider(times)` returns delay seconds, not source responses.
    `iter_apply(link_provider)` yields (retained slice, XYZ block). Providers
    must support the declared halo outside the retained observation. No edge
    extrapolation or padding of a physical source is silently introduced.
    """

    times: object
    delay_provider: object
    generation: int = 2
    measurement_order: int = 101
    delay_order: int = 7
    block_samples: int = 65536
    cache_bytes: int = 256*1024**2
    delay_bound_s: float = 12.
    backend: str = "numba"

    def __post_init__(self):
        times, dt = validate_times(self.times)
        times = times.copy()
        times.flags.writeable = False
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "dt", dt)
        if self.block_samples < 1 or self.cache_bytes < 0 or not np.isfinite(self.delay_bound_s) or self.delay_bound_s <= 0:
            raise ValueError("valid block size, nonnegative cache budget, and positive delay bound required")
        if self.backend not in ("numba", "jax"):
            raise ValueError("backend must be numba or jax")
        from .delay import check_order
        check_order(self.measurement_order); check_order(self.delay_order)
        if self.generation not in (1,2):
            raise ValueError("generation must be 1 or 2")
        object.__setattr__(self, "halo", conservative_halo(self.dt, self.delay_bound_s,
            self.measurement_order, self.delay_order, self.generation))
        object.__setattr__(self, "_cache", OrderedDict())

    def clear_cache(self):
        self._cache.clear()

    @property
    def cached_bytes(self):
        return sum(plan.nbytes for plan in self._cache.values())

    def _block(self, start, stop):
        key = start, stop
        grid = self.times[0]+np.arange(start-self.halo,stop+self.halo)*self.dt
        if key in self._cache:
            self._cache.move_to_end(key)
            return grid, self._cache[key]
        delays = self.delay_provider(grid)
        from .michelson import link_array
        checked = link_array(delays, grid.size, name="delays")
        if checked.ndim != 2 or np.any(checked < 0) or np.any(checked > self.delay_bound_s):
            raise ValueError("actual delay exceeds the declared delay bound or is invalid")
        plan = prepare_michelson(grid, checked, generation=self.generation,
            measurement_order=self.measurement_order, delay_order=self.delay_order)
        if self.backend == "jax":
            plan = plan.to_jax()
        if plan.nbytes <= self.cache_bytes:
            # Evict before retaining another plan; source arrays are never cached.
            while self._cache and self.cached_bytes+plan.nbytes > self.cache_bytes:
                self._cache.popitem(last=False)
            self._cache[key] = plan
        return grid, plan

    def iter_apply(self, link_provider, *, unit="frequency"):
        for start in range(0, self.times.size, self.block_samples):
            stop = min(start+self.block_samples, self.times.size)
            grid, plan = self._block(start, stop)
            result = plan.apply(link_provider(grid), unit=unit)
            yield slice(start,stop), result[...,self.halo:self.halo+stop-start]

    def apply_provider(self, link_provider, *, unit="frequency", max_output_bytes=512*1024**2):
        """Collect XYZ if it fits the limit; use iter_apply for large batches."""
        output = None
        for index, block in self.iter_apply(link_provider, unit=unit):
            if output is None:
                shape = block.shape[:-1]+(self.times.size,)
                if math.prod(shape)*8 > max_output_bytes:
                    raise ValueError("full output exceeds memory limit; consume iter_apply instead")
                output = np.empty(shape, dtype=np.float64)
            output[...,index] = np.asarray(block)
        return output
