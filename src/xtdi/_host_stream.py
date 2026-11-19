"""Backend-neutral bounded cache and NumPy output collector."""
from dataclasses import dataclass
import numpy as np
from .common import check_order, readonly

@dataclass(frozen=True, eq=False)
class HostStreamBase:
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
        from .common import validate_times
        from .common import conservative_halo
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
