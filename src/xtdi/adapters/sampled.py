"""Uniformly sampled strain polarizations queried at physical wavefront times.

Times are seconds in the caller's common time coordinate. Values are real
dimensionless h_plus and h_cross in the caller's transverse basis; no time,
phase, polarization rotation or unit conversion is inferred. The interpolation
is the same centered odd-order Lagrange stencil used by xTDI's delay primitive.
Every stencil must be covered by the supplied source samples: no zero padding,
clipping or extrapolation is permitted at this source interface.
"""
from dataclasses import dataclass

import numpy as np

from ..delay import FractionalDelay, _apply, check_order, real_array, readonly


@dataclass(frozen=True, eq=False)
class UniformQueryPlan:
    """Compact, reusable interpolation at arbitrary (possibly unordered) times.

    Source-independent parameters can be reused for other polarizations on the
    identical origin/cadence/length grid. Leading input dimensions are batches.
    Output shape is input.shape[:-1] + query_shape.
    """
    origin_s: float
    dt_s: float
    source_size: int
    query_shape: tuple
    centers: np.ndarray
    fractions: np.ndarray
    scales: np.ndarray
    side_coefficients: np.ndarray
    order: int

    @classmethod
    def prepare(cls, origin_s, dt_s, source_size, query_times, *, order=101):
        order = check_order(order)
        origin_s, dt_s = float(origin_s), float(dt_s)
        if not np.isfinite(origin_s) or not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError('finite origin and positive finite cadence required')
        if (isinstance(source_size, (bool, np.bool_)) or
                not isinstance(source_size, (int, np.integer)) or source_size < order+1):
            raise ValueError('source_size must be an integer covering the interpolation stencil')
        queries = real_array(query_times, name='query times')
        if queries.size == 0:
            raise ValueError('at least one source query required')
        shape = queries.shape
        indices = ((queries-origin_s)/dt_s).ravel()
        if np.any(~np.isfinite(indices)):
            raise ValueError('nonfinite normalized source times')
        integer = np.floor(indices)
        half = (order+1)//2
        if np.any(integer-(half-1) < 0) or np.any(integer+half >= source_size):
            raise ValueError('source samples do not cover all query interpolation stencils; no extrapolation')
        # Form the fraction relative to the neighbouring physical grid time.
        # Subtracting a large normalized sample index loses precision for a
        # year-long source table. The query's original seconds are authoritative.
        fractions = (queries.ravel()-(origin_s+integer*dt_s))/dt_s
        lower = fractions < 0
        upper = fractions >= 1
        integer[lower] -= 1
        fractions[lower] += 1
        integer[upper] += 1
        fractions[upper] -= 1
        # Adding one to a tiny negative fraction can round to exactly one.
        # Rebase its center before FractionalDelay normalizes the fraction;
        # otherwise the query would select the preceding source sample.
        rounded_upper = fractions >= 1
        integer[rounded_upper] += 1
        fractions[rounded_upper] -= 1
        if np.any(integer-(half-1) < 0) or np.any(integer+half >= source_size):
            raise ValueError('source samples do not cover the corrected query stencil')
        # Reuse the existing recurrence preparation. Its centers are not used:
        # source indices and query indices have different origins and lengths.
        stencil = FractionalDelay.prepare(fractions, fractions.size, order)
        return cls(origin_s, dt_s, int(source_size), shape,
                   readonly(integer.astype(np.int64)), stencil.fractions,
                   stencil.scales, stencil.side_coefficients, order)

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.centers, self.fractions, self.scales,
                                      self.side_coefficients))

    def apply(self, samples):
        values = real_array(samples, name='source samples')
        if values.ndim < 1 or values.shape[-1] != self.source_size:
            raise ValueError('source sample length differs from the prepared grid')
        return self._apply_array(values)

    def _apply_array(self, values):
        """Apply already validated immutable samples without a full-table scan."""
        result = _apply(values.reshape((-1, self.source_size)), self.centers,
                        self.fractions, self.scales, self.side_coefficients)
        return result.reshape(values.shape[:-1]+self.query_shape)


@dataclass(frozen=True, eq=False)
class SampledPolarizations:
    """Immutable sampled h_plus/h_cross provider with an explicit time grid.

    Values have shape (2,N), ordered plus,cross. Copying into owned read-only
    storage prevents later mutation from invalidating a reused source table.
    Preparing/evaluating queries is separate so setup and application can be
    measured independently. The callable convenience path includes query setup.
    """
    origin_s: float
    dt_s: float
    values: np.ndarray
    order: int = 101

    @classmethod
    def from_arrays(cls, hplus, hcross, *, origin_s, dt_s, order=101):
        order = check_order(order)
        hp = real_array(hplus, name='hplus')
        hc = real_array(hcross, name='hcross')
        if hp.ndim != 1 or hc.shape != hp.shape or hp.size < order+1:
            raise ValueError('aligned one-dimensional plus/cross samples covering the stencil required')
        if not np.isfinite(origin_s) or not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError('finite origin and positive finite cadence required')
        return cls(float(origin_s), float(dt_s), readonly(np.stack((hp, hc))), order)

    def prepare_queries(self, times):
        return UniformQueryPlan.prepare(self.origin_s, self.dt_s, self.values.shape[-1],
                                        times, order=self.order)

    def evaluate(self, plan):
        if (plan.origin_s, plan.dt_s, plan.source_size) != (
                self.origin_s, self.dt_s, self.values.shape[-1]):
            raise ValueError('query plan belongs to a different source time grid')
        return plan._apply_array(self.values)

    def __call__(self, times):
        return self.evaluate(self.prepare_queries(times))
