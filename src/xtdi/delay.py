"""Centered Lagrange interpolation, with pyTDI-compatible sampling boundaries.

For shift s=m+f in samples and h=(order+1)/2, apply the Lagrange
polynomial on offsets -h+1,...,h about n+m. Positive s advances the
input; physical delays therefore have negative s. Out-of-record input is
zero. The index clipping matches pyTDI's variable-delay implementation.

The compact coefficient recurrence is adapted from pyTDI's Lagrange
recurrence; see THIRD_PARTY_NOTICES. No reference code is called at runtime.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit


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


@njit(cache=True, fastmath=False)
def _apply(data, centers, fractions, scales, side_coefficients):
    # One small stencil is reused across all samples and all source batches.
    # No N x taps window or coefficient matrix is materialized.
    batch, size = data.shape
    query_size = centers.size
    half = side_coefficients.size
    weights = np.empty(2 * half, dtype=np.float64)
    output = np.empty((batch, query_size), dtype=np.float64)
    for i in range(query_size):
        fraction = fractions[i]
        scale = scales[i]
        side = fraction * (1.0 - fraction)
        for j in range(1, half):
            side *= side_coefficients[j]
            weights[half - 1 - j] = side / (j + fraction) * scale
            weights[half + j] = side / (j + 1.0 - fraction) * scale
        weights[half - 1] = (1.0 - fraction) * scale
        weights[half] = fraction * scale
        first = centers[i] - (half - 1)
        for b in range(batch):
            value = 0.0
            for k in range(2 * half):
                index = first + k
                if 0 <= index < size:
                    value += weights[k] * data[b, index]
            output[b, i] = value
    return output


@dataclass(frozen=True, eq=False)
class FractionalDelay:
    """Reusable stencil for one fixed shift vector, in dimensionless samples."""

    centers: np.ndarray
    fractions: np.ndarray
    scales: np.ndarray
    side_coefficients: np.ndarray
    order: int
    identity: bool

    @classmethod
    def prepare(cls, shifts, size, order=101):
        order = check_order(order)
        if not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("size must be a positive integer")
        shifts = real_array(shifts, name="shifts")
        if shifts.size == 1:
            shifts = np.full(size, shifts.item())
        if shifts.shape != (size,):
            raise ValueError("shifts must be scalar or have one value per sample")
        half = (order + 1) // 2
        integer = np.floor(shifts)
        fraction = shifts - integer
        # Clip before converting to integers, including extremely large shifts.
        centers = np.clip(np.arange(size, dtype=np.float64) + integer,
                          -half - 1, size + half - 1).astype(np.int64)
        scales = np.ones(size)
        for j in range(2, half):
            scales *= 1.0 - (fraction / j) ** 2
        if half > 1:
            scales *= (1.0 + fraction) * (1.0 - fraction / half)
        coefficients = np.zeros(half)
        for j in range(1, half):
            coefficients[j] = -(1.0 - j / half) / (1.0 + j / half)
        return cls(readonly(centers), readonly(fraction), readonly(scales),
                   readonly(coefficients), order, bool(np.all(shifts == 0)))

    @property
    def size(self):
        return self.centers.size

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.centers, self.fractions,
                                      self.scales, self.side_coefficients))

    def apply(self, data):
        values = real_array(data, name="data")
        if values.ndim < 1 or values.shape[-1] != self.size:
            raise ValueError("last data dimension must match the prepared sample grid")
        if self.identity:
            return values.copy()
        return self._apply_array(values)

    def _apply_array(self, values):
        if self.identity:
            return values.copy()
        shape = values.shape
        return _apply(values.reshape((-1, self.size)), self.centers,
                      self.fractions, self.scales, self.side_coefficients).reshape(shape)


def timeshift(data, shifts, order=101):
    """Shift real samples; positive shifts advance, negative shifts delay.

    The last axis is time; leading axes are independent streams. A singleton
    input reproduces pyTDI's constant-input special case.
    """
    check_order(order)
    values = real_array(data, name="data")
    if values.size == 1:
        real_array(shifts, name="shifts")
        return values.item()
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("data must contain samples")
    return FractionalDelay.prepare(shifts, values.shape[-1], order).apply(values)
