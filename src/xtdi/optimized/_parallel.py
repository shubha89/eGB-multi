"""Parallel independent query chunks, retaining the serial tap arithmetic."""
import numpy as np
from numba import njit, prange
from ._interpolation import apply_tiled

CHUNK = 1024


@njit(cache=True, fastmath=False, parallel=True)
def apply_parallel(data, centers, fractions, scales, side_coefficients):
    count = centers.size
    output = np.empty((data.shape[0], count))
    for block in prange((count+CHUNK-1)//CHUNK):
        first = block*CHUNK
        last = min(first+CHUNK, count)
        part = apply_tiled(data, centers[first:last], fractions[first:last],
                           scales[first:last], side_coefficients)
        for member in range(data.shape[0]):
            for j in range(last-first):
                output[member, first+j] = part[member, j]
    return output
