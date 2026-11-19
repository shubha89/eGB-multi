"""Parallelize independent sample chunks of the established prefix recurrence."""
import numpy as np
from numba import njit, prange
from ._preparation import _interpolate_pair
from ._stencils import _scale

CHUNK = 1024


@njit(cache=True, fastmath=False, parallel=True)
def prefix_states_parallel(delays, rates, parents, atoms, depths, fs, side_coefficients):
    nodes = parents.size
    size = delays.shape[-1]
    shifts = np.zeros((nodes, size))
    retarded_rates = np.zeros((nodes, size))
    negative_rates = -rates
    negative_delays = -delays
    chunks = (size+CHUNK-1)//CHUNK
    errors = np.zeros(chunks, dtype=np.uint8)
    for chunk in prange(chunks):
        first = chunk*CHUNK
        last = min(first+CHUNK, size)
        weights = np.empty(2*side_coefficients.size)
        invalid = 0
        # Parent-before-child ordering is retained within every time sample.
        # Interpolated atomic inputs are immutable and may cross chunk bounds.
        for node in range(1, nodes):
            parent = parents[node]
            atom = atoms[node]
            atom_rates = negative_rates[atom]
            atom_delays = negative_delays[atom]
            for i in range(first, last):
                if depths[node] == 1:
                    shifts[node, i] = 0.0+(-delays[atom, i])
                    retarded_rates[node, i] = 0.0+(-rates[atom, i])
                else:
                    scaled = shifts[parent, i]*fs
                    if not np.isfinite(scaled):
                        invalid = 1
                    else:
                        rate, shift = _interpolate_pair(atom_rates, atom_delays, i, scaled,
                                                       side_coefficients, weights)
                        retarded_rates[node, i] = retarded_rates[parent, i]+rate
                        shifts[node, i] = shifts[parent, i]+shift
        errors[chunk] = invalid
    return shifts, retarded_rates, errors


@njit(cache=True, fastmath=False, parallel=True)
def terminal_stencils_parallel(shifts, indices, fs, half):
    """Independent terminal samples; retain each scalar scale recurrence."""
    operators = indices.size
    size = shifts.shape[-1]
    chunks = (size+CHUNK-1)//CHUNK
    centers = np.empty((operators, size), dtype=np.int64)
    fractions = np.empty((operators, size))
    scales = np.empty((operators, size))
    identities = np.ones((operators, chunks), dtype=np.uint8)
    errors = np.zeros((operators, chunks), dtype=np.uint8)
    for task in prange(operators*chunks):
        operator = task//chunks
        chunk = task%chunks
        first = chunk*CHUNK
        last = min(first+CHUNK, size)
        identity = 1
        invalid = 0
        for i in range(first, last):
            shift = shifts[indices[operator], i]*fs
            if not np.isfinite(shift):
                invalid = 1
                continue
            if shift != 0.:
                identity = 0
            integer = np.floor(shift)
            fraction = shift-integer
            center = min(max(float(i)+integer, -half-1.), size+half-1.)
            centers[operator, i] = int(center)
            fractions[operator, i] = fraction
            scales[operator, i] = _scale(fraction, half)
        identities[operator, chunk] = identity
        errors[operator, chunk] = invalid
    return centers, fractions, scales, identities, errors
