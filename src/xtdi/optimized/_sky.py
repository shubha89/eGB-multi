"""Strict changing-sky kernels. No orbit or waveform model is evaluated here.

Each sample retains the existing physical-time expressions, Lagrange weight
recurrence and ascending tap sums. Parallel work is only between time tiles.
"""
import numpy as np
from numba import njit, prange, types
from numba.extending import intrinsic
from llvmlite import ir
from .._dot_policy import numpy_dot_uses_fma

_HOST_DOT_USES_FMA = numpy_dot_uses_fma()

TILE = 32


@intrinsic
def _fma64(typing_context, a, b, c):
    signature = types.float64(types.float64, types.float64, types.float64)
    def codegen(context, builder, sig, args):
        operation = builder.module.globals.get('llvm.fma.f64')
        if operation is None:
            operation = ir.Function(builder.module, ir.FunctionType(ir.DoubleType(), [ir.DoubleType()]*3), name='llvm.fma.f64')
        return builder.call(operation, args)
    return signature, codegen


@njit(cache=True, fastmath=False, inline='always')
def _dot3(vector, direction):
    # Preserve the independent host NumPy expression, including whether its
    # first sum contracts. The wrapper still verifies compatibility before
    # enabling this compiled path. No relaxed fastmath is enabled.
    first = vector[0]*direction[0]
    if _HOST_DOT_USES_FMA:
        first = _fma64(vector[2], direction[2], first)
    else:
        first = first + vector[2]*direction[2]
    return first + vector[1]*direction[1]


@njit(cache=True, fastmath=False, inline='always')
def _point(time, link, n, lengths, sender, receiver, direction, plus, cross):
    a = 0.0
    b = 0.0
    for i in range(3):
        for j in range(3):
            a += (n[link, time, i]*plus[i, j])*n[link, time, j]
            b += (n[link, time, i]*cross[i, j])*n[link, time, j]
    return a, b, 1.0-_dot3(n[link, time], direction)


@njit(cache=True, fastmath=False)
def _sky_tile(start, stop, times, n, lengths, sender, receiver, first,
              direction, plus, cross, queries, a, b, denominator, errors, tile):
    invalid = 0
    for t in range(start, stop):
        for link in range(6):
            aa, bb, dd = _point(t, link, n, lengths, sender, receiver,
                               direction, plus, cross)
            a[link, t] = aa
            b[link, t] = bb
            denominator[link, t] = dd
            queries[link, t] = (times[t]-lengths[link, t])-_dot3(sender[link, t], direction)
            if dd == 0.0:
                invalid = 1
        for r in range(3):
            queries[6+r, t] = times[t]-_dot3(receiver[first[r], t], direction)
    errors[tile] = invalid


def _sky_parallel_impl(times, n, lengths, sender, receiver, first, direction, plus, cross,
              queries, a, b, denominator, errors):
    size = times.size
    for tile in prange((size+TILE-1)//TILE):
        start = tile*TILE
        _sky_tile(start, min(start+TILE, size), times, n, lengths, sender,
                  receiver, first, direction, plus, cross,
                  queries, a, b, denominator, errors, tile)


def _sky_serial_impl(times, n, lengths, sender, receiver, first, direction, plus, cross,
                     queries, a, b, denominator, errors):
    size = times.size
    for tile in range((size+TILE-1)//TILE):
        start = tile*TILE
        _sky_tile(start, min(start+TILE, size), times, n, lengths, sender,
                  receiver, first, direction, plus, cross,
                  queries, a, b, denominator, errors, tile)


sky_serial = njit(cache=True, fastmath=False)(_sky_serial_impl)
sky_parallel = njit(cache=True, fastmath=False, parallel=True)(_sky_parallel_impl)


@njit(cache=True, fastmath=False, inline='always')
def _source_coordinate(query, origin, dt, source_size, half):
    normalized = (query-origin)/dt
    if not np.isfinite(normalized):
        return 0, 0.0, False
    integer = np.floor(normalized)
    if integer-(half-1) < 0 or integer+half >= source_size:
        return 0, 0.0, False
    fraction = (query-(origin+integer*dt))/dt
    if fraction < 0:
        integer -= 1.0
        fraction += 1.0
    elif fraction >= 1:
        integer += 1.0
        fraction -= 1.0
    if fraction >= 1:
        integer += 1.0
        fraction -= 1.0
    if integer-(half-1) < 0 or integer+half >= source_size:
        return 0, 0.0, False
    return int(integer), fraction, True


@njit(cache=True, fastmath=False)
def _links_tile(start, stop, times, n, lengths, sender, receiver, first, receivers,
                direction, plus, cross, origin, dt, values, coefficients,
                output, errors, tile):
    half = coefficients.size
    width = stop-start
    batch = values.shape[0]
    # Scratch belongs to this tile: no cross-thread reductions or shared writes.
    queries = np.empty((9, TILE))
    a = np.empty((6, TILE))
    b = np.empty((6, TILE))
    denominator = np.empty((6, TILE))
    polarized = np.empty((batch, 2, 9, TILE))
    weights = np.empty((2*half, TILE))
    fractions = np.empty(TILE)
    scales = np.empty(TILE)
    sides = np.empty(TILE)
    centers = np.empty(TILE, dtype=np.int64)
    accum = np.empty(TILE)
    invalid = 0
    for lane in range(width):
        t = start+lane
        for link in range(6):
            aa, bb, dd = _point(t, link, n, lengths, sender, receiver,
                               direction, plus, cross)
            a[link, lane] = aa
            b[link, lane] = bb
            denominator[link, lane] = dd
            queries[link, lane] = (times[t]-lengths[link, t])-_dot3(sender[link, t], direction)
            if dd == 0.0:
                invalid = 1
        for r in range(3):
            queries[6+r, lane] = times[t]-_dot3(receiver[first[r], t], direction)
    if invalid:
        errors[tile] = invalid
        return
    for q in range(9):
        for lane in range(width):
            center, fraction, valid = _source_coordinate(
                queries[q, lane], origin, dt, values.shape[-1], half)
            if not valid:
                invalid = 2
            centers[lane] = center
            fractions[lane] = fraction
            scale = 1.0
            for j in range(2, half):
                scale *= 1.0-(fraction/j)**2
            if half > 1:
                scale *= (1.0+fraction)*(1.0-fraction/half)
            scales[lane] = scale
            sides[lane] = fraction*(1.0-fraction)
        if invalid:
            errors[tile] = invalid
            return
        for j in range(1, half):
            for lane in range(width):
                fraction = fractions[lane]
                sides[lane] *= coefficients[j]
                weights[half-1-j, lane] = sides[lane]/(j+fraction)*scales[lane]
                weights[half+j, lane] = sides[lane]/(j+1.0-fraction)*scales[lane]
        for lane in range(width):
            weights[half-1, lane] = (1.0-fractions[lane])*scales[lane]
            weights[half, lane] = fractions[lane]*scales[lane]
        for member in range(batch):
            for pol in range(2):
                for lane in range(width):
                    accum[lane] = 0.0
                for tap in range(2*half):
                    for lane in range(width):
                        index = centers[lane]-(half-1)+tap
                        accum[lane] += weights[tap, lane]*values[member, pol, index]
                for lane in range(width):
                    polarized[member, pol, q, lane] = accum[lane]
    for member in range(batch):
        for link in range(6):
            r = 6+receivers[link]
            for lane in range(width):
                emitted = (a[link, lane]*polarized[member, 0, link, lane]
                           + b[link, lane]*polarized[member, 1, link, lane])
                received = (a[link, lane]*polarized[member, 0, r, lane]
                            + b[link, lane]*polarized[member, 1, r, lane])
                output[member, link, start+lane] = .5*(emitted-received)/denominator[link, lane]
    errors[tile] = 0


def _links_parallel_impl(times, n, lengths, sender, receiver, first, receivers,
                direction, plus, cross, origin, dt, values, coefficients,
                output, errors):
    size = times.size
    for tile in prange((size+TILE-1)//TILE):
        start = tile*TILE
        _links_tile(start, min(start+TILE, size), times, n, lengths, sender,
                    receiver, first, receivers, direction, plus, cross,
                    origin, dt, values, coefficients, output, errors, tile)


def _links_serial_impl(times, n, lengths, sender, receiver, first, receivers,
                       direction, plus, cross, origin, dt, values, coefficients,
                       output, errors):
    size = times.size
    for tile in range((size+TILE-1)//TILE):
        start = tile*TILE
        _links_tile(start, min(start+TILE, size), times, n, lengths, sender,
                    receiver, first, receivers, direction, plus, cross,
                    origin, dt, values, coefficients, output, errors, tile)


links_serial = njit(cache=True, fastmath=False)(_links_serial_impl)
links_parallel = njit(cache=True, fastmath=False, parallel=True)(_links_parallel_impl)


@njit(cache=True, fastmath=False)
def dot_probe(vectors, direction):
    output = np.empty(vectors.shape[0])
    for i in range(vectors.shape[0]):
        output[i] = _dot3(vectors[i], direction)
    return output
