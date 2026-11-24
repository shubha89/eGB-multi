"""Uniform, float64 Peters–Mathews source sampling with optional orbital interpolation."""
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from .pm_jax import prepare_pm, pm_strain


@partial(jax.jit, static_argnames=("size",))
def _direct(plan, origin, cadence, size):
    strain = pm_strain(plan, origin + cadence*jnp.arange(size, dtype=jnp.float64))
    return jnp.stack((strain.real, -strain.imag))


@lru_cache(maxsize=16)
def _weights(rows):
    # Round Chebyshev nodes to integer orbit counts. Every anchor and every
    # independent midpoint probe is an actual sample of the requested series.
    degree = 8
    nodes = np.rint((rows-1)*.5*(1-np.cos(np.arange(degree+1)*np.pi/degree))).astype(int)
    probes = (nodes[1:]+nodes[:-1])//2
    x = 2*nodes/(rows-1)-1
    basis = np.polynomial.chebyshev.chebvander(x, degree)
    weights = np.linalg.solve(basis.T, np.polynomial.chebyshev.chebvander(
        2*np.arange(rows)/(rows-1)-1, degree).T).T
    weights[nodes] = np.eye(degree+1)
    return nodes, probes, weights


@partial(jax.jit, static_argnames=("size", "period", "rows"))
def _anchors(plan, origin, cadence, size, period, rows):
    nodes, probes, weights = _weights(rows)
    width = period*rows
    groups = (size+width-1)//width
    # Shift the final block so that all its anchors are in bounds. Its
    # overlap with the previous block is discarded when producing the output.
    starts = jnp.minimum(jnp.arange(groups)*width, size-width)

    def sample(indices):
        idx = starts[:, None, None] + period*indices[None, :, None] + jnp.arange(period)
        h = pm_strain(plan, origin+cadence*idx.ravel()).reshape(groups, -1, period)
        return jnp.concatenate((h.real, -h.imag), axis=-1)

    h = sample(jnp.asarray(nodes))
    reference = sample(jnp.asarray(probes))
    residual = jnp.asarray(weights[probes]) @ h - reference
    peak = jnp.max(jnp.abs(h))
    # The renderer is a finite weighted sum. This bound guarantees that all
    # its partial sums fit in float64; invalid anchors/probes fail the check.
    safe_peak = np.finfo(float).max/(2*np.abs(weights).sum(axis=1).max())
    error = jnp.max(jnp.abs(residual))/peak
    return h, jnp.where((peak > 0) & (peak < safe_peak), error, jnp.inf)


@partial(jax.jit, static_argnames=("size", "period", "rows"))
def _render(anchors, size, period, rows):
    weights = jnp.asarray(_weights(rows)[2])
    width = period*rows
    tail = size-(anchors.shape[0]-1)*width

    def channel(h):
        out = weights[None, :, 0, None]*h[:, :1]
        for i in range(1, weights.shape[1]):
            out = out + weights[None, :, i, None]*h[:, i:i+1]
        return jnp.concatenate((out[:-1].ravel(), out[-1].ravel()[-tail:]))

    return jnp.stack((channel(anchors[..., :period]), channel(anchors[..., period:])))


def _orbital_grid(plan, cadence, size):
    # Limit angular drift per interpolation block *before* comparing direct
    # probes, to prevent undersampling two aliased phases that happen to agree.
    # n and Omega throughout the polynomial lie inside these bounds.
    low, high = np.asarray(plan.rate_bounds)
    e = float(plan.e_max)
    if plan.unwrap or e > .65:
        return None
    best = None
    mid = .5*(low+high)
    if not np.isfinite(mid*cadence) or mid*cadence < 2*np.pi/size:
        return None
    for turns in range(1, 13):
        period = round(turns*2*np.pi/(mid*cadence))
        if period < 1:
            continue
        slip = max(abs(low*cadence*period-turns*2*np.pi),
                   abs(high*cadence*period-turns*2*np.pi))
        # Account for more rapid angular motion near pericentre. The
        # numerical probe test below remains the actual acceptance test.
        pericentre_factor = np.sqrt(1+e)/(1-e)**1.5
        limit = min(size//period, 2048, .5/max(slip*pericentre_factor, 1e-16))
        if limit < 64:
            continue
        rows = 1 << int(np.floor(np.log2(limit)))
        if best is None or rows > best[1]:
            best = (period, rows)
    return best


def pm_uniform_polarizations(source, origin_s, dt_s, size, *, physics_mode="1pn",
                             method="direct", device=None, return_info=False,
                             **integration_options):
    """Return (2, size) device-resident h_plus and h_cross on a uniform grid.

    'direct' evaluates the orbit and quadrupole at every sample. 'interpolate'
    requests interpolation across nearly repeating orbits, with independent
    direct probes for every within-orbit sample offset in every block. Halve
    the block when the peak-normalized probe error exceeds 1e-11, then fall
    back to direct evaluation when the block becomes too short. This local
    convergence check is empirical. Use return_info=True to inspect the path.
    """
    if (isinstance(size, bool) or not isinstance(size, (int, np.integer))
            or size < 1 or not np.isfinite(dt_s) or dt_s <= 0
            or method not in {"direct", "interpolate"}):
        raise ValueError("supply a positive sample count and cadence, and a known sampling method")
    if isinstance(device, str):
        from .stride import _device
        device = _device(device)
    origin_s, dt_s = float(origin_s), float(dt_s)
    duration = (size-1)*dt_s
    end = origin_s + duration
    # A device may fuse i*dt+origin. Allow its final rounding at the upper
    # endpoint, including origins that nearly cancel the duration.
    if size > 1:
        end = np.nextafter(end + abs(np.spacing(duration)), np.inf)
    plan = prepare_pm(source, origin_s, end, physics_mode=physics_mode, device=device,
                      **integration_options)
    info = dict(method="direct", segments=len(plan.coefficients))
    result = None
    grid = _orbital_grid(plan, dt_s, size) if method == "interpolate" else None
    if grid is not None:
        period, rows = grid
        while rows >= 64:
            h, error = _anchors(plan, origin_s, dt_s, size, period, rows)
            error = float(error)
            if error <= 1e-11:
                info.update(method="interpolate", period_samples=period, rows=rows,
                            peak_relative_probe_error=error)
                result = _render(h, size, period, rows)
                break
            rows //= 2
    if result is None:
        result = _direct(plan, origin_s, dt_s, size)
        if not bool(jnp.all(jnp.isfinite(result))):
            raise ValueError("nonfinite sampled PM polarizations")
    return (result, info) if return_info else result
