"""Experimental batched, double-precision PM sampling on one uniform source grid."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import pm_jax as pm
from .eccentric_jax import pack_eccentric_sources
from .pm_sampling import pm_uniform_polarizations
from lisa_stride.sampled import SampledPolarizations
from xtdi.common import check_order
from xtdi.jax_response import _device, _require_x64


@partial(jax.jit, static_argnames=("size", "circular"))
def _one_segment_batch(parameters, origin, step, end, size, circular):
    times = origin + step * jnp.arange(size, dtype=jnp.float64)

    def sample(params):
        frequency, eccentricity, m1, m2 = params[:4]
        spans = jnp.stack((end - params[7], origin - params[7]))
        # Default element and phase tolerances of prepare_pm for the 1PN model.
        scales = jnp.array([1e-18 + 1e-10*frequency, 1e-14 + 1e-10*eccentricity, 1e-9, 1e-9])

        def refine(span):
            coarse, n0, omega0 = pm._integrate(params, span, advance=True, steps=1)
            fine, _, _ = pm._integrate(params, span, advance=True, steps=2)
            return coarse[0], n0, omega0, pm._refinement_error(coarse, fine, scales)

        coefficients, n0, omega0, error = jax.vmap(refine)(spans)
        hull = jnp.asarray(pm._BERNSTEIN) @ coefficients[:, :, :2]
        radial = n0[0] + hull[..., 0]
        ecc = eccentricity + hull[..., 1]
        n_min, n_max, e_max = radial.min(), radial.max(), ecc.max()
        mass = pm.G_M3_KG_S2 * ((m1 + m2) * pm.SOLAR_MASS_KG) / pm.C_M_PER_S**3
        omega_max = n_max * (1 + 3*(mass*n_max)**(2/3)/(1-e_max**2))
        root = jnp.sqrt(1 - e_max**2)
        beta_bound = e_max/(1+root) + 8*(mass*omega_max)**(2/3)*e_max/root
        # Use the same physical, anomaly-unwrapping and phase-reduction bounds
        # as prepare_pm. Other regimes take the ordinary adaptive route.
        eligible = ((spans[0] > 0) & (spans[1] < 0) & jnp.all(error <= 1)
                    & (3*(mass*frequency)**(2/3) < 1-eccentricity**2)
                    & (n_min > 0) & (ecc.min() >= 0) & (e_max < 1)
                    & (beta_bound < jnp.sqrt(0.5)) & (jnp.abs(spans).max()*omega_max < 2**26)
                    & (jnp.abs(jnp.array([origin, end, params[7]])).max() < 2**35))
        plan = pm.PMPlan(params, n0[0], omega0[0], coefficients, spans,
                         jnp.stack((origin, end)), "1pn", jnp.stack((n_min, omega_max)),
                         e_max, circular=circular, unwrap=False, short_phase=True)
        strain = pm.pm_strain(plan, times)
        return jnp.stack((strain.real, -strain.imag)), eligible & jnp.all(jnp.isfinite(strain))

    return jax.vmap(sample)(parameters)


def sampled_pm_batch(sources, origin_s, dt_s, size, *, order=39, device=None):
    """Return (B,2,size) JAX polarizations for independent full-PM, 1PN sources.

    The time grid is in barycentric seconds. Each source keeps its own epoch,
    parameters and checked evolution. One-segment solves use the default PM
    tolerances. If any source needs further refinement or a different phase
    treatment, the group goes through the scalar adaptive solver.
    """
    _require_x64()
    order = check_order(order)
    if (not isinstance(size, (int, np.integer)) or size < order+1 or
            not np.isfinite(origin_s) or not np.isfinite(dt_s) or dt_s <= 0):
        raise ValueError("Provide a finite uniform grid with positive cadence and full stencil support")
    sources = tuple(sources)
    if not sources:
        raise ValueError("At least one source is required")
    params = pack_eccentric_sources(sources)
    if (not np.all(np.isfinite(params)) or np.any(params[:, [0, 2, 3, 4]] <= 0)
            or np.any((params[:, 1] < 0) | (params[:, 1] >= 1)) or np.any(params[:, 8] != 0)):
        raise ValueError("Invalid PM parameters (requires fdot=0)")
    params[:, 6] = np.arctan2(np.sin(params[:, 6]), np.cos(params[:, 6]))
    target = _device(device)
    origin, step = (jax.device_put(np.float64(x), target) for x in (origin_s, dt_s))
    duration = (size-1) * float(dt_s)
    end = np.nextafter(float(origin_s) + duration + abs(np.spacing(duration)), np.inf)
    if not np.isfinite(end):
        raise ValueError("Source grid endpoint must be finite")
    parts, indices, flags = [], [], []
    for circular in (True, False):
        group = np.flatnonzero((params[:, 1] == 0) == circular)
        if group.size:
            values, valid = _one_segment_batch(jax.device_put(params[group], target),
                                               origin, step, end, size, circular)
            parts.append(values)
            indices.extend(group)
            flags.append(jnp.all(valid))
    if bool(jnp.all(jnp.stack(flags))):
        values = parts[0] if len(parts) == 1 else jnp.concatenate(parts)[np.argsort(indices)]
    else:
        values = jnp.stack([pm_uniform_polarizations(s, origin_s, dt_s, size, device=target)
                            for s in sources])
    return SampledPolarizations(origin, step, values, order)
