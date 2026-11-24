"""Double-precision dense Peters–Mathews evolution for the eGB source.

Integrate changes in radial mean motion, time eccentricity, and the residual
radial/azimuthal phases after removing their constant initial rates. Time is
in seconds, frequencies are angular, and phases are in radians.
"""
from dataclasses import dataclass, field
from functools import partial
from math import comb

import jax
import jax.numpy as jnp
import numpy as np

from .constants import C_M_PER_S, G_M3_KG_S2, SOLAR_MASS_KG
from .eccentric_jax import _physics_switches, _strain_from_orbit, pack_eccentric_sources
from .phase import _phase_backend


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PMPlan:
    params: jax.Array
    n0: jax.Array
    omega0: jax.Array
    # Forward segments first, followed by backward segments (both from t0).
    coefficients: jax.Array
    spans: jax.Array
    bounds: jax.Array
    physics_mode: str = field(metadata={"static": True})
    rate_bounds: jax.Array
    e_max: jax.Array
    circular: bool = field(metadata={"static": True})
    unwrap: bool = field(metadata={"static": True})
    short_phase: bool = field(metadata={"static": True})


def _initial(params, advance):
    omega, e, m1, m2 = params[:4]
    mass_time = (G_M3_KG_S2 * ((m1 + m2) * SOLAR_MASS_KG)) / C_M_PER_S**3
    eta = m1 * m2 / (m1 + m2)**2
    k = 3 * (mass_time * omega)**(2/3) / (1 - e*e) if advance else 0.
    n = omega * jax.lax.optimization_barrier(1. - k)
    k_radial = 3 * (mass_time * n)**(2/3) / (1 - e*e) if advance else 0.
    # Preserve the same truncated forward/inverse 1PN maps as the SciPy path.
    return n, n*(1 + k_radial), n*k_radial, eta*mass_time**(5/3)


@partial(jax.jit, static_argnames=("advance", "steps"))
def _integrate(params, span, *, advance, steps):
    n0, omega0, precession_rate, mass_factor = _initial(params, advance)
    e0 = params[1]
    h = span / steps

    def rhs(state):
        dn, de = state[:2]
        n, e = n0 + dn, e0 + de
        e2 = e*e
        q = 1 - e2
        flux = mass_factor * n**(8/3)
        ndot = (96/5)*flux*n*(1 + (73/24)*e2 + (37/96)*e2**2)/q**3.5
        edot = -(304/15)*flux*e*(1 + (121/304)*e2)/q**2.5
        # omega = n + const*n^(5/3)/(1-e^2). Evaluate its small change
        # without subtracting two nearly identical orbital frequencies.
        domega = dn
        if advance:
            log_ratio = (5/3)*jnp.log1p(dn/n0) - jnp.log1p(-de*(e+e0)/(1-e0*e0))
            domega = dn + precession_rate*jnp.expm1(log_ratio)
        valid = (n > 0) & (e >= 0) & (e < 1)
        return jnp.where(valid, jnp.array([ndot, edot, dn, domega]), jnp.nan)

    def half_step(y, first):
        half = .5*h
        k2 = rhs(y + .5*half*first)
        k3 = rhs(y + .5*half*k2)
        k4 = rhs(y + half*k3)
        change = (half/6)*(first + 2*k2 + 2*k3 + k4)
        end = y + change
        return change, rhs(end)

    def step(carry, _):
        y, first = carry
        first_half, middle = half_step(y, first)
        second_half, last = half_step(y + first_half, middle)
        dy = first_half + second_half
        # Hermite polynomial matching values and rates at 0, 1/2, and 1.
        # Form the midpoint corrections from increments to avoid subtracting
        # accumulated phases. u^2*(1-u)^2 preserves both endpoint slopes.
        b = h*first
        c = 3*dy - h*(2*first+last)
        d = h*(first+last) - 2*dy
        value = 16*(first_half - .5*(b + .5*(c + .5*d)))
        slope = 16*(h*middle - (b+c+.75*d))
        a2 = value - .5*slope
        coeff = jnp.stack((y, b, c+a2, d+slope-2*a2, a2-2*slope, slope))
        return (y+dy, last), coeff

    start = jnp.zeros(4, dtype=params.dtype)
    _, coeff = jax.lax.scan(step, (start, rhs(start)), None, length=steps)
    return coeff, n0, omega0


def _polynomial(coeff, x):
    result = coeff[-1]
    for i in range(len(coeff)-2, -1, -1):
        result = result*x + coeff[i]
    return result


_DEGREE = 5
# Linear maps of power-series coefficients. Bernstein coefficients bound
# the polynomial on [0,1] through the convex hull property.
_BERNSTEIN = np.array([[comb(i,j)/comb(_DEGREE,j) if j <= i else 0.
                        for j in range(_DEGREE+1)] for i in range(_DEGREE+1)])


def _restrict_half(start):
    return np.array([[comb(j,i)*.5**i*start**(j-i) if j >= i else 0.
                      for j in range(_DEGREE+1)] for i in range(_DEGREE+1)])


@jax.jit
def _refinement_error(coarse, fine, scales):
    def bound(start, finer):
        # Remove the shared constant separately to retain accuracy at late
        # times. Bound the polynomial difference on both half segments.
        c = coarse.at[:, 0].set(0.)
        f = finer.at[:, 0].set(finer[:, 0]-coarse[:, 0])
        diff = _restrict_half(start) @ c - f
        return jnp.max(jnp.abs(_BERNSTEIN @ diff) / scales)
    return jnp.maximum(bound(0., fine[::2]), bound(.5, fine[1::2]))


def prepare_pm(source, start_s, end_s, *, physics_mode="1pn", device=None,
               pm_ode_rtol=1e-10, pm_ode_phase_atol=1e-9, max_steps=4096):
    """Solve once over a closed time interval. Reuse the plan for time blocks.

    Element refinement uses relative tolerances plus the SciPy absolute floors.
    Phase refinement uses an absolute tolerance in radians even at late epochs.
    Halve steps until two independent resolutions agree throughout each segment.
    Return the coarse solution that passed. Raise if refinement is exhausted.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Peters–Mathews evolution requires JAX float64")
    if isinstance(device, str):
        from .stride import _device
        device = _device(device)
    values = pack_eccentric_sources(source)[0]
    limits = np.array([start_s, end_s, pm_ode_rtol, pm_ode_phase_atol])
    if (not np.all(np.isfinite(values)) or not np.all(np.isfinite(limits))
            or not 0 <= source.eccentricity < 1 or np.any(values[[0, 2, 3, 4]] <= 0)
            or start_s > end_s or min(pm_ode_rtol, pm_ode_phase_atol) <= 0
            or source.fdot != 0):
        raise ValueError("invalid PM source, time interval or tolerances (requires fdot=0)")
    orbital, advance = _physics_switches(physics_mode)
    if advance:
        mass_time = G_M3_KG_S2*source.total_mass_kg/C_M_PER_S**3
        if 3*(mass_time*source.mean_motion)**(2/3) >= 1-source.eccentricity**2:
            raise ValueError("the 1PN map gives nonpositive initial radial mean motion")
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    # Reduce the initial orbital longitude before adding the integrated phase.
    values[6] = np.arctan2(np.sin(source.phi0), np.cos(source.phi0))
    params = jax.device_put(values, device)
    forward = max(end_s-source.t0, 0.)
    backward = min(start_s-source.t0, 0.)
    spans = [s for s in (forward, backward) if s] or [0.]
    steps = 1
    scales = jax.device_put(np.array([1e-18 + pm_ode_rtol*source.mean_motion,
                                      1e-14 + pm_ode_rtol*source.eccentricity,
                                      pm_ode_phase_atol, pm_ode_phase_atol]), device)
    coarse = [_integrate(params, s, advance=advance, steps=steps) for s in spans]
    while steps <= max_steps:
        fine = [_integrate(params, s, advance=advance, steps=2*steps) for s in spans]
        errors = [float(_refinement_error(c[0], f[0], scales)) for c, f in zip(coarse, fine)]
        if all(error <= 1. for error in errors):
            coeff, n0, omega0 = coarse[0]
            if len(spans) == 2:
                coeff = jnp.concatenate((coeff, coarse[1][0]))
            # Bound both elements throughout each segment, including between nodes.
            elements = np.asarray(coeff[:, :, :2])
            hull = _BERNSTEIN @ elements
            n_bounds, e_bounds = float(n0)+hull[..., 0], source.eccentricity+hull[..., 1]
            if n_bounds.min() <= 0 or e_bounds.min() < 0 or e_bounds.max() >= 1:
                coarse, steps = fine, 2*steps
                continue
            # If |beta_phi| < sin(pi/4), adjacent values of
            # 2*arg(1-beta_phi*exp(-iu)) cannot differ by pi on any grid.
            # Bound beta using its rationalized 1PN expression and eta >= 0.
            emax = e_bounds.max()
            root = np.sqrt(1-emax**2)
            mass_time = G_M3_KG_S2*source.total_mass_kg/C_M_PER_S**3
            nmax = n_bounds.max()
            omega_max = nmax*(1+3*(mass_time*nmax)**(2/3)/(1-emax**2))
            xmax = (mass_time*omega_max)**(2/3) if orbital else 0.
            beta_bound = emax/(1+root) + 8*xmax*emax/root
            rate_bounds = np.array([n_bounds.min(), omega_max if advance else nmax])
            # Exact leading products in the short reduction below require
            # representable turn and block counts at the chosen split widths.
            short = max(abs(forward), abs(backward))*rate_bounds[1] < 2**26
            short &= max(abs(start_s), abs(end_s), abs(source.t0)) < 2**35
            return PMPlan(
                params=params, n0=n0, omega0=omega0, coefficients=coeff,
                spans=jax.device_put(np.array(spans if spans != [0.] else [1.]), device),
                bounds=jax.device_put(np.array([start_s, end_s], dtype=float), device),
                physics_mode=physics_mode, rate_bounds=jax.device_put(rate_bounds, device),
                e_max=jax.device_put(emax, device), circular=bool(source.eccentricity == 0.),
                unwrap=bool(beta_bound >= np.sqrt(.5)), short_phase=bool(short),
            )
        coarse, steps = fine, 2*steps
    raise ValueError("PM evolution failed domain or refinement checks within max_steps")


def _residuals(plan, times):
    dt = times - plan.params[7]
    coeff = plan.coefficients
    steps = coeff.shape[0] // plan.spans.size
    if plan.spans.size == 2:
        branch = (dt < 0).astype(jnp.int32)
        position = dt*(steps/plan.spans[branch])
        offset = branch*steps
    else:
        position = dt*(steps/plan.spans[0])
        offset = 0
    if coeff.shape[0] == 1:
        return tuple(_polynomial(coeff[0, :, k], position) for k in range(4))
    idx = jnp.clip(jnp.floor(position).astype(jnp.int32), 0, steps-1)
    u = position - idx
    return tuple(_polynomial(jnp.moveaxis(coeff[:, :, k][idx+offset], -1, 0), u) for k in range(4))


def _in_bounds(plan, times):
    return (times >= plan.bounds[0]) & (times <= plan.bounds[1])


@jax.jit
def pm_evolution(plan, times):
    """Radial n, time eccentricity, unwrapped radial and azimuthal phases.

    Phases start at zero at source.t0. Queries outside the solved interval
    yield NaN. Query arrays may be unordered.
    """
    times = jnp.asarray(times, dtype=jnp.float64)
    dn, de, dl, da = _residuals(plan, times)
    dt = times - plan.params[7]
    values = (plan.n0+dn, plan.params[1]+de, plan.n0*dt+dl, plan.omega0*dt+da)
    return tuple(jnp.where(_in_bounds(plan, times), value, jnp.nan) for value in values)


@jax.jit
def pm_strain(plan, times):
    """Complex source h_plus - i*h_cross on a one-dimensional time grid."""
    orbital, advance = _physics_switches(plan.physics_mode)
    times = jnp.asarray(times, dtype=jnp.float64)
    dn, de, dl, da = _residuals(plan, times)
    n, e = plan.n0+dn, (0. if plan.circular else plan.params[1]+de)
    mass_time = G_M3_KG_S2*((plan.params[2]+plan.params[3])*SOLAR_MASS_KG)/C_M_PER_S**3
    omega = n*(1 + 3*(mass_time*n)**(2/3)/(1-e*e)) if advance else n
    def periodic(rate, offset):
        if not plan.short_phase:
            return _phase_backend(rate, times, plan.params[7], offset, 0., jnp,
                                  jax.lax.optimization_barrier, quadratic=False)
        # Split epoch differences into a dyadic block and a short remainder.
        # Both leading products are exact for |t| < 2^36 s, LISA rates < 1
        # rad/s and < 2^28 turns. Vector widths can then change without a
        # shape-dependent fused multiply/subtract on two large rounded phases.
        block = 256.
        tick = jnp.floor(times/block)
        tick0 = jnp.floor(plan.params[7]/block)
        coarse_dt = (tick-tick0)*block
        fine_dt = (times-tick*block) - (plan.params[7]-tick0*block)
        mantissa, exponent = jnp.frexp(rate)
        high = jnp.ldexp(jnp.rint(mantissa*2**24), exponent-24)
        tau_hi = 6.2831852436065673828125
        tau_lo = 6.3573019094112786e-8
        carrier_turns = jnp.rint(rate*coarse_dt/(2*jnp.pi))
        large = high*coarse_dt - carrier_turns*tau_hi
        small = (rate-high)*coarse_dt - carrier_turns*tau_lo
        offset_turns = jnp.rint(offset/(2*jnp.pi))
        residual = (offset-offset_turns*tau_hi) - offset_turns*tau_lo
        return (large+small) + (rate*fine_dt+residual)

    strain = _strain_from_orbit(plan.params, n, omega, e, 0. if plan.circular else periodic(plan.n0, dl),
                                periodic(plan.omega0, da+plan.params[6]), orbital, unwrap=plan.unwrap)
    return jnp.where(_in_bounds(plan, times), strain, jnp.nan)
