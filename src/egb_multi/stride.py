"""eGB source and orbit adapters for the bundled STRIDE response.

Source times are barycentric seconds; strain is dimensionless. Positions
are in light-seconds, directed delays in seconds, and links in receiver/emitter
order. A sampled source retains its original plus/cross basis: pass
source.polarization_basis() and polarization_angle=source.psi to STRIDE.
"""
import numpy as np
import jax
import jax.numpy as jnp

from lisa_stride import (
    DetectorResponse, SampledPolarizations, StreamedDetectorResponse, TDIPlan,
    xyz_to_aet,
)
from .eccentric import (
    eccentric_evolution_mode_label, eccentric_physics_options,
    eccentric_polarizations,
)

__all__ = [
    "DetectorResponse", "SampledPolarizations", "StreamedDetectorResponse",
    "TDIPlan", "xyz_to_aet", "sampled_polarizations", "geometry_from_lisaorbits",
]


def sampled_polarizations(source, times, *, physics_mode="1pn",
                          evolution_mode="fixed", source_backend="numpy",
                          device=None, order=39, **evolution_options):
    """Generate one source on a uniform, caller-padded strain grid.

    source_backend='jax' retains the fixed-element source on the chosen JAX
    device. Peters--Mathews evolution uses the existing NumPy/SciPy source,
    with one explicit transfer to the response device. This checked setup
    helper is outside an outer JIT; STRIDE's array interfaces compose under JIT.

    Nonzero prescribed fdot is rejected because eGB's NumPy and JAX source
    APIs currently assign it different scope. Use their explicit APIs if
    that prescription is intended. Source psi is applied by the response,
    rather than being included a second time in these polarizations.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX float64 before constructing sampled sources")
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("times must be finite, increasing and one-dimensional")
    dt = float(times[1] - times[0])
    expected = times[0] + np.arange(times.size) * dt
    tolerance = 4 * np.finfo(np.float64).eps * max(1., np.max(np.abs(times)))
    if np.max(np.abs(times - expected)) > tolerance:
        raise ValueError("sampled polarizations require a uniform time grid")
    if (isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer))
            or order < 1 or order % 2 != 1 or times.size < order + 1):
        raise ValueError("an odd positive interpolation order and its full stencil are required")
    if source.fdot != 0:
        raise ValueError("the shared source adapter requires fdot=0; use explicit eGB APIs")
    mode = eccentric_evolution_mode_label(evolution_mode)
    if source_backend == "numpy":
        hp, hc = eccentric_polarizations(
            source, times, physics_mode=physics_mode,
            evolution_mode=mode, **evolution_options,
        )
        return SampledPolarizations.from_arrays(
            hp, hc, origin_s=float(times[0]), dt_s=dt,
            order=order, device=device,
        )
    if source_backend != "jax":
        raise ValueError("source_backend must be numpy or jax")
    if mode != "fixed":
        raise ValueError("JAX source generation supports fixed elements; select numpy for PM evolution")
    if evolution_options:
        raise ValueError("PM integration options apply to the numpy source backend")
    from .eccentric_jax import (
        pack_eccentric_sources, eccentric_complex_strain_batch_jax_array_options,
    )
    options = eccentric_physics_options(physics_mode)
    target = _device(device)
    packed = jax.device_put(pack_eccentric_sources(source), target)
    grid = jax.device_put(times, target)
    strain = eccentric_complex_strain_batch_jax_array_options(
        packed, grid,
        include_1pn_orbital_corrections=options.include_1pn_orbital_corrections,
        include_periastron_advance=options.include_periastron_advance,
    )[0]
    # eGB's complex source is h_plus - i h_cross.
    values = jnp.stack((strain.real, -strain.imag))
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("nonfinite source polarizations")
    return SampledPolarizations(
        jax.device_put(np.float64(times[0]), target),
        jax.device_put(np.float64(dt), target), values, order,
    )


def _device(device):
    if device is None or not isinstance(device, str):
        return device
    platform, _, index = device.partition(":")
    devices = jax.devices(platform)
    number = int(index or 0)
    if number < 0 or number >= len(devices):
        raise ValueError("requested JAX device is unavailable")
    return devices[number]


def geometry_from_lisaorbits(orbits, times, *, frame="ecliptic"):
    """Return the existing eGB retarded-emitter geometry in STRIDE's format.

    The source basis and the orbit positions must share the requested frame.
    Geometry can be cached while the provider and detector grid remain fixed.
    Sky position is deliberately absent from this preparation.
    """
    from .external import state_from_lisaorbits
    from .eccentric_jax import precompute_jax_link_geometry
    state = state_from_lisaorbits(orbits, times, frame=frame)
    return precompute_jax_link_geometry(state, orbits=orbits, frame=frame)
