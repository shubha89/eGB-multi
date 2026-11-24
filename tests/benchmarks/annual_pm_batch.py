"""Time fixed- or independent-sky PM-on eGB through second-generation XYZ."""
import argparse
from dataclasses import replace
import json
import math
from statistics import median
import time

import jax
import jax.numpy as jnp
import numpy as np

from egb_multi.constants import C_M_PER_S
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import precompute_jax_link_geometry
from egb_multi.geometry import lisa_orbit
from egb_multi.stride import (FixedSkyResponse, StreamedDetectorResponse, annual_pm_grid,
                              sampled_pm_batch, sampled_pm_uniform)
from lisa_stride import LINK_LABELS
from lisa_stride.batching import sampled_response_batch_array
from xtdi.common import conservative_halo


SOURCE_EPOCH = 3_939_700.
OBSERVATION_EPOCH = SOURCE_EPOCH + 5000.


def numerical_provider(path):
    """Use six directed travel times and simultaneous BME spacecraft positions."""
    import h5py
    from lisatools.detector import Orbits

    with h5py.File(path) as f:
        epoch = float(f.attrs["time_origin_s"])
        last = (int(f.attrs["size"])-1) * float(f.attrs["dt"])
    orbit = Orbits(str(path), force_backend="cpu")
    orbit.configure(dt=37.5)

    def geometry(times):
        local = times - epoch
        if local.min() < 0 or local.max() > last:
            raise ValueError("Orbit table does not cover the requested halo")
        positions = np.stack([orbit.get_pos(local, i)/C_M_PER_S for i in (1, 2, 3)])
        receiver = np.stack([positions[int(link[0])-1] for link in LINK_LABELS])
        sender = np.stack([positions[int(link[1])-1] for link in LINK_LABELS])
        delta = receiver - sender
        return dict(times=times, n=delta/np.linalg.norm(delta, axis=-1)[..., None],
                    length=np.stack([orbit.get_light_travel_times(local, int(link)) for link in LINK_LABELS]),
                    p_receiver=receiver, p_sender=sender)
    return geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sky", choices=["fixed", "distinct"], default="fixed")
    parser.add_argument("--source", choices=["standard", "vectorized"], default="vectorized")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frequency-mhz", type=float, default=1.)
    parser.add_argument("--eccentricity", type=float, default=0.1)
    parser.add_argument("--cadence", default="likelihood", help="seconds, waveform, or likelihood")
    parser.add_argument("--annual", action="store_true", help="one Julian year (default is a short pilot)")
    parser.add_argument("--orbit", help="optional BME LISATools numerical orbit table with time_origin_s")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.batch_size < 1 or args.repeats < 1:
        parser.error("batch size and repeats must be positive")
    jax.config.update("jax_enable_x64", True)
    device = jax.devices(args.device)[0]
    base = EccentricBinaryParams(np.pi*args.frequency_mhz/1000., args.eccentricity,
                                 m1_solar=1.1, m2_solar=.4, inclination=.9, phi0=.7, beta=.3, lambda_=1.2)
    rule = args.cadence if args.cadence in ("waveform", "likelihood") else float(args.cadence)
    grid = annual_pm_grid(base, cadence_s=rule)
    dt = grid.cadence_s
    count = grid.samples if args.annual else min(grid.samples, 2048)
    halo = conservative_halo(dt, 12., 25, 7)
    pad = halo + 24 + math.ceil(1000./dt)
    origin = OBSERVATION_EPOCH - pad*dt
    source_origin = origin - SOURCE_EPOCH
    size = count + 2*pad

    def analytic_geometry(times):
        return precompute_jax_link_geometry(lisa_orbit(times))

    provider = numerical_provider(args.orbit) if args.orbit else analytic_geometry
    times = OBSERVATION_EPOCH + np.arange(count)*dt
    # FixedSkyResponse retains its own blocks. Independent skies reuse the
    # stream's detector-only cache between physical proposals.
    stream = StreamedDetectorResponse(times, provider, device=device, generation=2,
                                     block_samples=131072, measurement_order=25, delay_order=7,
                                     cache_bytes=(20 * 2**30 if args.sky == "distinct" else 0),
                                     execution_schedule="fused_unrolled")
    members = [replace(base, m1_solar=base.m1_solar*(1+.0003*i/64),
                       m2_solar=base.m2_solar*(1-.0004*i/64), psi=.002*i,
                       inclination=base.inclination+.001*i, phi0=base.phi0+.009*i,
                       lambda_=base.lambda_ + (.1*i if args.sky == "distinct" else 0.))
               for i in range(args.batch_size)]
    start = time.perf_counter()
    fixed = None
    if args.sky == "fixed":
        fixed = FixedSkyResponse.prepare(stream, base.propagation_direction(), *base.polarization_basis(),
                                         origin_s=origin, dt_s=dt, samples=size)
    setup = time.perf_counter()-start

    def evaluate(proposals, *, reference=False):
        if args.source == "standard" or reference:
            values = jnp.stack([sampled_pm_uniform(s, source_origin, dt, size, device=device).values
                                for s in proposals])
        else:
            values = sampled_pm_batch(proposals, source_origin, dt, size, device=device).values
        angles = jax.device_put(np.array([s.psi for s in proposals]), device)
        if fixed is not None and not reference:
            return fixed.apply_array(values, angles)
        sky = [jax.device_put(np.stack(items), device) for items in zip(
            *((s.propagation_direction(), *s.polarization_basis()) for s in proposals), strict=True)]
        return sampled_response_batch_array(stream, values, origin, dt, *sky, angles,
                                           microbatch=1 if reference else len(proposals))

    # Validate each lane against the ordinary source and independent-sky response.
    references = []
    for s in members:
        xyz, valid = evaluate([s], reference=True)
        if not bool(valid[0]):
            raise ValueError("Invalid scalar reference")
        references.append(xyz[0])
    references = jnp.stack(references)
    xyz, valid = evaluate(members)
    err = float(jnp.max(jnp.linalg.norm(xyz-references, axis=-1) / jnp.linalg.norm(references, axis=-1)))
    if not bool(jnp.all(valid)) or not np.isfinite(err) or err >= 1e-10:
        raise ValueError(f"Scalar/batch XYZ comparison failed: {err}")
    del xyz, references
    measurements = []
    for repeat in range(args.repeats):
        trial = [replace(s, phi0=s.phi0+.001*(repeat+1)) for s in members]
        start = time.perf_counter()
        xyz, valid = jax.block_until_ready(evaluate(trial))
        if not bool(jnp.all(valid)):
            raise ValueError("Invalid timed response")
        measurements.append(time.perf_counter()-start)
        del xyz
    print(json.dumps(dict(sky=args.sky, source=args.source, batch=len(members), cadence_s=dt,
                          samples=count, device=device.device_kind, numerical_orbit=bool(args.orbit),
                          fixed_sky_preparation_s=setup if fixed is not None else None,
                          warm_batch_ms=1000*median(measurements),
                          ms_per_waveform=1000*median(measurements)/len(members),
                          max_channel_relative_l2=err, wall_s=measurements), indent=2))


if __name__ == "__main__":
    main()
