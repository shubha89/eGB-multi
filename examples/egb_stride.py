"""eGB polarizations through the bundled STRIDE response with retarded geometry.

Run from an installation of egb-multi[orbits]; uses 4096 output samples by default.
No waveform array is written unless --output is supplied.
"""
import time
from dataclasses import asdict
import jax
jax.config.update("jax_enable_x64", True)

from _response_common import arguments, inputs, finish
from egb_multi.external import default_lisaorbits
from egb_multi.stride import (
    StreamedDetectorResponse, geometry_from_lisaorbits, sampled_polarizations,
)


def main():
    args, margin = arguments(__doc__)
    source, source_times, times, _ = inputs(args, margin)
    if args.backend == "numba" and args.device != "cpu":
        raise ValueError("Numba is a CPU backend")
    tick = time.perf_counter()
    orbit = default_lisaorbits("esa-trailing")
    # Detector-only provider: independent of mass, eccentricity, sky and psi.
    provider = lambda query: geometry_from_lisaorbits(orbit, query)
    options = dict(measurement_order=25, padding_order=121,
                   block_samples=32768, cache_bytes=256*1024**2)
    if args.backend == "jax":
        options.update(device=args.device, execution_schedule="fused_unrolled" if args.device == "gpu" else "staged")
    detector = StreamedDetectorResponse(times, provider, backend=args.backend, **options)
    setup_s = time.perf_counter()-tick
    tick = time.perf_counter()
    sampled = sampled_polarizations(
        source, source_times, physics_mode="1pn", evolution_mode=args.evolution,
        pm_sampling=args.pm_sampling, source_backend=args.source_backend, device=args.device,
    )
    jax.block_until_ready(sampled.values)
    source_s = time.perf_counter()-tick
    if args.backend == "numba":
        sampled = sampled.to_host()
    tick = time.perf_counter()
    xyz = detector.sampled_response(
        sampled, source.propagation_direction(), *source.polarization_basis(),
        order=39, polarization_angle=source.psi,
    )
    response_s = time.perf_counter()-tick
    finish(args, times, xyz, dict(
        engine="STRIDE", response_backend=args.backend, device=args.device,
        source_backend=args.source_backend, source_parameters=asdict(source),
        physics_mode="1pn", evolution_mode=args.evolution, pm_sampling=args.pm_sampling,
        geometry="esa-trailing; ecliptic; retarded emitter",
        source_order=39, tdi_order=25, setup_s=setup_s,
        source_s=source_s, first_response_s=response_s,
        detector_cached_bytes=detector.cached_bytes,
        timing_scope="first call includes compilation and detector-block preparation",
    ))


if __name__ == "__main__":
    main()
