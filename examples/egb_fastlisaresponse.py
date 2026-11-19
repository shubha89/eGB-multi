"""Use native fastlisaresponse with eGB polarizations on CPU or CUDA.

The optional --compare-stride check uses the same sampled input and the same
instantaneous-emitter orbit prescription. Defaults are a short CPU example.
"""
import time
from dataclasses import asdict
from importlib.metadata import version
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from _response_common import arguments, inputs, finish
from egb_multi.constants import C_M_PER_S
from egb_multi.fastlisa import complex_polarizations
from egb_multi.stride import sampled_polarizations


def matched_geometry(provider, times):
    """CPU orbit queries in the frame/time origin of the native orbit file."""
    from lisa_stride import LINK_LABELS
    positions = {sc: provider.get_pos(times, sc)/C_M_PER_S for sc in (1, 2, 3)}
    senders, receivers, directions, lengths = [], [], [], []
    for link in LINK_LABELS:
        r, e = map(int, link)
        sender, receiver = positions[e], positions[r]
        delta = receiver-sender
        senders.append(sender); receivers.append(receiver)
        directions.append(delta/np.linalg.norm(delta, axis=-1)[:, None])
        lengths.append(provider.get_light_travel_times(times, int(link)))
    return dict(times=times, n=np.array(directions), length=np.array(lengths),
                p_sender=np.array(senders), p_receiver=np.array(receivers))


def main():
    args, margin = arguments(__doc__, native=True)
    source, source_times, times, keep = inputs(args, margin)
    try:
        from fastlisaresponse import pyResponseTDI
        from lisatools.detector import EqualArmlengthOrbits, Orbits
    except ImportError as exc:
        raise SystemExit('Install the optional interface: pip install "egb-multi[fastlisa]"') from exc
    cuda = args.backend != "cpu"
    if cuda:
        import cupy as xp
        if xp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CUDA was requested but no device is available")
        source_device = "gpu"
    else:
        xp = np
        source_device = "cpu"
    tick = time.perf_counter()
    orbit = (Orbits(str(args.orbit_file), force_backend=args.backend) if args.orbit_file
             else EqualArmlengthOrbits(force_backend=args.backend))
    # Configure only the requested padded interval, rather than allocating
    # a fine interpolation grid over the entire multi-year ephemeris.
    orbit_grid = np.arange(int(np.ceil((source_times[-1]+300.)/args.orbit_dt))+1)*args.orbit_dt
    if orbit_grid[-1] > orbit.t_base[-1]:
        raise ValueError("orbit file does not cover the padded interval")
    orbit.configure(t_arr=orbit_grid)
    # Preparation can be reused when the orbit, cadence and record length stay fixed.
    native = pyResponseTDI(
        1./args.dt, source_times.size, order=25, tdi="2nd generation",
        orbits=orbit, tdi_chan="XYZ", force_backend=args.backend,
    )
    # Receiver/emitter labels are explicit for compatibility with the checked
    # fastlisaresponse 1.1.17 path; no upstream source or orbit data are modified.
    native.link_space_craft_0_in = native.response_orbits.link_space_craft_r
    setup_s = time.perf_counter()-tick
    if native.num_pts != source_times.size:
        raise ValueError("orbit support is shorter than the requested padded record")
    tick = time.perf_counter()
    sampled = sampled_polarizations(
        source, source_times, physics_mode="1pn", evolution_mode=args.evolution,
        source_backend=args.source_backend, device=source_device,
    )
    jax.block_until_ready(sampled.values)
    source_s = time.perf_counter()-tick
    tick = time.perf_counter()
    # Explicitly share JAX CUDA strain; CPU input uses a NumPy host array.
    values = xp.from_dlpack(sampled.values) if cuda else np.asarray(sampled.values)
    h_native = complex_polarizations(source, values, xp=xp)  # h+ + i hx
    native.get_projections(h_native, source.lambda_, source.beta, t0=5000.)
    if cuda:
        xp.cuda.get_current_stream().synchronize()
    links_s = time.perf_counter()-tick
    tick = time.perf_counter()
    # The native global XYZ sign is mapped to the eGB/pyTDI convention.
    xyz_device = -xp.stack(native.get_tdi_delays())[:, keep]
    if cuda:
        xp.cuda.get_current_stream().synchronize()
    tdi_s = time.perf_counter()-tick
    xyz = xp.asnumpy(xyz_device) if cuda else np.asarray(xyz_device)
    summary = dict(
        engine="fastlisaresponse", fastlisaresponse_version=version("fastlisaresponse"),
        response_backend=args.backend, source_backend=args.source_backend,
        source_parameters=asdict(source), physics_mode="1pn", evolution_mode=args.evolution,
        geometry="shared native orbit; ecliptic; instantaneous emitter",
        native_order=25, orbit_dt_s=args.orbit_dt,
        setup_s=setup_s, source_s=source_s, projection_s=links_s, tdi_s=tdi_s,
        timing_scope="single first call; compilation, transfers and host readback are not a controlled warm benchmark",
    )
    if args.compare_stride:
        from lisa_stride import StreamedDetectorResponse
        # Host ephemeris queries use a CPU provider with the same file and grid.
        comparison_orbit = orbit
        if cuda:
            comparison_orbit = Orbits(orbit.filename, force_backend="cpu")
            comparison_orbit.configure(t_arr=orbit_grid)
        detector = StreamedDetectorResponse(
            times, lambda query: matched_geometry(comparison_orbit, query),
            device=source_device, measurement_order=25, padding_order=121,
            execution_schedule="fused_unrolled" if cuda else "staged",
        )
        reference = detector.sampled_response(
            sampled, source.propagation_direction(), *source.polarization_basis(),
            order=39, polarization_angle=source.psi,
        )
        norm = np.linalg.norm(reference)
        if norm == 0:
            raise ValueError("zero response norm")
        summary["native_vs_stride_relative_l2"] = float(np.linalg.norm(xyz-reference)/norm)
        summary["comparison_scope"] = "same sampled polarizations and instantaneous geometry; no fitted alignment or equal-accuracy claim"
    finish(args, times, xyz, summary)


if __name__ == "__main__":
    main()
