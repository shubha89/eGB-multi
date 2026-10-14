"""Current-checkout CPU batch timing for 64 distinct fixed-source waveforms.

The independent-output experiment vmaps the existing eGB single-source link
kernel; it introduces no source equations. The public source-summing API is
timed separately. Run batch sizes in separate processes for useful peak RSS.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from egb_jax_eccentric import (
    EccentricBinaryParams, LINK_LABELS, default_lisaorbits,
    state_from_lisaorbits, precompute_jax_link_geometry,
    eccentric_links_jax, prepare_xyz_from_links,
)
from egb_multi import eccentric_jax as source_kernel


def peak_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024**2) if sys.platform == "darwin" else value / 1024)


# Bound the change of normalized, unweighted XYZ vectors. This benchmark
# does not use a PSD and does not certify a noise-weighted fitting factor.
NORMALIZED_OVERLAP_BOUND_ALLOWANCE = 1e-8


def normalized_vector_change_bound(relative_error):
    if not np.isfinite(relative_error) or not 0 <= relative_error < 1:
        raise AssertionError("Invalid relative waveform discrepancy")
    return 2 * relative_error / (1 - relative_error)


def run(args):
    if (args.sources < 1 or args.batch_size < 1 or args.sources % args.batch_size
            or args.block_samples < 16 or args.repeats < 1):
        raise ValueError("Use a positive batch size dividing the source count")
    dt, halo = 5., 128
    rng = np.random.default_rng(20261014)
    sources = [EccentricBinaryParams(
        mean_motion=np.pi * rng.uniform(.001, .003), eccentricity=rng.uniform(.05, .5),
        m1_solar=.6, m2_solar=.4, distance_m=3.085677581491367e19,
        beta=rng.uniform(-.8, .8), lambda_=rng.uniform(-np.pi, np.pi),
        psi=rng.uniform(0, np.pi), inclination=rng.uniform(.2, 2.9),
        phi0=rng.uniform(0, 2 * np.pi), fdot=0.,
    ) for _ in range(args.sources)]
    times = np.arange(-halo, args.block_samples + halo) * dt
    tick = perf_counter()
    orbit = default_lisaorbits("esa-trailing")
    state = state_from_lisaorbits(orbit, times, frame="ecliptic")
    geometry = precompute_jax_link_geometry(state, orbits=orbit, frame="ecliptic")
    delays = {label: geometry["length"][i] for i, label in enumerate(LINK_LABELS)}
    geometry_s = perf_counter() - tick
    tick = perf_counter()
    tdi = prepare_xyz_from_links(state, generation=2, measurement_order=3,
        delay_order=3, link_delays=delays)
    tdi_setup_s = perf_counter() - tick
    packed_host = source_kernel.pack_eccentric_response_sources(sources)
    packed = jnp.asarray(packed_host)
    device_geometry = tuple(jnp.asarray(geometry[name]) for name in
        ("times", "n", "length", "p_sender", "p_receiver"))
    independent_kernel = jax.jit(jax.vmap(source_kernel._single_source_links,
        in_axes=(0, None, None, None, None, None)))

    def independent_waveforms():
        result = np.empty((args.sources, 3, times.size), dtype=np.complex128)
        for start in range(0, args.sources, args.batch_size):
            # np.asarray synchronizes the current main source kernel.
            links = np.asarray(independent_kernel(
                packed[start:start + args.batch_size], *device_geometry))
            for offset in range(args.batch_size):
                labelled = {label: links[offset, i] for i, label in enumerate(LINK_LABELS)}
                xyz = tdi(labelled)
                result[start + offset] = np.array([xyz[ch] for ch in "XYZ"])
        return result

    def summed_catalogue():
        links = eccentric_links_jax(sources, geometry, batch_size=args.batch_size,
            physics_mode="1pn_periastron", evolution_mode="fixed")
        xyz = tdi(links)
        return np.array([xyz[ch] for ch in "XYZ"])

    tick = perf_counter()
    independent = independent_waveforms()
    independent_first_pass_s = perf_counter() - tick
    independent_times = []
    for _ in range(args.repeats):
        tick = perf_counter()
        independent = independent_waveforms()
        independent_times.append(perf_counter() - tick)
        assert np.isfinite(independent).all()

    # All independent outputs are checked against the public scalar API.
    errors = []
    for index, source in enumerate(sources):
        scalar = tdi(eccentric_links_jax(source, geometry, batch_size=1,
            physics_mode="1pn_periastron", evolution_mode="fixed"))
        expected = np.array([scalar[ch] for ch in "XYZ"])
        error = float(np.linalg.norm(independent[index] - expected) / np.linalg.norm(expected))
        if normalized_vector_change_bound(error) > NORMALIZED_OVERLAP_BOUND_ALLOWANCE:
            raise AssertionError(f"Independent batch discrepancy exceeds its declared bound: source={index}, relative_L2={error:.17g}")
        errors.append(error)

    tick = perf_counter()
    summed = summed_catalogue()
    summed_first_pass_s = perf_counter() - tick
    summed_times = []
    for _ in range(args.repeats):
        tick = perf_counter()
        summed = summed_catalogue()
        summed_times.append(perf_counter() - tick)
        assert np.isfinite(summed).all()
    expected_sum = independent.sum(axis=0)
    sum_error = float(np.linalg.norm(summed - expected_sum) / np.linalg.norm(expected_sum))
    if normalized_vector_change_bound(sum_error) > NORMALIZED_OVERLAP_BOUND_ALLOWANCE:
        raise AssertionError("Summed catalogue exceeds its declared normalized-vector discrepancy bound")

    result = dict(
        batch_size=args.batch_size, source_count=args.sources,
        retained_samples_per_source=args.block_samples, padded_samples=times.size,
        retained_duration_days=args.block_samples * dt / 86400, dt_s=dt,
        tdi_generation=2, interpolation_orders=[3, 3], repeats=args.repeats,
        geometry_setup_s=geometry_s, tdi_setup_s=tdi_setup_s,
        independent_first_pass_s=independent_first_pass_s,
        independent_warm_repeat_s=independent_times,
        independent_warm_median_s=float(np.median(independent_times)),
        independent_output_shape=list(independent.shape),
        summed_first_pass_s=summed_first_pass_s,
        summed_warm_repeat_s=summed_times,
        summed_warm_median_s=float(np.median(summed_times)),
        summed_output_shape=list(summed.shape),
        process_peak_rss_mib=peak_rss_mib(),
        rss_scope="Peak of this isolated worker, including imports, setup, JIT, both modes and checks",
        maximum_independent_scalar_relative_error=max(errors),
        maximum_independent_normalized_vector_change_bound=normalized_vector_change_bound(max(errors)),
        summed_normalized_vector_change_bound=normalized_vector_change_bound(sum_error),
        normalized_overlap_bound_allowance=NORMALIZED_OVERLAP_BOUND_ALLOWANCE,
        strict_2e12_relative_screen_passed=(max(errors) <= 2e-12 and sum_error <= 2e-12),
        summed_vs_independent_relative_error=sum_error,
        source_parameter_sha256=hashlib.sha256(packed_host.tobytes()).hexdigest(),
        geometry_sha256=hashlib.sha256(b"".join(np.asarray(geometry[n]).tobytes()
            for n in ("times", "n", "length", "p_sender", "p_receiver"))).hexdigest(),
        source_kernel_sha256=hashlib.sha256(Path(source_kernel.__file__).read_bytes()).hexdigest(),
        backend="CPU float64; independent vmap of current eGB single-source kernel; PyTDI per source",
        python=platform.python_version(),
        versions={n: metadata.version(n) for n in ("numpy", "scipy", "jax", "jaxlib", "pytdi", "lisaorbits")},
        scope="One padded block for a fixed 64-source collection, not full-year/search timing or a maximum supported batch size",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sources", type=int, default=64)
    parser.add_argument("--block-samples", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=3)
    run(parser.parse_args())
