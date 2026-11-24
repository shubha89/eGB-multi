"""Time synchronized FP64 PM source calls and verify them on the annual campaign grid."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
import numpy as np
import scipy

from egb_multi.eccentric import EccentricBinaryParams, eccentric_polarizations
from egb_multi.pm_jax import prepare_pm, pm_evolution
from egb_multi.pm_sampling import pm_uniform_polarizations


def timed(call, repeats):
    start = perf_counter()
    result = jax.block_until_ready(call())
    first = 1000*(perf_counter()-start)
    del result
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        result = jax.block_until_ready(call())
        samples.append(1000*(perf_counter()-start))
        del result
    return dict(first_in_process_ms=first, warm_ms=samples, median_ms=float(np.median(samples)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--full-reference", action="store_true",
                        help="also compare every sample to SciPy at 3 mHz and e=0.1")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repeats required")
    count, dt = 12627040, 2.5
    t = np.arange(count, dtype=float)*dt
    jt = jax.device_put(t)
    suite = dict(samples=count, dt_s=dt, observation_s=31557600., padding_s_each_end=5000.,
                 platform=platform.platform(), device=str(jt.device), jax=jax.__version__,
                 numpy=np.__version__, scipy=scipy.__version__, cases=[])
    template = EccentricBinaryParams(.01, .1, m1_solar=1.1, m2_solar=.4, inclination=.9, phi0=.7)
    # Disjoint regular windows plus random times give a common reference across the year.
    rng = np.random.default_rng(371)
    idx = np.unique(np.r_[0, count-1, rng.integers(count, size=8192),
                          np.concatenate([i+np.arange(256) for i in
                                          np.linspace(0, count-256, 16, dtype=int)])])
    for frequency in (1., 3., 10.):
        for ecc in (0., .1, .19):
            source = replace(template, mean_motion=np.pi*(frequency*1e-3), eccentricity=ecc)
            def prepare():
                return prepare_pm(source, t[0], t[-1], device=jt.device)
            def sample(method):
                return pm_uniform_polarizations(source, t[0], dt, count, device=jt.device, method=method)
            row = dict(frequency_mHz=frequency, eccentricity=ecc)
            row["prepare"] = timed(prepare, args.repeats)
            row["fresh_orbit_arrays"] = timed(lambda: pm_evolution(prepare(), jt), args.repeats)
            for method in ("direct", "interpolate"):
                row[method] = timed(lambda: sample(method), args.repeats)
            direct = np.asarray(sample("direct"))
            interp, info = pm_uniform_polarizations(source, t[0], dt, count, method="interpolate",
                                                     device=jt.device, return_info=True)
            row["sampling"] = info
            interp = np.asarray(interp)
            diff = interp-direct
            row["full_relative_l2_vs_direct"] = float(np.linalg.norm(diff)/np.linalg.norm(direct))
            row["full_peak_error_vs_direct"] = float(np.max(np.abs(diff))/np.max(np.abs(direct)))
            hp, hc = eccentric_polarizations(source, t[idx], evolution_mode="peters_mathews",
                                             pm_ode_rtol=2e-13, pm_ode_phase_atol=1e-12)
            ref = np.array((hp,hc))
            row["reference_samples"] = len(idx)
            row["relative_l2_vs_scipy"] = float(np.linalg.norm(interp[:,idx]-ref)/np.linalg.norm(ref))
            if not (row["full_peak_error_vs_direct"] < 1e-10 and row["relative_l2_vs_scipy"] < 1e-8):
                raise RuntimeError(f"accuracy failure {row}")
            if args.full_reference and frequency == 3. and ecc == .1:
                # Bound RAM on the 16 GB Mac; each reference chunk integrates
                # from the physical source epoch using the existing SciPy path.
                err2 = ref2 = peak = delta_peak = 0.
                start = perf_counter()
                for i in range(0, count, 262144):
                    sl = slice(i, min(count, i+262144))
                    ref_block = np.asarray(eccentric_polarizations(
                        source, t[sl], evolution_mode="peters_mathews",
                        pm_ode_rtol=2e-13, pm_ode_phase_atol=1e-12))
                    delta = interp[:,sl]-ref_block
                    err2 += np.sum(delta*delta)
                    ref2 += np.sum(ref_block*ref_block)
                    peak = max(peak, np.abs(ref_block).max())
                    delta_peak = max(delta_peak, np.abs(delta).max())
                row["all_samples_scipy"] = dict(relative_l2=float(np.sqrt(err2/ref2)),
                                                 relative_peak=float(delta_peak/peak),
                                                 verification_s=perf_counter()-start,
                                                 samples=count, block_size=262144)
                if np.sqrt(err2/ref2) >= 1e-8:
                    raise RuntimeError("full annual SciPy source reference failed")
            del direct, interp, diff
            suite["cases"].append(row)
            args.output.write_text(json.dumps(suite, indent=2)+"\n")
            print(frequency, ecc, info, *(round(row[k]["median_ms"], 2) for k in
                                         ("prepare","fresh_orbit_arrays","direct","interpolate")), flush=True)


if __name__ == "__main__":
    main()
