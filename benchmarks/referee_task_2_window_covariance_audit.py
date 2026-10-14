"""Compare native simulation spectra with their finite-Welch expectations.

No estimator is adopted here. Noise cutoff and digital sampling are matched
per archived realization. The expected spectral window and mean-removal bias
are calculated from a stationary frozen secondary covariance, and integral
grid convergence is tested independently. The remaining scatter/model error,
moving-arm laser residual and annual optimized FF are distinct quantities.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/referee/.mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs/referee/.cache"))

import numpy as np
from scipy.signal import get_window
from egb_multi.external import default_lisaorbits, link_delays_from_lisaorbits
from egb_multi.spectral_window import expected_windowed_covariance
from referee_task_2_covariance_grid import chunked_model
from referee_task_2_frozen_noise_audit import instrument
from referee_task_2_psd_certification import spectra
from referee_task_2_objective_orientation import assert_local, write_json


def effective_segments(count, length):
    w = get_window("hann", length)
    correlation = np.dot(w[:length//2], w[length//2:])/np.dot(w, w)
    return count/(1 + 2*(1-1/count)*correlation**2)


def run(args):
    assert_local()
    out, ensemble = args.output.resolve(), args.ensemble.resolve()
    if not out.is_relative_to(ROOT) or not ensemble.is_relative_to(ROOT):
        raise ValueError("Use this checkout")
    out.mkdir(parents=True, exist_ok=False)
    folders = [ROOT/p for p in json.loads((ensemble/"ensemble_sources.json").read_text())["folders"]]
    sources = [Path(__file__), ROOT/"src/egb_multi/spectral_window.py", ROOT/"src/egb_multi/noise_covariance.py",
        ROOT/"benchmarks/referee_task_2_covariance_grid.py", ROOT/"benchmarks/referee_task_2_frozen_noise_audit.py",
        ROOT/"benchmarks/referee_task_2_psd_certification.py", ROOT/"benchmarks/referee_task_2_objective_orientation.py"]
    hashes = {}
    for path in sources:
        rel = path.relative_to(ROOT)
        target = out/"snapshot"/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    input_hashes = {}
    for folder in folders:
        for filename in ("noise_and_delays.npz", "psd_metadata.json"):
            path = folder/filename
            input_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/"manifest.json", dict(created_utc=datetime.now(timezone.utc).isoformat(), sha256=hashes,
        input_sha256=input_hashes, configuration={key:str(value) if isinstance(value, Path) else value for key,value in vars(args).items()},
        integral_relative_error_allowance=args.grid_budget,
        scope="Welch expectation and model diagnostics; no production FF certification"))
    orbit = default_lisaorbits("esa-trailing")
    fs, grid_samples = .2, args.grid_samples
    if grid_samples < 4*max(args.segment_lengths) or grid_samples % 4 or any(grid_samples % size for size in args.segment_lengths):
        raise ValueError("Use an integral grid divisible by four and at least four times the longest segment")
    f_grid = np.fft.rfftfreq(grid_samples, 1/fs)
    definitions, cached = [], {}
    for folder in folders:
        metadata = json.loads((folder/"psd_metadata.json").read_text())
        duration = metadata["dt_s"]*metadata["samples_before_endpoint_crop"]
        cutoff = metadata["instrument_kwargs"].get("noises_f_min_hz") or 1/duration
        epoch = metadata["instrument_kwargs"].get("t0", 0.) + duration/2
        key = (cutoff, epoch)
        definitions.append((key, folder))
        if key in cached:
            continue
        print("Native covariance grid", key, len(f_grid), flush=True)
        delays = link_delays_from_lisaorbits(orbit, np.array([epoch]))
        covariance = np.zeros((len(f_grid), 2, 2), complex)
        covariance[1:] = chunked_model(f_grid[1:], instrument(cutoff, 5.), delays)
        # Frozen TDI has zero DC response to each raw secondary measurement.
        # Nyquist is a real endpoint of the sampled process, not doubled.
        covariance[-1] = covariance[-1].real/2
        cached[key] = covariance
    rows, failures = [], []
    for length in args.segment_lengths:
        observed, expected, pointwise, counts, grid_errors = [], [], [], [], []
        for key, folder in definitions:
            f, values, count = spectra(folder, length)
            _, windowed = expected_windowed_covariance(cached[key], fs, length)
            _, coarse = expected_windowed_covariance(cached[key][::2], fs, length)
            _, coarsest = expected_windowed_covariance(cached[key][::4], fs, length)
            matrix = np.zeros_like(windowed)
            matrix[:,0,0], matrix[:,1,1] = values[0].real, values[1].real
            matrix[:,0,1], matrix[:,1,0] = values[2], values[2].conj()
            observed.append(matrix)
            expected.append(windowed)
            pointwise.append(cached[key][::grid_samples//length][:length//2+1])
            counts.append(effective_segments(count, length))
            mask = (f>=2e-5)&(f<=.04)
            scale = np.sqrt(windowed[:,0,0].real*windowed[:,1,1].real)
            error = np.max(abs(coarse-windowed), axis=(1,2))/scale
            old_error = np.max(abs(coarsest-coarse), axis=(1,2))/scale
            grid_errors.append(dict(folder=str(folder.relative_to(ROOT)), maximum_relative_error=float(error[mask].max()),
                previous_refinement_error=float(old_error[mask].max())))
        actual, model, direct = np.mean(observed, axis=0), np.mean(expected, axis=0), np.mean(pointwise, axis=0)
        effective = len(counts)**2 / np.sum(1/np.array(counts))
        bins = []
        for low, high in [(2e-5,1e-4),(1e-4,3e-4),(3e-4,1e-3),(1e-3,3e-3),(3e-3,.01),(.01,.02),(.02,.028),(.029,.031),(.032,.04)]:
            mask = (f>=low)&(f<high)
            normalize = np.sqrt(model[mask,0,0].real*model[mask,1,1].real)
            delta = (actual[mask,0,1]-model[mask,0,1])/normalize
            bins.append(dict(band_hz=[low,high], bins=int(mask.sum()),
                observed_over_windowed_A=float(np.mean(actual[mask,0,0].real/model[mask,0,0].real)),
                observed_over_windowed_E=float(np.mean(actual[mask,1,1].real/model[mask,1,1].real)),
                windowed_over_pointwise_A_min=float(np.min(model[mask,0,0].real/direct[mask,0,0].real)),
                windowed_over_pointwise_A_max=float(np.max(model[mask,0,0].real/direct[mask,0,0].real)),
                normalized_CSD_difference_rms=float(np.sqrt(np.mean(abs(delta)**2)))))
        maximum = max(row["maximum_relative_error"] for row in grid_errors)
        if maximum > args.grid_budget:
            failures.append(f"segment {length}: spectral-integral refinement exceeds allowance")
        record = dict(nperseg=length, independent_realizations=len(folders), effective_segments=effective,
            expected_uncorrelated_CSD_scatter_rms=1/np.sqrt(effective), integral_controls=grid_errors, bands=bins)
        rows.append(record)
        write_json(out/"window_comparison.json", rows)
        np.savez_compressed(out/f"windowed_covariance_segment{length}.npz", frequency_hz=f, observed=actual,
            expected_windowed=model, pointwise=direct)
        print("Segment", length, "integral error", maximum, "null band", bins[-2], flush=True)
    write_json(out/"validation_summary.json", dict(status="passed_spectral_integral_controls" if not failures else "discrepancy_requires_investigation",
        failures=failures, production_certified=False, independent_realizations=len(folders),
        limitations=["Stationary frozen-arm secondary covariance; moving/locking laser residual omitted",
            "Sampling and cutoff matched to each realization, not the proposed continuum production model",
            "Broad simulation agreement does not bound narrow-line optimized FF error",
            "Spectral-bin correlations and statistical errors must not be inferred from independent-bin counts"]))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ensemble", type=Path, default=ROOT/"outputs/referee/task_2_8_20261008_noise_ensemble")
    p.add_argument("--grid-samples", type=int, default=1048576)
    p.add_argument("--segment-lengths", type=int, nargs="+", default=[16384,32768,65536])
    p.add_argument("--grid-budget", type=float, default=1e-4)
    run(p.parse_args())
