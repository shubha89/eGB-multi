"""Locate the first loss of laser/secondary superposition in native streams.

All three runs use native amplitudes, matched seed and fixed cutoff, with
ideal clocks/ranges, no GW and only laser/test-mass/OMS contributions.
Under these assumptions the fluctuations are linear in the selected inputs.
For every exported intermediate stream test (combined - laser) - secondary.
This identifies where a discrepancy first appears; it does not assign a
physical origin without a precision or independent propagation control.
"""

import argparse
from dataclasses import fields
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/referee/.mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs/referee/.cache"))

import numpy as np
from lisainstrument import Instrument, SchedulerConfigSerial
from lisainstrument.instru import instru_formulas
from egb_multi.noise import pytdi_data_from_lisainstrument_result, xyz_from_pytdi_data, aet_from_xyz
from referee_laser_on_off import digest, spectra
from referee_task_2_objective_orientation import assert_local, write_json


def run(args):
    assert_local()
    out, orbit = args.output.resolve(), args.orbit.resolve()
    if not out.is_relative_to(ROOT) or not orbit.is_relative_to(ROOT):
        raise ValueError("Use this checkout")
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in [Path(__file__), ROOT / "src/egb_multi/noise.py", ROOT / "benchmarks/referee_laser_on_off.py", ROOT / "benchmarks/referee_task_2_objective_orientation.py", orbit]:
        rel = path.relative_to(ROOT)
        target = out / "snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    native = Path(instru_formulas.__file__)
    target = out / "snapshot/upstream/instru_formulas.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(native, target)
    kwargs = dict(size=args.samples, dt=5., t0=args.epoch, seed=args.seed,
        orbits=str(orbit), orbit_dataset="tcb/ltt", physics_upsampling=1,
        aafilter=None, telemetry_downsampling=args.samples,
        interpolation=("lagrange", 45), noises_f_min_hz=1/(args.cutoff_days*86400))
    write_json(out / "manifest.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        sha256=hashes, upstream_formula_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        configuration=kwargs, scope="Native stage superposition diagnostic; not FF certification"))
    runs, input_hashes = {}, {}
    for name, enabled in [("on", ["laser", "test-mass", "oms"]), ("laser", ["laser"]), ("off", ["test-mass", "oms"])]:
        start = time.perf_counter()
        instrument = Instrument(**kwargs)
        instrument.disable_all_noises(excluding=enabled)
        raw = instrument.export_numpy_full(cfgscheduler=SchedulerConfigSerial(chunk_size=args.samples))
        runs[name] = raw
        selected = {}
        for group in ("mosa_data", "sat_data"):
            data = getattr(raw, group)
            for field in fields(data):
                if "noise" not in field.name:
                    continue
                for label, value in getattr(data, field.name).items():
                    selected[f"{group}.{field.name}.{label}"] = digest(value)
        input_hashes[name] = selected
        print(name, "generated", time.perf_counter()-start, flush=True)
    matched_laser = [key for key in input_hashes["on"] if "laser" in key]
    matched_secondary = [key for key in input_hashes["on"] if "laser" not in key]
    if any(input_hashes["on"][key] != input_hashes["laser"][key] for key in matched_laser):
        raise ValueError("Laser inputs are not matched")
    if any(input_hashes["on"][key] != input_hashes["off"][key] for key in matched_secondary):
        raise ValueError("Secondary inputs are not matched")
    for name in ("laser", "off"):
        np.testing.assert_array_equal(runs["on"].t, runs[name].t)
        for label in runs["on"].mosa_data.mprs:
            np.testing.assert_array_equal(runs["on"].mosa_data.mprs[label], runs[name].mosa_data.mprs[label])
    rows = []
    for group in ("mosa_data", "sat_data"):
        data = getattr(runs["on"], group)
        for field in fields(data):
            if "fluctuations" not in field.name:
                continue
            for label, combined in getattr(data, field.name).items():
                laser = np.asarray(getattr(getattr(runs["laser"], group), field.name)[label])
                secondary = np.asarray(getattr(getattr(runs["off"], group), field.name)[label])
                combined = np.asarray(combined)
                error = (combined-laser)-secondary
                selected = slice(256, -256) if error.size > 1024 else slice(None)
                rms = lambda x: float(np.sqrt(np.mean(np.asarray(x)[selected]**2)))
                off_rms, error_rms = rms(secondary), rms(error)
                rows.append(dict(stream=f"{group}.{field.name}.{label}",
                    samples=int(error.size), error_rms=error_rms,
                    maximum_abs_error=float(np.max(abs(error[selected]))),
                    secondary_rms=off_rms,
                    error_over_secondary_rms=error_rms/off_rms if off_rms > 0 else None,
                    combined_maximum_abs=float(np.max(abs(combined[selected])))))
    write_json(out / "stage_errors.json", rows)
    # Native N1-12 locking: MOSA13 copies primary MOSA12 and adds the
    # independent REF OMS fluctuation. All other terms in this local sum
    # vanish in the chosen ideal reference. Verify, rather than assume, it.
    primary = np.asarray(runs["laser"].mosa_data.local_carrier_fluctuations["12"])
    small = np.asarray(runs["off"].mosa_data.local_carrier_fluctuations["13"])
    actual = np.asarray(runs["on"].mosa_data.local_carrier_fluctuations["13"])
    np.testing.assert_array_equal(primary, runs["laser"].mosa_data.local_carrier_fluctuations["13"])
    np.testing.assert_array_equal(actual, primary+small)
    # Error-free TwoSum preserves the same physical sum in two float64
    # limbs. Decimal verifies the identity independently for sampled inputs.
    hi = primary+small
    z = hi-primary
    lo = (primary-(hi-z))+(small-z)
    recovered = (hi-primary)+lo
    indices = np.linspace(256, len(primary)-257, 128, dtype=int)
    decimal_difference = []
    with localcontext() as context:
        context.prec = 80
        for index in indices:
            exact = Decimal.from_float(float(primary[index])) + Decimal.from_float(float(small[index]))
            represented = Decimal.from_float(float(hi[index])) + Decimal.from_float(float(lo[index]))
            decimal_difference.append(float(abs(exact-represented)))
    local_control = dict(
        operation="N1-12 adjacent locking p13 = p12 + central_frequency * REF_OMS13",
        native_combined_equals_float64_sum=True,
        maximum_native_secondary_recovery_error_Hz=float(np.max(abs((actual-primary)-small))),
        maximum_compensated_secondary_recovery_error_Hz=float(np.max(abs(recovered-small))),
        maximum_decimal_sum_representation_error_Hz=max(decimal_difference),
        decimal_precision=80, decimal_checked_samples=len(indices),
        scope="First adjacent-locking sum only; not a corrected full instrument/TDI run")
    write_json(out / "first_operation_precision.json", local_control)
    print("First-operation precision", local_control, flush=True)
    # Preserve final streams and laser-only TDI for later physical controls.
    values, carrier = {}, float(instrument.central_freq)
    for name, raw in runs.items():
        data = pytdi_data_from_lisainstrument_result(raw)
        ae = aet_from_xyz(xyz_from_pytdi_data(data, generation=2, order=45, delay_order=5))
        values[name] = {ch: ae[ch][198:-198]/carrier for ch in ("A", "E")}
    np.savez_compressed(out / "noise.npz", t=runs["on"].t[198:-198],
        **{f"{name}_{ch}": x for name, channels in values.items() for ch, x in channels.items()})
    nperseg = min(16384, (args.samples-396)//4)
    f, secondary, _, _ = spectra(values["off"], 5., nperseg)
    metrics = {}
    for name, channels in [("laser_only", values["laser"]), ("superposition", {ch: (values["on"][ch]-values["laser"][ch])-values["off"][ch] for ch in ("A", "E")})]:
        _, power, _, _ = spectra(channels, 5., nperseg)
        mask = (f >= 2e-5) & (f <= .04)
        metrics[name] = {ch: float(np.max(power[ch][mask]/secondary[ch][mask])) for ch in ("A", "E")}
    meaningful = [row for row in rows if row["error_rms"] > 0]
    summary = dict(status="stage_diagnostic_completed", production_certified=False,
        matched_laser_inputs=len(matched_laser), matched_secondary_inputs=len(matched_secondary),
        cutoff_days=args.cutoff_days, samples=args.samples, final_PSD_ratios=metrics,
        nonzero_streams=len(meaningful), largest_errors=sorted(meaningful, key=lambda row: row["error_rms"], reverse=True)[:15])
    write_json(out / "validation_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    del runs
    gc.collect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--orbit", type=Path, default=ROOT/"outputs/referee/task_2_8_20261008_annual_laser_residual/native_annual_orbits.h5")
    p.add_argument("--samples", type=int, default=65536)
    p.add_argument("--cutoff-days", type=float, default=365.25)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--epoch", type=float, default=0.)
    run(p.parse_args())
