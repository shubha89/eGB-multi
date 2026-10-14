"""Short native regeneration/retention control; not annual FF validation."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]
import numpy as np
from egb_multi.comparison import TDIConvention
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import default_lisaorbits
from egb_multi.native_validation import ChunkedNativeResponse
from referee_task_2_objective_orientation import assert_local
import referee_task_2_production as production
import referee_task_2_waveform_cache as cache


def run():
    assert_local()
    out = ROOT / "outputs/referee/task_2_9_20261013_waveform_cache_control"
    out.mkdir(parents=True, exist_ok=False)
    folder = out / "points/native_control"
    inputs = folder / "inputs"
    inputs.mkdir(parents=True)
    frozen = production.read(production.FROZEN)
    convention = TDIConvention(**frozen["tdi_convention"])
    duration, dt = 21600., frozen["waveform_dt_s"]
    times = np.arange(round(duration / dt)) * dt
    source = EccentricBinaryParams(mean_motion=np.pi*.001, eccentricity=.1,
        m1_solar=.5, m2_solar=.5, inclination=.8, beta=.2, lambda_=-1., psi=.4, phi0=.2)

    def generate():
        started = time.monotonic()
        native = ChunkedNativeResponse(default_lisaorbits("esa-trailing"), times, convention,
            block_samples=4096, cached_blocks=2)
        signal = native(source, "peters_mathews", pm_ode_rtol=frozen["pm_ode_rtol"],
            pm_ode_phase_atol=frozen["pm_ode_phase_atol"])
        assert all(np.isfinite(signal.channels[ch]).all() for ch in ("A", "E"))
        path = inputs / "production_signal.npz"
        np.savez_compressed(path, A=signal.channels["A"], E=signal.channels["E"])
        production.save(inputs / "production_source.json", dict(source=asdict(source), evolution="peters_mathews",
            convention=asdict(convention), duration_s=duration, dt_s=dt, signal_sha256=cache.digest(path),
            signal_array_sha256={ch:hashlib.sha256(signal.channels[ch].tobytes()).hexdigest() for ch in ("A", "E")},
            pm_ode_rtol=frozen["pm_ode_rtol"], pm_ode_phase_atol=frozen["pm_ode_phase_atol"],
            injection_wall_s=time.monotonic()-started))

    generate()
    original_meta = (inputs / "production_source.json").read_bytes()
    first = production.read(inputs / "production_source.json")
    cache.discard(folder, "first_control_generation_completed")
    with cache.preserve_regenerated_metadata(production, folder):
        generate()
    assert (inputs / "production_source.json").read_bytes() == original_meta
    assert cache.digest(inputs / "production_signal.npz") == first["signal_sha256"]
    cache.discard(folder, "second_control_generation_completed")
    result = dict(status="passed_short_native_cache_control", samples=len(times), duration_s=duration, dt_s=dt,
        bitwise_AE_and_NPZ_regeneration=True, original_metadata_bytes_preserved=True,
        no_generated_waveforms_remaining=not any(inputs.glob("production_signal*.npz")),
        source=asdict(source), convention=asdict(convention),
        scope="Six-hour deterministic native regeneration and serialization control; no annual FF or Euler scientific run")
    production.save(out / "validation_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
