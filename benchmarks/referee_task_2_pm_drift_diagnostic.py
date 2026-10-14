"""Initial-rate PM drift estimates; not waveforms or fitting-factor results."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egb_multi.constants import JULIAN_YEAR_S
from egb_multi.eccentric import EccentricBinaryParams, peters_mathews_derivatives


def run():
    rows = []
    for f0 in (0.001, 0.003):
        source = EccentricBinaryParams(mean_motion=math.pi * f0,
            eccentricity=0.1, m1_solar=0.5, m2_solar=0.5)
        ndot, edot = peters_mathews_derivatives(source.mean_motion,
            source.eccentricity, source.total_mass_kg, source.symmetric_mass_ratio)
        fdot = float(ndot) / math.pi
        rows.append(dict(f0_hz=f0, initial_eccentricity=source.eccentricity,
            masses_solar=[0.5, 0.5], duration_s=JULIAN_YEAR_S,
            initial_fdot_hz_s=fdot,
            leading_carrier_phase_drift_rad=math.pi * fdot * JULIAN_YEAR_S**2,
            initial_rate_delta_e=float(edot) * JULIAN_YEAR_S))
    files = [Path(__file__), ROOT / "src/egb_multi/constants.py",
        ROOT / "src/egb_multi/eccentric.py"]
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_root=str(ROOT / "src/egb_multi"), rows=rows,
        formulas=dict(carrier="f0=n0/pi", phase="pi*initial_fdot*T**2",
            eccentricity="initial_edot*T"),
        scope="Initial-rate leading PM drift relative to a frozen Newtonian carrier. "
            "Not an integrated 1PN phase, optimized overlap, or fitting-factor error estimate.",
        production_model="Full Peters-Mathews injections; independently fitted linear chirp in circular templates.",
        manuscript_difference="Saved manuscript uses automatic PM switching at mismatch tolerance 1e-2; "
            "frozen production always evolves injections. Do not infer uniform insignificance from these two examples.",
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    out = ROOT / "outputs/referee/task_2_9_20261013_pm_drift_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift_estimates.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
