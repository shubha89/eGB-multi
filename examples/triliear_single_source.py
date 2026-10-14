"""Single-source triliear response example.

Run with:

    python examples/triliear_single_source.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi import lisa_orbit, triliear_tensor, triliear_waveform
from egb_multi.constants import JULIAN_YEAR_S


def main() -> None:
    f0 = 1.0e-4
    times = np.linspace(0.0, JULIAN_YEAR_S, 1024)
    state = lisa_orbit(times)

    k = np.array([0.2, -0.4, 0.7])
    k = k / np.linalg.norm(k)
    e = np.array([1.0, 0.2j, -0.1, 0.3, -0.4j])

    envelope = np.ones_like(times, dtype=np.complex128)
    a = triliear_tensor(state, f0)
    w = triliear_waveform(envelope, a, e, k)

    print(f"samples: {times.size}")
    print(f"channels: {w.shape[1]}")
    print("rms channel amplitudes:", np.sqrt(np.mean(np.abs(w) ** 2, axis=0)))


if __name__ == "__main__":
    main()
