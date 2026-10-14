"""Compute pyTDI A/E/T from six exact-link GW responses.

Run with:

    python examples/triliear_pytdi_aet.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi import GalacticBinaryParams, compute_aet, lisa_orbit


def main() -> None:
    times = np.arange(4096.0)
    state = lisa_orbit(times)
    source = GalacticBinaryParams(
        f0=1.0e-3,
        amplitude=1.0e-22,
        beta=0.2,
        lambda_=0.4,
        psi=0.7,
        inclination=1.1,
        phi0=0.3,
    )

    aet = compute_aet(state, source, generation=1, measurement_order=5, delay_order=5)
    for channel in ("A", "E", "T"):
        rms = np.sqrt(np.mean(np.abs(aet[channel]) ** 2))
        print(f"{channel}: rms={rms:.6e}")


if __name__ == "__main__":
    main()
