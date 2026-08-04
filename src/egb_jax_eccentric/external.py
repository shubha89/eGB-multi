"""Adapters for optional LISA ecosystem packages."""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
from numpy.typing import NDArray

from .geometry import LISAState


def has_package(module: str) -> bool:
    return find_spec(module) is not None


def default_lisaorbits(kind: str = "equal"):
    """Construct a `lisaorbits` orbit model lazily."""

    if not has_package("lisaorbits"):
        raise ImportError("lisaorbits is not installed")
    from lisaorbits import EqualArmlengthOrbits, KeplerianOrbits

    match kind:
        case "equal":
            return EqualArmlengthOrbits()
        case "keplerian":
            return KeplerianOrbits()
        case "esa-trailing":
            # This installed lisaorbits package does not ship an ESA OEM orbit
            # file. KeplerianOrbits is the bundled unequal-arm, second-order
            # trailing analytic orbit model; the mean anomaly phase places the
            # constellation barycenter behind Earth by about 20 deg at t=0.
            return KeplerianOrbits(m_init1=-np.deg2rad(20.0))
        case _:
            raise ValueError("kind must be 'equal', 'keplerian', or 'esa-trailing'")


def state_from_lisaorbits(orbits, t: NDArray[np.float64]) -> LISAState:
    """Convert a `lisaorbits` object into this package's `LISAState`.

    `lisaorbits` uses 1-based spacecraft labels and SI meters. The internal
    response model uses zero-based spacecraft labels and light-seconds.
    """

    from .constants import C_M_PER_S

    times = np.asarray(t, dtype=np.float64)
    positions_m = np.asarray(orbits.compute_position(times, [1, 2, 3]), dtype=np.float64)
    if positions_m.shape[:2] != (times.size, 3):
        positions_m = np.moveaxis(positions_m, 0, 1)
    if positions_m.shape != (times.size, 3, 3):
        raise ValueError(f"unexpected lisaorbits position shape {positions_m.shape}")
    return LISAState(t=times, positions=positions_m / C_M_PER_S)
