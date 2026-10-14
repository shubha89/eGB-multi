"""LISA spacecraft geometry and eccentric orbit approximation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import AU_LIGHT_S, LISA_ARM_LIGHT_S, YEAR_ANGULAR_FREQUENCY


@dataclass(frozen=True)
class LISAOrbitConfig:
    """Configuration for the analytic LISA orbit approximation.

    Positions are returned in light-seconds. The default eccentricity is the
    small-e value implied by a 2.5 Gm arm length at 1 AU.
    """

    arm_length: float = LISA_ARM_LIGHT_S
    orbital_radius: float = AU_LIGHT_S
    eccentricity: float | None = None
    alpha0: float = 0.0
    xi0: float = 0.0

    @property
    def e(self) -> float:
        if self.eccentricity is not None:
            return self.eccentricity
        return self.arm_length / (2.0 * np.sqrt(3.0) * self.orbital_radius)


@dataclass(frozen=True)
class LISAState:
    """Spacecraft positions and derived relative geometry."""

    t: NDArray[np.float64]
    positions: NDArray[np.float64]

    @property
    def barycenter(self) -> NDArray[np.float64]:
        return np.mean(self.positions, axis=1)

    @property
    def relative_positions(self) -> NDArray[np.float64]:
        return self.positions - self.barycenter[:, None, :]

    def arm_vector(self, sender: int, receiver: int, *, relative: bool = True) -> NDArray[np.float64]:
        positions = self.relative_positions if relative else self.positions
        return positions[:, receiver, :] - positions[:, sender, :]

    def arm_length(self, sender: int, receiver: int, *, relative: bool = True) -> NDArray[np.float64]:
        return np.linalg.norm(self.arm_vector(sender, receiver, relative=relative), axis=1)

    def unit_arm(self, sender: int, receiver: int, *, relative: bool = True) -> NDArray[np.float64]:
        vector = self.arm_vector(sender, receiver, relative=relative)
        length = np.linalg.norm(vector, axis=1)
        if np.any(length == 0.0):
            raise ValueError("coincident spacecraft positions produce a zero-length arm")
        return vector / length[:, None]


def lisa_orbit(t: ArrayLike, config: LISAOrbitConfig | None = None) -> LISAState:
    """Return second-order eccentric analytic spacecraft positions.

    The formula is the standard small-e, cartwheeling LISA constellation used in
    low-frequency response studies. It is suitable for developing the response
    formalism and can be replaced by ephemeris or exact mission orbits later.
    """

    cfg = config or LISAOrbitConfig()
    times = np.atleast_1d(np.asarray(t, dtype=np.float64))
    alpha = YEAR_ANGULAR_FREQUENCY * times + cfg.alpha0
    e = cfg.e
    r = cfg.orbital_radius

    positions = np.empty((times.size, 3, 3), dtype=np.float64)
    for spacecraft in range(3):
        beta = 2.0 * np.pi * spacecraft / 3.0 + cfg.xi0
        a_minus_b = alpha - beta

        positions[:, spacecraft, 0] = (
            r * np.cos(alpha)
            + 0.5 * e * r * (np.cos(2.0 * alpha - beta) - 3.0 * np.cos(beta))
            + 0.125
            * e**2
            * r
            * (3.0 * np.cos(3.0 * alpha - 2.0 * beta) - 10.0 * np.cos(alpha) - 5.0 * np.cos(alpha - 2.0 * beta))
        )
        positions[:, spacecraft, 1] = (
            r * np.sin(alpha)
            + 0.5 * e * r * (np.sin(2.0 * alpha - beta) - 3.0 * np.sin(beta))
            + 0.125
            * e**2
            * r
            * (3.0 * np.sin(3.0 * alpha - 2.0 * beta) - 10.0 * np.sin(alpha) + 5.0 * np.sin(alpha - 2.0 * beta))
        )
        positions[:, spacecraft, 2] = (
            -np.sqrt(3.0) * e * r * np.cos(a_minus_b)
            + np.sqrt(3.0) * e**2 * r * (np.cos(a_minus_b) ** 2 + 2.0 * np.sin(a_minus_b) ** 2)
        )

    return LISAState(t=times, positions=positions)
