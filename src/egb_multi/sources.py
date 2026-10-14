"""Source parameterizations shared by local and external response backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .tensors import matrix_to_tracefree5, normalize_direction


@dataclass(frozen=True)
class GalacticBinaryParams:
    """Non-eccentric galactic-binary parameters in FastGB convention.

    Parameters are ordered as FastGB expects them:

    `[f0, fdot, ampl, beta, lambda, psi, incl, phi0]`

    where `beta` is ecliptic latitude and `lambda_` is ecliptic longitude.
    """

    f0: float
    fdot: float = 0.0
    amplitude: float = 1.0
    beta: float = 0.0
    lambda_: float = 0.0
    psi: float = 0.0
    inclination: float = 0.0
    phi0: float = 0.0

    def as_fastgb_array(self) -> NDArray[np.float64]:
        return np.array(
            [
                self.f0,
                self.fdot,
                self.amplitude,
                self.beta,
                self.lambda_,
                self.psi,
                self.inclination,
                self.phi0,
            ],
            dtype=np.float64,
        )

    def propagation_direction(self) -> NDArray[np.float64]:
        """Return FastGB's propagation direction vector `k`."""

        cos_beta = np.cos(self.beta)
        return normalize_direction(
            [
                -cos_beta * np.cos(self.lambda_),
                -cos_beta * np.sin(self.lambda_),
                -np.sin(self.beta),
            ]
        )

    def polarization_matrix(self) -> NDArray[np.complex128]:
        """Return the complex trace-free polarization matrix used locally.

        This mirrors the polarization convention in FastGB's Python
        implementation, including amplitude, inclination and polarization angle.
        """

        theta = np.pi / 2.0 - self.beta
        sinth, costh = np.sin(theta), np.cos(theta)
        sinph, cosph = np.sin(self.lambda_), np.cos(self.lambda_)

        u = np.array([costh * cosph, costh * sinph, -sinth], dtype=np.float64)
        v = np.array([sinph, -cosph, 0.0], dtype=np.float64)
        eplus = np.outer(v, v) - np.outer(u, u)
        ecross = np.outer(u, v) + np.outer(v, u)

        cosiota = np.cos(self.inclination)
        cosps = np.cos(2.0 * self.psi)
        sinps = np.sin(2.0 * self.psi)
        aplus = self.amplitude * (1.0 + cosiota * cosiota)
        across = -2.0 * self.amplitude * cosiota
        dp = aplus * cosps - 1.0j * across * sinps
        dc = -aplus * sinps - 1.0j * across * cosps
        return dp * eplus + dc * ecross

    def tracefree_coefficients(self) -> NDArray[np.complex128]:
        return matrix_to_tracefree5(self.polarization_matrix())

    def slow_envelope(self, t: NDArray[np.float64], *, t0: float = 0.0) -> NDArray[np.complex128]:
        """Return a simple carrier-stripped phase envelope.

        Eccentricity is intentionally out of scope here. The phase convention is
        local to our triliear experiments and should be benchmarked against the
        chosen external backend before using residuals scientifically.
        """

        dt = np.asarray(t, dtype=np.float64) - t0
        return np.exp(1.0j * (self.phi0 + np.pi * self.fdot * dt**2))
