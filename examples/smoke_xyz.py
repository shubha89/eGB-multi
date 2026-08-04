"""Small pyTDI XYZ smoke example for the JAX eccentric response."""

from __future__ import annotations

import numpy as np

from egb_jax_eccentric import EccentricBinaryParams, eccentric_xyz_jax, lisa_orbit


def main() -> None:
    times = np.arange(512.0)
    state = lisa_orbit(times)
    source = EccentricBinaryParams(
        mean_motion=np.pi * 1.0e-4,
        eccentricity=0.05,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.8,
        phi0=0.2,
    )

    xyz = eccentric_xyz_jax(state, source, batch_size=1, physics_mode="1pn")
    for channel, value in xyz.items():
        print(channel, value.shape, np.sqrt(np.mean(np.abs(value) ** 2)))


if __name__ == "__main__":
    main()
