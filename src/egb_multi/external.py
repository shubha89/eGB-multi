"""Adapters for optional LISA ecosystem packages."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from .geometry import LISAState
from .sources import GalacticBinaryParams


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


@lru_cache(maxsize=4)
def cartesian_frame_rotation(from_frame: str, to_frame: str) -> NDArray[np.float64]:
    """Rotation between ICRS and barycentric mean ecliptic J2000.

    Use Astropy's frame-bias/obliquity transformation, also used by lisaorbits,
    rather than a fitted angle or an approximate fixed obliquity.
    """
    if from_frame not in {'icrs','ecliptic'} or to_frame not in {'icrs','ecliptic'}:
        raise ValueError('frame must be icrs or ecliptic (BME J2000)')
    if from_frame == to_frame:
        matrix=np.eye(3)
    else:
        import astropy.units as units
        from astropy.coordinates import ICRS, BarycentricMeanEcliptic, CartesianRepresentation
        from astropy.time import Time
        representation=CartesianRepresentation(np.eye(3)*units.m)
        frame=BarycentricMeanEcliptic(equinox=Time('J2000')) if from_frame=='ecliptic' else ICRS()
        target=ICRS() if to_frame=='icrs' else BarycentricMeanEcliptic(equinox=Time('J2000'))
        matrix=frame.realize_frame(representation).transform_to(target).cartesian.xyz.to_value(units.m)
    matrix.setflags(write=False)
    return matrix


def state_from_lisaorbits(orbits, t: NDArray[np.float64], *, frame: str = 'ecliptic') -> LISAState:
    """Convert a `lisaorbits` object into this package's `LISAState`.

    `lisaorbits` uses 1-based spacecraft labels and SI meters. The internal
    response model uses zero-based spacecraft labels and light-seconds.
    Native positions are ICRS. The default converts to ecliptic for source beta,
    lambda and polarization basis are barycentric mean ecliptic J2000.
    Explicit frame='icrs' returns native Cartesian positions, including for
    reproducing historical runs that omitted the frame conversion.
    """

    from .constants import C_M_PER_S

    times = np.asarray(t, dtype=np.float64)
    positions_m = np.asarray(orbits.compute_position(times, [1, 2, 3]), dtype=np.float64)
    if positions_m.shape[:2] != (times.size, 3):
        positions_m = np.moveaxis(positions_m, 0, 1)
    if positions_m.shape != (times.size, 3, 3):
        raise ValueError(f"unexpected lisaorbits position shape {positions_m.shape}")
    rotation=cartesian_frame_rotation('icrs',frame)
    return LISAState(t=times, positions=(positions_m @ rotation.T) / C_M_PER_S)


def link_delays_from_lisaorbits(orbits, t: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    """Deterministic reception-time TCB light travel times in seconds.

    Native label ij means received at i, emitted by j. No instrument noise,
    clock measurements or measured pseudoranges enter this adapter. These
    delays are appropriate to a clean response with ideal timing/ranging.
    """
    from .tdi import LINK_LABELS

    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or np.any(~np.isfinite(times)):
        raise ValueError('Reception times must be a nonempty finite vector')
    values = np.asarray(orbits.compute_ltt(times, [int(label) for label in LINK_LABELS]), dtype=np.float64)
    if values.shape != (times.size, len(LINK_LABELS)) or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('Invalid reception-time light travel times')
    return {label: values[:, index].copy() for index, label in enumerate(LINK_LABELS)}


@dataclass(frozen=True)
class FastGBResult:
    x: NDArray[np.complex128]
    y: NDArray[np.complex128]
    z: NDArray[np.complex128]
    kmin: NDArray[np.int32]

    @property
    def xyz(self) -> NDArray[np.complex128]:
        return np.stack([self.x, self.y, self.z], axis=1)


class FastGBBackend:
    """Thin wrapper around `fastgb.fastgb.FastGB`."""

    def __init__(
        self,
        *,
        delta_t: float,
        duration: float,
        n_samples: int,
        orbits=None,
        t0: float = 0.0,
        tinit: float = 0.0,
    ) -> None:
        if not has_package("fastgb"):
            raise ImportError("fastgb is not installed")
        if orbits is None:
            orbits = default_lisaorbits("equal")
        from fastgb import fastgb

        self.model = fastgb.FastGB(orbits=orbits, delta_t=delta_t, T=duration, N=n_samples, t0=t0, tinit=tinit)

    def frequency_domain_xyz(self, sources: GalacticBinaryParams | list[GalacticBinaryParams], *, tdi2: bool = False) -> FastGBResult:
        if isinstance(sources, GalacticBinaryParams):
            source_list = [sources]
        else:
            source_list = sources
        params = np.stack([source.as_fastgb_array() for source in source_list], axis=0)
        x, y, z, kmin = self.model.get_fd_tdixyz(params, tdi2=tdi2)
        return FastGBResult(
            x=np.asarray(x, dtype=np.complex128),
            y=np.asarray(y, dtype=np.complex128),
            z=np.asarray(z, dtype=np.complex128),
            kmin=np.asarray(kmin, dtype=np.int32),
        )
