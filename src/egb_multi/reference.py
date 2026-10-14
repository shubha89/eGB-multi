"""Brute-force time-domain reference responses."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .external import has_package, cartesian_frame_rotation
from .geometry import LISAState
from .response import beta
from .sources import GalacticBinaryParams
from .tdi import LINK_LABELS, pytdi_data_from_links

# Rosetta Stone / pyTDI convention: link ij is received at i, emitted by j.
LISAGW_LINKS: tuple[int, ...] = (21, 32, 13, 31, 23, 12)


def _position_component(orbits, sc: int, component: int):
    def func(t):
        position = np.asarray(orbits.compute_position(np.asarray(t, dtype=np.float64), sc), dtype=np.float64)
        if position.ndim == 3:
            position = position[:, 0, :]
        return position[:, component]

    return func


def _light_travel_time(orbits, link: int):
    def func(t):
        return np.asarray(orbits.compute_ltt(np.asarray(t, dtype=np.float64), link), dtype=np.float64).reshape(-1)

    return func


def lisagwresponse_galactic_binary(source: GalacticBinaryParams, orbits, *, shift_sun: str = "never"):
    """Build a `lisagwresponse.GalacticBinary` from local parameters.

    The local circular-binary convention is the FastGB-like complex analytic
    signal whose physical strain is its real part.  `lisagwresponse` uses a real
    strain with phase `2 pi f t - phi0` and the opposite overall sign, so the
    adapter flips both `A` and `phi0` to make brute-force links comparable to
    `np.real(link_responses(...))`. Sky and polarization are transformed from
    barycentric mean ecliptic J2000 to ICRS; psi is a basis angle and cannot
    simply be copied while rotating the sky.
    """

    if not has_package("lisagwresponse"):
        raise ImportError("lisagwresponse is not installed")
    from lisagwresponse import GalacticBinary

    rotation=cartesian_frame_rotation('ecliptic','icrs')
    sky=-rotation @ source.propagation_direction()
    ra=np.arctan2(sky[1],sky[0])
    dec=np.arctan2(sky[2],np.hypot(sky[0],sky[1]))
    # Unit plus tensor in the source's ecliptic polarization basis, rotated
    # by psi. Project it into the native ICRS basis to obtain the spin-2 angle.
    u=np.array([np.sin(source.beta)*np.cos(source.lambda_),
                np.sin(source.beta)*np.sin(source.lambda_),-np.cos(source.beta)])
    v=np.array([np.sin(source.lambda_),-np.cos(source.lambda_),0.])
    ep=np.outer(v,v)-np.outer(u,u)
    ec=np.outer(u,v)+np.outer(v,u)
    physical_plus=rotation @ (np.cos(2*source.psi)*ep-np.sin(2*source.psi)*ec) @ rotation.T
    native_u=np.array([np.sin(ra),-np.cos(ra),0.])
    native_v=np.array([-np.sin(dec)*np.cos(ra),-np.sin(dec)*np.sin(ra),np.cos(dec)])
    native_ep=np.outer(native_u,native_u)-np.outer(native_v,native_v)
    native_ec=np.outer(native_u,native_v)+np.outer(native_v,native_u)
    psi=.5*np.arctan2(np.sum(physical_plus*native_ec),np.sum(physical_plus*native_ep))

    x = {sc: _position_component(orbits, sc, 0) for sc in (1, 2, 3)}
    y = {sc: _position_component(orbits, sc, 1) for sc in (1, 2, 3)}
    z = {sc: _position_component(orbits, sc, 2) for sc in (1, 2, 3)}
    ltt = {link: _light_travel_time(orbits, link) for link in LISAGW_LINKS}
    return GalacticBinary(
        A=-source.amplitude,
        f=source.f0,
        df=source.fdot,
        phi0=-source.phi0,
        iota=source.inclination,
        psi=psi,
        ra=ra,
        dec=dec,
        x=x,
        y=y,
        z=z,
        ltt=ltt,
        orbits="provided",
        shift_sun=shift_sun,
    )


def brute_force_links(
    source: GalacticBinaryParams,
    orbits,
    t: NDArray[np.float64],
    *,
    shift_sun: str = "never",
) -> dict[str, NDArray[np.float64]]:
    """Compute brute-force time-domain links with `lisagwresponse`."""

    gb = lisagwresponse_galactic_binary(source, orbits, shift_sun=shift_sun)
    response = gb.compute_gw_response(np.asarray(t, dtype=np.float64), np.array(LISAGW_LINKS))
    return {label: response[:, i] for i, label in enumerate(LINK_LABELS)}


def brute_force_xyz(
    state: LISAState,
    source: GalacticBinaryParams,
    orbits,
    *,
    generation: int = 1,
    measurement_order: int = 5,
    delay_order: int = 5,
    shift_sun: str = "never",
) -> dict[str, NDArray[np.complex128]]:
    """Compute pyTDI X/Y/Z from brute-force `lisagwresponse` links."""

    try:
        from pytdi.michelson import compute_factorized_michelson
    except ImportError as exc:
        raise ImportError("pytdi is required for brute_force_xyz") from exc

    data = pytdi_data_from_links(
        state,
        {key: value.astype(np.complex128) for key, value in brute_force_links(source, orbits, state.t, shift_sun=shift_sun).items()},
    )
    return {
        "X": compute_factorized_michelson(data, rot=0, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
        "Y": compute_factorized_michelson(data, rot=1, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
        "Z": compute_factorized_michelson(data, rot=2, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
    }


def brute_force_pair_channels(
    state: LISAState,
    source: GalacticBinaryParams,
    orbits,
    *,
    shift_sun: str = "never",
) -> NDArray[np.complex128]:
    """Build the paper's three pair channels from brute-force links."""

    links = brute_force_links(source, orbits, state.t, shift_sun=shift_sun)
    pairs = [("12", "21", 0, 1), ("13", "31", 0, 2), ("23", "32", 1, 2)]
    out = []
    for forward, reverse, sender, receiver in pairs:
        length = state.arm_length(sender, receiver)
        out.append(np.exp(beta(source.f0, length, fdot=source.fdot, t=state.t)) * links[forward] + links[reverse])
    return np.stack(out, axis=1)
