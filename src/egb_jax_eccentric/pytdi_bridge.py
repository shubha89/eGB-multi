"""pyTDI bridge for JAX eccentric link responses."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .eccentric import EccentricBinaryParams
from .eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry
from .geometry import LISAState

LINKS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 2),
    (2, 1),
    (1, 0),
)
# Rosetta Stone / pyTDI convention: label ij is received at i, emitted by j.
LINK_LABELS: tuple[str, ...] = ("21", "32", "13", "31", "23", "12")


def _sampling_frequency(t: NDArray[np.float64]) -> float:
    if t.size < 2:
        raise ValueError("at least two time samples are required")
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1e-9, atol=1e-12):
        raise ValueError("pyTDI data requires a uniformly sampled time grid")
    return float(1.0 / dt[0])


def pytdi_data_from_links(
    state: LISAState,
    links: dict[str, NDArray[np.complex128]],
    *,
    compute_delay_derivatives: bool = True,
):
    """Create a `pytdi.Data` object from six GW-only link responses.

    The link responses are assigned to `sci_ij`. Local metrology/reference
    measurements are set to zero, so pyTDI's intermediate variables reduce to
    the supplied GW link content.
    """

    try:
        from pytdi import Data
    except ImportError as exc:
        raise ImportError("pytdi is required for pytdi_data_from_links") from exc

    missing = sorted(set(LINK_LABELS) - set(links))
    if missing:
        raise ValueError(f"missing link responses for {missing}")

    zero = np.zeros(state.t.size, dtype=np.complex128)
    measurements = {key: zero for key in Data.MEASUREMENTS}
    for label in LINK_LABELS:
        value = np.asarray(links[label], dtype=np.complex128)
        if value.shape != (state.t.size,):
            raise ValueError(f"link {label} has shape {value.shape}, expected {(state.t.size,)}")
        measurements[f"sci_{label}"] = value

    delays = {}
    for label, (sender, receiver) in zip(LINK_LABELS, LINKS, strict=True):
        delays[f"d_{label}"] = state.arm_length(sender, receiver)

    data = Data(measurements, delays, _sampling_frequency(state.t))
    if compute_delay_derivatives:
        data.compute_delay_derivatives()
    return data


def xyz_from_links(
    state: LISAState,
    links: dict[str, NDArray[np.complex128]],
    *,
    generation: int = 1,
    measurement_order: int = 3,
    delay_order: int = 3,
) -> dict[str, NDArray[np.complex128]]:
    """Compute pyTDI Michelson X/Y/Z from six GW-only link responses.

    With zero reference/metrology channels, eta_ij equals sci_ij exactly.
    Pass these intermediates directly to avoid interpolating zero channels.
    """

    try:
        from pytdi.michelson import compute_factorized_michelson
    except ImportError as exc:
        raise ImportError("pytdi is required for xyz_from_links") from exc

    data = pytdi_data_from_links(state, links)
    etas = {f"eta_{label}": data.measurements[f"sci_{label}"] for label in LINK_LABELS}
    return {
        "X": compute_factorized_michelson(
            data,
            etas=etas,
            rot=0,
            order=measurement_order,
            delay_order=delay_order,
            generation=generation,
            unit="frequency",
        ),
        "Y": compute_factorized_michelson(
            data,
            etas=etas,
            rot=1,
            order=measurement_order,
            delay_order=delay_order,
            generation=generation,
            unit="frequency",
        ),
        "Z": compute_factorized_michelson(
            data,
            etas=etas,
            rot=2,
            order=measurement_order,
            delay_order=delay_order,
            generation=generation,
            unit="frequency",
        ),
    }


def prepare_xyz_from_links(
    state: LISAState,
    *,
    generation: int = 1,
    measurement_order: int = 3,
    delay_order: int = 3,
) -> Callable[[dict[str, NDArray]], dict[str, NDArray]]:
    """Prepare reusable pyTDI XYZ operators for a fixed time grid and orbit.

    The returned callable accepts six GW-only links on exactly this grid.
    Nested delays and frequency-unit Doppler factors are computed once using
    pyTDI; measurement interpolation is still evaluated on every call. Both
    generations retain pyTDI's factorized evaluation order, including its
    boundary behavior and time-varying delays. No constant-arm approximation
    is introduced. Reference/metrology/noise beatnotes are not accepted.

    Prepare a new callable if times, positions, or interpolation settings
    change. Cache per block for large datasets: stored delay arrays use memory
    proportional to the number of samples. The callable does not cache source
    links and can be reused as source parameters change.
    """
    from pytdi.core import LISATDICombination

    if generation not in (1, 2):
        raise ValueError("generation must be 1 or 2")
    for name, order in (("measurement_order", measurement_order), ("delay_order", delay_order)):
        if not isinstance(order, (int, np.integer)) or order < 1 or order % 2 != 1:
            raise ValueError(f"{name} must be a positive odd integer")
    size = state.t.size
    zeros = np.zeros(size)
    data = pytdi_data_from_links(state, dict.fromkeys(LINK_LABELS, zeros))

    def build(components):
        return LISATDICombination(components).build(**data.args, order=delay_order)

    def prepare_channel(labels):
        # Same factorization as pytdi.michelson.compute_factorized_michelson.
        # Ordered delay lists must not be reordered: unequal-arm delays do not
        # generally commute. Frequency-unit derivatives are retained by build.
        a, b, c, d = labels
        arm1 = build({f"eta_{a}": [(1, [])], f"eta_{b}": [(1, [f"D_{a}"])]})
        arm2 = build({f"eta_{c}": [(1, [])], f"eta_{d}": [(1, [f"D_{c}"])]})
        ab, cd = [f"D_{a}", f"D_{b}"], [f"D_{c}", f"D_{d}"]
        if generation == 1:
            final = build({"x_arm_1": [(-1, []), (1, cd)],
                           "x_arm_2": [(1, []), (-1, ab)]})

            def evaluate(etas):
                arms = {"x_arm_1": arm1(etas, order=measurement_order, unit="frequency"),
                        "x_arm_2": arm2(etas, order=measurement_order, unit="frequency")}
                return final(arms, order=measurement_order, unit="frequency")
        else:
            round1 = build({"arm_a": [(1, [])], "arm_b": [(1, ab)]})
            round2 = build({"arm_a": [(1, cd)], "arm_b": [(1, [])]})
            final = build({"roundtrip_a": [(-1, []), (1, cd + ab)],
                           "roundtrip_b": [(1, []), (-1, ab + cd)]})

            def evaluate(etas):
                arms = {"arm_a": arm1(etas, order=measurement_order, unit="frequency"),
                        "arm_b": arm2(etas, order=measurement_order, unit="frequency")}
                rounds = {"roundtrip_a": round1(arms, order=measurement_order, unit="frequency"),
                          "roundtrip_b": round2(arms, order=measurement_order, unit="frequency")}
                return final(rounds, order=measurement_order, unit="frequency")
        return evaluate

    channels = dict(zip("XYZ", [prepare_channel(labels) for labels in
        (("12", "21", "13", "31"), ("23", "32", "21", "12"), ("31", "13", "32", "23"))], strict=True))

    def evaluate_xyz(links):
        missing = sorted(set(LINK_LABELS) - set(links))
        if missing:
            raise ValueError(f"missing link responses for {missing}")
        etas = {}
        for label in LINK_LABELS:
            values = np.asarray(links[label], dtype=np.complex128)
            if values.shape != (size,):
                raise ValueError(f"link {label} has shape {values.shape}, expected {(size,)}")
            etas[f"eta_{label}"] = values
        return {name: channel(etas) for name, channel in channels.items()}

    return evaluate_xyz


def aet_from_xyz(xyz: dict[str, NDArray]) -> dict[str, NDArray]:
    """Convert Michelson X/Y/Z to orthogonal A/E/T channels."""

    x = np.asarray(xyz["X"])
    y = np.asarray(xyz["Y"])
    z = np.asarray(xyz["Z"])
    return {
        "A": (z - x) / np.sqrt(2.0),
        "E": (x - 2.0 * y + z) / np.sqrt(6.0),
        "T": (x + y + z) / np.sqrt(3.0),
    }


def eccentric_xyz_jax(
    state: LISAState,
    sources: EccentricBinaryParams | list[EccentricBinaryParams],
    *,
    geometry: dict[str, NDArray[np.float64]] | None = None,
    batch_size: int = 1024,
    physics_mode: str = "1pn",
    generation: int = 1,
    measurement_order: int = 3,
    delay_order: int = 3,
) -> dict[str, NDArray[np.complex128]]:
    """Compute JAX eccentric links and route them through pyTDI Michelson XYZ."""

    if geometry is None:
        geometry = precompute_jax_link_geometry(state)
    links = eccentric_links_jax(
        sources,
        geometry,
        batch_size=batch_size,
        physics_mode=physics_mode,
    )
    return xyz_from_links(
        state,
        links,
        generation=generation,
        measurement_order=measurement_order,
        delay_order=delay_order,
    )


def eccentric_aet_jax(
    state: LISAState,
    sources: EccentricBinaryParams | list[EccentricBinaryParams],
    **kwargs,
) -> dict[str, NDArray]:
    """Compute JAX eccentric links, pyTDI XYZ, and orthogonal A/E/T channels."""

    return aet_from_xyz(eccentric_xyz_jax(state, sources, **kwargs))
