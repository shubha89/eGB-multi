"""FastGB-convention response helpers for eccentric-source comparisons.

This module keeps FastGB's frequency-domain response algebra and swaps the
source content for arbitrary harmonic trace-free polarization components. It is
intended for apples-to-apples convention checks against FastGB, not as the
default realistic time-domain LISA response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import C_M_PER_S
from .eccentric import (
    EccentricBinaryParams,
    eccentric_harmonic_angular_frequency,
    eccentric_harmonic_indices,
    eccentric_harmonic_projection_angular_frequency,
    eccentric_polarizations,
)
from .external import FastGBResult
from .sources import GalacticBinaryParams
from .tensors import matrix_to_tracefree5, tracefree_matrix


@dataclass(frozen=True)
class FastGBConventionComponent:
    """One arbitrary-polarization harmonic in FastGB's response convention."""

    frequency: float
    tracefree_coefficients: NDArray[np.complex128]
    propagation_direction: NDArray[np.float64]
    phase0: float = 0.0


def circular_component_from_fastgb_source(source: GalacticBinaryParams) -> FastGBConventionComponent:
    """Return the exact circular component matching FastGB's source convention.

    FastGB flips ``phi0`` internally before constructing the slow response, so
    the component stores ``phase0=-source.phi0``.
    """

    return FastGBConventionComponent(
        frequency=source.f0,
        tracefree_coefficients=source.tracefree_coefficients(),
        propagation_direction=source.propagation_direction(),
        phase0=-source.phi0,
    )


def eccentric_components_for_fastgb_response(
    source: EccentricBinaryParams,
    *,
    order: int = 1,
    samples: int = 512,
    circular_reference: GalacticBinaryParams | None = None,
) -> list[FastGBConventionComponent]:
    """Return eccentric harmonic components for the FastGB response algebra."""

    if samples < 32:
        raise ValueError("samples must be at least 32")
    projection_frequency = eccentric_harmonic_projection_angular_frequency(source)
    period = 2.0 * np.pi / projection_frequency
    sample_times = source.t0 + np.linspace(0.0, period, samples, endpoint=False)
    hplus, hcross = eccentric_polarizations(source, sample_times)

    eplus, ecross = source.polarization_basis()
    cosps = np.cos(2.0 * source.psi)
    sinps = np.sin(2.0 * source.psi)
    fastgb_eplus = cosps * eplus - sinps * ecross
    fastgb_ecross = sinps * eplus + cosps * ecross
    direction = source.propagation_direction()

    components: list[FastGBConventionComponent] = []
    for harmonic in eccentric_harmonic_indices(order):
        angular_frequency = eccentric_harmonic_angular_frequency(source, harmonic)
        carrier_projection = np.exp(-1.0j * angular_frequency * sample_times)
        hplus_coeff = np.mean(hplus * carrier_projection)
        hcross_coeff = np.mean(hcross * carrier_projection)
        # The minus sign on h_cross is required by FastGB's circular
        # polarization convention. Without it, the e -> 0 component is not
        # equivalent to FastGB's standard GB source.
        matrix = 2.0 * hplus_coeff * fastgb_eplus - 2.0 * hcross_coeff * fastgb_ecross
        components.append(
            FastGBConventionComponent(
                frequency=angular_frequency / (2.0 * np.pi),
                tracefree_coefficients=matrix_to_tracefree5(matrix),
                propagation_direction=direction,
            )
        )
    if circular_reference is None:
        return components

    if 2 not in eccentric_harmonic_indices(order):
        raise ValueError("circular calibration requires an eccentric order that includes harmonic 2")
    circular_source = EccentricBinaryParams(**{**source.__dict__, "eccentricity": 0.0})
    circular_component = eccentric_components_for_fastgb_response(
        circular_source,
        order=0,
        samples=samples,
        circular_reference=None,
    )[0]
    reference_component = circular_component_from_fastgb_source(circular_reference)
    denom = np.vdot(circular_component.tracefree_coefficients, circular_component.tracefree_coefficients)
    scale = 0.0j if denom == 0 else np.vdot(circular_component.tracefree_coefficients, reference_component.tracefree_coefficients) / denom
    return [
        FastGBConventionComponent(
            frequency=component.frequency,
            tracefree_coefficients=scale * component.tracefree_coefficients,
            propagation_direction=component.propagation_direction,
            phase0=component.phase0,
        )
        for component in components
    ]


def _component_slow_part(model, component: FastGBConventionComponent) -> tuple[NDArray[np.complex128], int]:
    """Construct FastGB's slow one-way response ``Gs`` for one component."""

    tm = np.asarray(model.tm, dtype=np.float64)
    positions = np.asarray(model.position, dtype=np.float64)
    arm_length = float(model.arm_length)
    frequency = float(component.frequency)
    direction = np.asarray(component.propagation_direction, dtype=np.float64)
    matrix = tracefree_matrix(component.tracefree_coefficients)

    r = np.zeros((4, 3, tm.size), dtype=np.float64)
    r[0] = positions[1] - positions[0]  # 12
    r[1] = positions[2] - positions[0]  # 13
    r[2] = positions[2] - positions[1]  # 23
    r[3] = -r[1]  # 31
    r /= arm_length

    kdotr = np.zeros((6, tm.size), dtype=np.float64)
    kdotr[::2] = np.dot(direction, r[:3])
    kdotr[1::2] = -kdotr[::2]

    kdotp = np.dot(direction, positions) / C_M_PER_S
    fstar = C_M_PER_S / (arm_length * 2.0 * np.pi)
    fonfs = frequency / fstar

    q = int(np.rint(frequency * model.T))
    angular_bin = 2.0 * np.pi * q / model.T
    omega = 2.0 * np.pi * frequency

    # A convention: 12, 23, 31.
    arm_indices = np.array([0, 2, 3])
    arm_projection = np.einsum("ati,ij,atj->at", r[arm_indices].transpose(0, 2, 1), matrix, r[arm_indices].transpose(0, 2, 1))

    slow_phase = component.phase0 + (omega - angular_bin) * tm
    spacecraft_phase = omega * kdotp - slow_phase[None, :]

    # Gs convention: 12, 23, 31, 21, 32, 13.
    kdotr_indices = np.array([0, 4, 3, 1, 5, 2])
    spacecraft_indices = np.array([0, 1, 2, 1, 2, 0])
    projection_indices = np.array([0, 1, 2, 0, 1, 2])
    arg = 0.5 * fonfs * (1.0 + kdotr[kdotr_indices])
    slow = (
        0.25
        * np.sinc(arg / np.pi)
        * np.exp(-1.0j * (arg + spacecraft_phase[spacecraft_indices]))
        * arm_projection[projection_indices]
    )
    return slow.astype(np.complex128), q


def _compute_fastgb_xyz(model, slow: NDArray[np.complex128], q: int, frequency: float) -> tuple[NDArray[np.complex128], int]:
    """Apply FastGB's XYZ algebra to one slow-response component."""

    fstar = C_M_PER_S / (float(model.arm_length) * 2.0 * np.pi)
    om_l = frequency / fstar
    transfer = 4.0 * om_l * np.sin(om_l) * np.exp(-1.0j * om_l)

    x_slow = slow[3] - slow[2] + (slow[0] - slow[5]) * np.exp(-1.0j * om_l)
    y_slow = slow[4] - slow[0] + (slow[1] - slow[3]) * np.exp(-1.0j * om_l)
    z_slow = slow[5] - slow[1] + (slow[2] - slow[4]) * np.exp(-1.0j * om_l)

    xyz_slow = transfer * np.stack([x_slow, y_slow, z_slow], axis=0)
    xyz = np.fft.fftshift(np.fft.fft(xyz_slow, axis=1), axes=1)
    xyz *= 0.5 * model.T / model.N
    kmin = int(q - model.N / 2)
    return xyz.astype(np.complex128), kmin


def frequency_domain_xyz_from_components(
    fastgb_backend,
    components: FastGBConventionComponent | list[FastGBConventionComponent],
) -> FastGBResult:
    """Compute FastGB-convention FD X/Y/Z for arbitrary harmonic components.

    Components with different central bins are returned on a common sparse
    frequency window spanning all component windows.
    """

    component_list = [components] if isinstance(components, FastGBConventionComponent) else components
    if not component_list:
        raise ValueError("at least one component is required")

    pieces = []
    kmins = []
    for component in component_list:
        slow, q = _component_slow_part(fastgb_backend.model, component)
        xyz, kmin = _compute_fastgb_xyz(fastgb_backend.model, slow, q, component.frequency)
        pieces.append(xyz)
        kmins.append(kmin)

    global_kmin = min(kmins)
    global_kmax = max(kmin + fastgb_backend.model.N for kmin in kmins)
    out = np.zeros((3, global_kmax - global_kmin), dtype=np.complex128)
    for xyz, kmin in zip(pieces, kmins, strict=True):
        start = kmin - global_kmin
        out[:, start : start + fastgb_backend.model.N] += xyz

    return FastGBResult(
        x=out[0][None, :],
        y=out[1][None, :],
        z=out[2][None, :],
        kmin=np.array([global_kmin], dtype=np.int32),
    )


def fastgb_convention_circular_xyz(fastgb_backend, source: GalacticBinaryParams) -> FastGBResult:
    """Compute circular X/Y/Z through the local FastGB-convention algebra."""

    return frequency_domain_xyz_from_components(fastgb_backend, circular_component_from_fastgb_source(source))


def fastgb_convention_eccentric_xyz(
    fastgb_backend,
    source: EccentricBinaryParams,
    *,
    order: int = 1,
    samples: int = 512,
    circular_reference: GalacticBinaryParams | None = None,
) -> FastGBResult:
    """Compute eccentric X/Y/Z using FastGB's response algebra."""

    return frequency_domain_xyz_from_components(
        fastgb_backend,
        eccentric_components_for_fastgb_response(
            source,
            order=order,
            samples=samples,
            circular_reference=circular_reference,
        ),
    )
