"""Forward response models for the triliear LISA formalism."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import LISAState
from .sources import GalacticBinaryParams
from .tensors import normalize_direction, tracefree_matrix

TDI_CHANNELS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 2, 0),
)


def beta(f0: float, arm_length: ArrayLike, *, fdot: float = 0.0, t: ArrayLike | None = None) -> NDArray[np.complex128]:
    """Return beta = -2 pi i f_eff L for a carrier frequency or chirp.

    If `fdot` is supplied, `f_eff = f0 + fdot * t` is used. This keeps the chirp
    support explicit while preserving the monochromatic Eq. (8) behavior.
    """

    length = np.asarray(arm_length, dtype=np.float64)
    if fdot and t is None:
        raise ValueError("t is required when fdot is non-zero")
    frequency = f0 if not fdot else f0 + fdot * np.asarray(t, dtype=np.float64)
    return -2j * np.pi * frequency * length


def one_way_doppler(
    state: LISAState,
    sender: int,
    receiver: int,
    k: ArrayLike,
    e: ArrayLike,
    h,
) -> NDArray[np.complex128]:
    """Evaluate the one-way Doppler response in Eq. (6).

    `h` must be a vectorized callable accepting retarded times in seconds.
    """

    direction = normalize_direction(k)
    matrix = tracefree_matrix(e)
    n = state.unit_arm(sender, receiver)
    length = state.arm_length(sender, receiver)
    p = state.positions

    nen = np.einsum("ti,ij,tj->t", n, matrix, n)
    denominator = 1.0 - np.einsum("ti,i->t", n, direction)
    delay_sender = state.t - length - np.einsum("ti,i->t", p[:, sender, :], direction)
    delay_receiver = state.t - np.einsum("ti,i->t", p[:, receiver, :], direction)
    return 0.5 * nen / denominator * (h(delay_sender) - h(delay_receiver))


def pair_observable_from_oneway(
    y_forward: ArrayLike,
    y_reverse: ArrayLike,
    f0: float,
    arm_length: ArrayLike,
    *,
    fdot: float = 0.0,
    t: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Return the TDI-like pair observable exp(beta) y_ab + y_ba."""

    return np.exp(beta(f0, arm_length, fdot=fdot, t=t)) * np.asarray(y_forward) + np.asarray(y_reverse)


def _quadratic_form_coefficients(n: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Coefficients q where n.T @ E(e) @ n = q @ e."""

    return np.stack(
        [
            n[:, 0] ** 2 - n[:, 2] ** 2,
            2.0 * n[:, 0] * n[:, 1],
            2.0 * n[:, 0] * n[:, 2],
            n[:, 1] ** 2 - n[:, 2] ** 2,
            2.0 * n[:, 1] * n[:, 2],
        ],
        axis=-1,
    ).astype(np.complex128)


def triliear_tensor(
    state: LISAState,
    f0: float,
    *,
    fdot: float = 0.0,
    include_half: bool = True,
) -> NDArray[np.complex128]:
    """Build A[t, channel, e_index, ktilde_index] from Eqs. (11)-(14).

    Channels are ordered as spacecraft pairs (1,2), (1,3), (2,3) using zero-based
    Python indices `(0, 1)`, `(0, 2)`, `(1, 2)`. The affine direction vector is
    `ktilde = (1, kx, ky, kz)`.
    """

    nt = state.t.size
    out = np.zeros((nt, len(TDI_CHANNELS), 5, 4), dtype=np.complex128)
    scale = 0.5 if include_half else 1.0

    for channel, (first, second, complement) in enumerate(TDI_CHANNELS):
        n = state.unit_arm(first, second)
        length = state.arm_length(first, second)
        b = beta(f0, length, fdot=fdot, t=state.t if fdot else None)
        a_pair = 2.0 * b * (1.0 + b + (2.0 / 3.0) * b**2)
        a_comp = -(b**2)[:, None] * state.relative_positions[:, complement, :] / length[:, None] * (1.0 + b)[:, None]
        q = _quadratic_form_coefficients(n)

        out[:, channel, :, 0] = scale * q * a_pair[:, None]
        out[:, channel, :, 1:] = scale * q[:, :, None] * a_comp[:, None, :]

    return out


def triliear_tensors(
    state: LISAState,
    f0: ArrayLike,
    *,
    fdot: ArrayLike | float = 0.0,
    include_half: bool = True,
) -> NDArray[np.complex128]:
    """Build source-batched triliear tensors.

    Returns an array with shape `(source, time, channel, e_index, ktilde_index)`.
    """

    frequencies = np.atleast_1d(np.asarray(f0, dtype=np.float64))
    chirps = np.broadcast_to(np.asarray(fdot, dtype=np.float64), frequencies.shape)
    out = np.zeros((frequencies.size, state.t.size, len(TDI_CHANNELS), 5, 4), dtype=np.complex128)
    scale = 0.5 if include_half else 1.0

    for channel, (first, second, complement) in enumerate(TDI_CHANNELS):
        n = state.unit_arm(first, second)
        length = state.arm_length(first, second)
        freq_eff = frequencies[:, None] + chirps[:, None] * state.t[None, :]
        b = -2j * np.pi * freq_eff * length[None, :]
        a_pair = 2.0 * b * (1.0 + b + (2.0 / 3.0) * b**2)
        a_comp = -(b**2)[:, :, None] * state.relative_positions[None, :, complement, :] / length[None, :, None] * (1.0 + b)[:, :, None]
        q = _quadratic_form_coefficients(n)

        out[:, :, channel, :, 0] = scale * a_pair[:, :, None] * q[None, :, :]
        out[:, :, channel, :, 1:] = scale * q[None, :, :, None] * a_comp[:, :, None, :]

    return out


def triliear_waveform(
    waveform: ArrayLike,
    a_tensor: ArrayLike,
    e: ArrayLike,
    k: ArrayLike,
) -> NDArray[np.complex128]:
    """Evaluate W[t, channel] = f[t] A[t, channel, j, l] e[j] ktilde[l]."""

    f = np.asarray(waveform, dtype=np.complex128)
    a = np.asarray(a_tensor, dtype=np.complex128)
    coeffs = np.asarray(e, dtype=np.complex128)
    direction = normalize_direction(k)
    ktilde = np.concatenate(([1.0], direction)).astype(np.complex128)
    return f[:, None] * np.einsum("tcjl,j,l->tc", a, coeffs, ktilde)


def triliear_waveforms(
    waveforms: ArrayLike,
    a_tensors: ArrayLike,
    e: ArrayLike,
    k: ArrayLike,
) -> NDArray[np.complex128]:
    """Evaluate source-batched triliear waveforms.

    Inputs are shaped as:

    - `waveforms`: `(source, time)`
    - `a_tensors`: `(source, time, channel, 5, 4)`
    - `e`: `(source, 5)`
    - `k`: `(source, 3)`

    The output shape is `(source, time, channel)`.
    """

    f = np.asarray(waveforms, dtype=np.complex128)
    a = np.asarray(a_tensors, dtype=np.complex128)
    coeffs = np.asarray(e, dtype=np.complex128)
    directions = normalize_direction(k)
    ones = np.ones((directions.shape[0], 1), dtype=np.complex128)
    ktilde = np.concatenate((ones, directions.astype(np.complex128)), axis=1)
    return f[:, :, None] * np.einsum("stcjl,sj,sl->stc", a, coeffs, ktilde)


def triliear_source_batch(
    state: LISAState,
    sources: GalacticBinaryParams | list[GalacticBinaryParams],
    *,
    sum_sources: bool = True,
) -> NDArray[np.complex128]:
    """Evaluate a batch of non-eccentric galactic binaries with one vectorized path."""

    source_list = [sources] if isinstance(sources, GalacticBinaryParams) else sources
    f0 = np.array([source.f0 for source in source_list], dtype=np.float64)
    fdot = np.array([source.fdot for source in source_list], dtype=np.float64)
    e = np.stack([source.tracefree_coefficients() for source in source_list], axis=0)
    k = np.stack([source.propagation_direction() for source in source_list], axis=0)
    waveforms = np.stack([source.slow_envelope(state.t) for source in source_list], axis=0)

    result = triliear_waveforms(waveforms, triliear_tensors(state, f0, fdot=fdot), e, k)
    return np.sum(result, axis=0) if sum_sources else result
