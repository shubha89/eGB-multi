"""Small tensor helpers for the paper's trace-free parameterization."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def tracefree_matrix(e: ArrayLike) -> NDArray[np.complex128]:
    """Return the trace-free symmetric matrix encoded by five coefficients.

    The response model uses

        [[e1, e2, e3],
         [e2, e4, e5],
         [e3, e5, -e1 - e4]]

    The function accepts a final dimension of length 5 and preserves leading
    batch dimensions.
    """

    coeffs = np.asarray(e, dtype=np.complex128)
    if coeffs.shape[-1] != 5:
        raise ValueError("expected final dimension of length 5")

    out = np.zeros((*coeffs.shape[:-1], 3, 3), dtype=np.complex128)
    out[..., 0, 0] = coeffs[..., 0]
    out[..., 0, 1] = out[..., 1, 0] = coeffs[..., 1]
    out[..., 0, 2] = out[..., 2, 0] = coeffs[..., 2]
    out[..., 1, 1] = coeffs[..., 3]
    out[..., 1, 2] = out[..., 2, 1] = coeffs[..., 4]
    out[..., 2, 2] = -coeffs[..., 0] - coeffs[..., 3]
    return out


def matrix_to_tracefree5(matrix: ArrayLike) -> NDArray[np.complex128]:
    """Extract the five coefficients from a trace-free symmetric matrix."""

    mat = np.asarray(matrix, dtype=np.complex128)
    if mat.shape[-2:] != (3, 3):
        raise ValueError("expected final dimensions (3, 3)")
    return np.stack(
        [mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2], mat[..., 1, 1], mat[..., 1, 2]],
        axis=-1,
    )


def normalize_direction(k: ArrayLike) -> NDArray[np.float64]:
    """Normalize a real propagation direction vector."""

    direction = np.asarray(k, dtype=np.float64)
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("direction vector must be non-zero")
    return direction / norm
