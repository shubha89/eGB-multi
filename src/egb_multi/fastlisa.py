"""Tensor-based polarization conversion for optional fastlisaresponse use.

This module does not import fastlisaresponse or change its implementation.
See examples/egb_fastlisaresponse.py for its native projection and TDI calls.
"""
import numpy as np


def polarization_basis_matrix(source):
    """Map eGB plus/cross components to fastlisaresponse's ecliptic basis.

    Includes source.psi exactly once. The map is obtained from tensor
    contractions, with no fitted phase, amplitude, time or polarization.
    Both bases use the propagation direction -n_source, angles in radians.
    """
    lam, beta = source.lambda_, source.beta
    u = np.array([np.sin(lam), -np.cos(lam), 0.])
    v = np.array([-np.sin(beta)*np.cos(lam),
                  -np.sin(beta)*np.sin(lam), np.cos(beta)])
    native = np.array([np.outer(u, u)-np.outer(v, v),
                       np.outer(u, v)+np.outer(v, u)])
    plus, cross = source.polarization_basis()
    cosine, sine = np.cos(2*source.psi), np.sin(2*source.psi)
    own = np.array([cosine*plus-sine*cross, sine*plus+cosine*cross])
    matrix = .5 * np.einsum("aij,bij->ab", native, own)
    reconstructed = np.einsum("ab,aij->bij", matrix, native)
    if np.linalg.norm(reconstructed-own) > 1e-12:
        raise ValueError("source and native polarization tensors are inconsistent")
    return matrix


def complex_polarizations(source, values, *, xp=np):
    """Convert real (...,2,N) eGB strain to native h_plus + i*h_cross.

    xp may be NumPy or CuPy; callers explicitly transfer or share GPU input
    before this call. This converts the basis only, with no TDI sign change.
    """
    values = xp.asarray(values)
    if values.ndim < 2 or values.shape[-2] != 2 or xp.iscomplexobj(values):
        raise ValueError("real plus/cross values with shape (...,2,N) required")
    if values.dtype != xp.float64 or not bool(xp.all(xp.isfinite(values))):
        raise ValueError("finite float64 strain required")
    matrix = xp.asarray(polarization_basis_matrix(source), dtype=xp.float64)
    native = xp.einsum("ab,...bn->...an", matrix, values)
    return native[..., 0, :] + 1j*native[..., 1, :]
