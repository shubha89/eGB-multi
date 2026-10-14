"""Candidate-independent overlap control for two positive covariance metrics.

If every generalized covariance eigenvalue is in [1-epsilon,1+epsilon],
the same pair of normalized waveforms changes overlap by at most
2*epsilon/(1-epsilon). The bound also controls maxima over an unchanged
template family. It applies to the two specified metrics, not to untested
noise physics or missing cross-frequency covariance.
"""

import numpy as np


def relative_covariance_eigenvalues(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[1:] != (2, 2):
        raise ValueError("Aligned two-channel covariance matrices required")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("Finite covariances required")
    for value in (reference, candidate):
        scale = np.max(np.abs(value), axis=(1, 2))
        if np.any(scale <= 0) or np.any(np.max(abs(value-value.conj().swapaxes(1, 2)), axis=(1, 2)) > 1e-12*scale):
            raise ValueError("Hermitian covariances required")
    root = np.linalg.cholesky(reference)
    np.linalg.cholesky(candidate)
    inverse = np.linalg.inv(root)
    relative = np.einsum("fab,fbc,fdc->fad", inverse, candidate, inverse.conj())
    return np.linalg.eigvalsh(relative)


def normalized_overlap_change_bound(covariance_epsilon):
    """Conservative bound for ANY fixed waveforms and the same search family.

    In the reference-whitened coordinates, the inverse candidate covariance
    has eigenvalues in [1/(1+e),1/(1-e)]. Divide by the irrelevant global
    scale 1/(1-e**2); its eigenvalues then lie in [1-e,1+e]. For unit a,b,
    write c'= (c+a.H.b)/D, where ||H||<=e and 1-e<=D<=1+e.
    Triangle and Cauchy-Schwarz give |c'-c|<=2e/(1-e).
    Taking the supremum over the identical family preserves the same bound.
    """
    epsilon = float(covariance_epsilon)
    if not np.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("A finite relative covariance epsilon in [0,1) is required")
    return min(2., 2.*epsilon/(1.-epsilon))
