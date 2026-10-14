import numpy as np
import pytest

from egb_multi.metric_perturbation import relative_covariance_eigenvalues, normalized_overlap_change_bound


def test_complete_complex_block_metric_overlap_and_family_supremum_bound():
    # Independent full dense inverse calculation, including off-diagonal CSD.
    from scipy.linalg import block_diag
    rng = np.random.default_rng(476)
    for target in (1e-5, .02, .3):
        reference, candidate = [], []
        for _ in range(9):
            q = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))
            c = q@q.conj().T + np.eye(2)
            root = np.linalg.cholesky(c)
            u, _ = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
            relative = u@np.diag([1-target, 1+target])@u.conj().T
            reference.append(c); candidate.append(root@relative@root.conj().T)
        eigenvalues = relative_covariance_eigenvalues(reference, candidate)
        assert np.max(abs(eigenvalues-1)) == pytest.approx(target, abs=3e-14)
        metrics = [np.linalg.inv(block_diag(*rows)) for rows in (reference, candidate)]
        waveforms = rng.normal(size=(100, 18))+1j*rng.normal(size=(100, 18))
        overlaps = []
        for metric in metrics:
            norms = np.einsum('bi,ij,bj->b', waveforms.conj(), metric, waveforms).real
            projections = (waveforms.conj()@metric@waveforms[0]).real
            overlaps.append(projections/np.sqrt(norms*norms[0]))
        bound = normalized_overlap_change_bound(target)
        assert np.max(abs(overlaps[0]-overlaps[1])) <= bound
        assert abs(np.max(overlaps[0][1:])-np.max(overlaps[1][1:])) <= bound


def test_invalid_or_nonpositive_metrics_are_rejected():
    valid = np.eye(2)[None]
    with pytest.raises(ValueError):
        relative_covariance_eigenvalues(valid, np.array([[[1., .1], [0., 1.]]]))
    with pytest.raises(np.linalg.LinAlgError):
        relative_covariance_eigenvalues(valid, -valid)
    for epsilon in (-.01, 1., np.nan):
        with pytest.raises(ValueError):
            normalized_overlap_change_bound(epsilon)
