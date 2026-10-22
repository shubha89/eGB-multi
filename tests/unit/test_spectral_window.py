import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.signal import get_window

from egb_multi.spectral_window import expected_windowed_covariance


@pytest.mark.parametrize("length", [15, 16])
@pytest.mark.parametrize("detrend", [False, "constant"])
def test_window_expectation_matches_independent_dense_covariance(length, detrend):
    fs, grid = 0.2, 256
    f = np.fft.rfftfreq(grid, 1 / fs)
    # A real lag-dependent cross covariance tests both orientation and sign.
    two = np.zeros((len(f), 2, 2), complex)
    two[:, 0, 0] = 2 + np.cos(2 * np.pi * f / fs)
    two[:, 1, 1] = 3 + 0.4 * np.cos(4 * np.pi * f / fs)
    two[:, 0, 1] = 0.3 * np.exp(-2j * np.pi * f / fs)
    two[:, 1, 0] = two[:, 0, 1].conj()
    one = two.copy()
    one[1:-1] *= 2
    frequencies, actual = expected_windowed_covariance(one, fs, length, detrend=detrend)
    full = np.concatenate((two, two[-2:0:-1].conj()))
    lag = fs * np.fft.ifft(full, axis=0).real
    w = get_window("hann", length)
    transform = np.exp(-2j * np.pi * frequencies[:, None] * np.arange(length)[None, :] / fs)
    projection = np.eye(length) if detrend is False else np.eye(length) - np.ones((length, length)) / length
    operator = (transform * w) @ projection
    expected = np.zeros_like(actual)
    for a in range(2):
        for b in range(2):
            covariance = toeplitz(lag[:length, a, b], np.r_[lag[0, a, b], lag[-1:-length:-1, a, b]])
            expected[:, a, b] = np.einsum("ki,ij,kj->k", operator, covariance, operator.conj()) / (fs * np.sum(w**2))
    expected[1:-1 if length % 2 == 0 else None] *= 2
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)


def test_white_noise_hann_mean_removal_has_known_first_bin_bias():
    # For periodic Hann: mean removal subtracts 2/3 of the DC power and
    # 1/6 of the first positive bin's power; higher bins are unaffected.
    spectrum = np.broadcast_to(np.eye(2), (129, 2, 2)).copy()
    spectrum[[0, -1]] *= 0.5
    _, result = expected_windowed_covariance(spectrum, 1, 32)
    np.testing.assert_allclose(result[0], np.eye(2) / 6, atol=1e-14)
    np.testing.assert_allclose(result[1], np.eye(2) * 5 / 6, atol=1e-14)
    np.testing.assert_allclose(result[2:-1], np.broadcast_to(np.eye(2), result[2:-1].shape), atol=1e-14)


def test_window_model_rejects_insufficient_resolution_and_bad_covariance():
    spectrum = np.broadcast_to(np.eye(2, dtype=complex), (33, 2, 2)).copy()
    with pytest.raises(ValueError, match="twice"):
        expected_windowed_covariance(spectrum, 1, 64)
    with pytest.raises(ValueError, match="detrend"):
        expected_windowed_covariance(spectrum, 1, 16, detrend="linear")
    spectrum[4, 0, 1] = 0.2j
    with pytest.raises(ValueError, match="Hermitian"):
        expected_windowed_covariance(spectrum, 1, 16)
