"""Finite-segment expectations for a stationary real A/E noise process.

This is a validation calculation, not a new production weighting choice.
For segment length L, detrend P and window w, the expected Fourier covariance
is diag(F W P R P^T W F^H)/(fs sum(w**2)), where R is the stationary
Toeplitz covariance implied by a two-sided spectrum. One-sided doubling is
applied after this calculation, excluding DC and Nyquist. The lag sums below
evaluate this expression without allocating an L by L matrix.

The supplied positive-frequency grid must cover [0, fs/2] uniformly. Its
inverse transform provides lag covariance; refining that grid controls the
periodic approximation to the continuous spectral integral. It must cover
at least twice the segment length. Spectral resolution, moving instruments,
filter transients and frequency-bin covariance are separate requirements.
"""

import numpy as np
from scipy.signal import correlate, get_window


def expected_windowed_covariance(
    spectrum_onesided, fs_hz, segment_samples, *, window="hann", detrend="constant"
):
    """Expected Welch covariance, including S_AE = E[A(f) conj(E(f))].

    Input shape is (M/2+1, channels, channels), with even M and frequencies k*fs/M.
    Density units are unchanged. Only real processes, real windows and
    detrend=False or removal of the segment mean are supported. Overlap and
    realization count affect estimator variance, not this expectation.
    """
    supplied = np.asarray(spectrum_onesided, dtype=complex)
    length = int(segment_samples)
    if length != segment_samples or length < 4 or not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("Positive sample rate and integer segment length >= 4 required")
    if supplied.ndim != 3 or supplied.shape[1] < 1 or supplied.shape[1] != supplied.shape[2] or not np.all(np.isfinite(supplied)):
        raise ValueError("Finite square positive-frequency covariance required")
    channels = supplied.shape[1]
    grid_samples = 2 * (len(supplied) - 1)
    if grid_samples < 2 * length:
        raise ValueError("Spectral integral grid must cover at least twice the segment length")
    scale = float(np.max(abs(supplied)))
    if scale <= 0:
        raise ValueError("Nonzero noise covariance required")
    if np.max(abs(supplied - supplied.swapaxes(1, 2).conj())) > 1e-12 * scale:
        raise ValueError("Covariance must be Hermitian")
    if np.min(np.linalg.eigvalsh(supplied)) < -1e-12 * scale:
        raise ValueError("Covariance must be positive semidefinite")
    if np.max(abs(supplied[[0, -1]].imag)) > 1e-12 * scale:
        raise ValueError("DC and Nyquist covariance must be real for sampled real noise")
    if detrend is not False and detrend != "constant":
        raise ValueError("Only detrend=False or detrend='constant' is supported")
    weights = get_window(window, length, fftbins=True) if isinstance(window, (str, tuple)) else np.asarray(window, dtype=float)
    if weights.shape != (length,) or not np.all(np.isfinite(weights)) or np.sum(weights**2) <= 0:
        raise ValueError("Finite nonzero real segment window required")

    # Convert the scipy-style one-sided density to the full two-sided density.
    positive = supplied.copy()
    positive[1:-1] *= 0.5
    full = np.concatenate((positive, positive[-2:0:-1].conj()), axis=0)
    lag = fs_hz * np.fft.ifft(full, axis=0)
    # A real multichannel process has real (not necessarily symmetric) lags.
    if np.max(abs(lag.imag)) > 1e-11 * fs_hz * scale:
        raise ValueError("Input does not define real stationary lag covariance")
    lag = lag.real
    window_lag = correlate(weights, weights, mode="full", method="fft")[length - 1:]
    folded = np.zeros((length, channels, channels), dtype=complex)
    folded[:] = lag[:length] * window_lag[:, None, None]
    folded[1:] += lag[-(length - 1):] * window_lag[:0:-1, None, None]
    numerator = np.fft.fft(folded, axis=0)

    if detrend == "constant":
        zero = np.zeros((1, channels, channels))
        positive_sum = np.concatenate((zero, np.cumsum(lag[1:length], axis=0)))
        negative_sum = np.concatenate((zero, np.cumsum(lag[-1:-length:-1], axis=0)))
        rows = lag[0] + positive_sum + negative_sum[::-1]
        columns = rows[::-1]
        weighted_rows = np.fft.fft(weights[:, None, None] * rows, axis=0)
        weighted_columns = np.fft.fft(weights[:, None, None] * columns, axis=0).conj()
        transformed_window = np.fft.fft(weights)
        numerator -= transformed_window.conj()[:, None, None] * weighted_rows / length
        numerator -= transformed_window[:, None, None] * weighted_columns / length
        numerator += abs(transformed_window[:, None, None])**2 * rows.sum(axis=0) / length**2

    result = numerator[:length // 2 + 1] / (fs_hz * np.sum(weights**2))
    if length % 2 == 0:
        result[1:-1] *= 2
    else:
        result[1:] *= 2
    # Remove only floating-point anti-Hermitian roundoff, never a PSD floor.
    result = (result + result.swapaxes(1, 2).conj()) / 2
    return np.fft.rfftfreq(length, 1 / fs_hz), result
