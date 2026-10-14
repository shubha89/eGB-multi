"""Real, coherent A/E comparisons with explicit TDI and PSD conventions.

Real samples use dt*rfft, a one-sided density PSD, and 4*df weighting.
DC and the Nyquist endpoint are excluded; no taper or time shift is applied.
An optional complex cross PSD supplies the full Hermitian A/E covariance;
omitting it explicitly selects diagonal weighting. This module does not search
frequency or sky and does not label a fixed-candidate overlap a fitting factor.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TDIConvention:
    generation: int
    measurement_order: int
    delay_order: int
    orbit: str
    delays: str
    unit: str = "fractional_frequency"
    aet: str = "A=(Z-X)/sqrt(2);E=(X-2Y+Z)/sqrt(6)"

    def __post_init__(self):
        if self.generation not in (1, 2):
            raise ValueError("TDI generation must be 1 or 2")
        if any(v < 1 or v % 2 != 1 for v in (self.measurement_order, self.delay_order)):
            raise ValueError("Interpolation orders must be positive odd integers")
        if not self.orbit or not self.delays:
            raise ValueError("Orbit and delay conventions must be recorded")
        if self.unit not in ("fractional_frequency", "frequency_hz"):
            raise ValueError("Unknown response unit")


@dataclass
class AESeries:
    t: np.ndarray
    channels: Mapping[str, np.ndarray]
    convention: TDIConvention

    def __post_init__(self):
        self.t = np.array(self.t, dtype=float, copy=True)
        if self.t.ndim != 1 or self.t.size < 4 or not np.all(np.isfinite(self.t)):
            raise ValueError("At least four finite, one-dimensional timestamps required")
        steps = np.diff(self.t)
        if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-9, atol=1e-10):
            raise ValueError("Timestamps must be increasing and uniformly sampled")
        converted = {}
        for channel in ("A", "E"):
            value = np.asarray(self.channels[channel])
            if np.iscomplexobj(value) or value.shape != self.t.shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{channel} must contain finite real samples aligned with timestamps")
            converted[channel] = np.array(value, dtype=float, copy=True)
        self.channels = converted

    @property
    def dt(self):
        return float(self.t[1] - self.t[0])


@dataclass
class AEPSD:
    frequency_hz: np.ndarray
    channels: Mapping[str, np.ndarray]
    convention: TDIConvention
    provenance: dict
    cross_ae: np.ndarray | None = None

    def __post_init__(self):
        self.frequency_hz = np.array(self.frequency_hz, dtype=float, copy=True)
        f = self.frequency_hz
        if f.ndim != 1 or f.size < 2 or not np.all(np.isfinite(f)) or f[0] < 0 or np.any(np.diff(f) <= 0):
            raise ValueError("PSD frequencies must be finite, nonnegative and strictly increasing")
        if self.provenance.get("sidedness") != "one_sided" or self.provenance.get("scaling") != "density":
            raise ValueError("PSD must explicitly record one-sided density normalization")
        if not self.provenance.get("noise_model"):
            raise ValueError("PSD noise model must be recorded")
        self.channels = {ch: np.array(self.channels[ch], dtype=float, copy=True) for ch in ("A", "E")}
        if any(value.shape != f.shape for value in self.channels.values()):
            raise ValueError("PSD arrays must match the frequency grid")
        if self.cross_ae is not None:
            if self.provenance.get('CSD_convention') != 'S_AE=E[tilde_A*conj(tilde_E)]':
                raise ValueError('Cross PSD must record the S_AE covariance convention')
            self.cross_ae = np.array(self.cross_ae, dtype=complex, copy=True)
            if self.cross_ae.shape != f.shape:
                raise ValueError('Cross PSD must match the frequency grid')

    def as_fractional_frequency(self, central_frequency_hz: float):
        if self.convention.unit != "frequency_hz":
            raise ValueError("Only a frequency-Hz PSD can be converted")
        if not np.isfinite(central_frequency_hz) or central_frequency_hz <= 0:
            raise ValueError("A finite positive optical carrier frequency is required")
        from dataclasses import replace
        provenance = dict(self.provenance, central_frequency_hz=float(central_frequency_hz), conversion="PSD_fractional=PSD_Hz/central_frequency_hz**2")
        cross = None if self.cross_ae is None else self.cross_ae / central_frequency_hz**2
        return AEPSD(self.frequency_hz, {ch: value / central_frequency_hz**2 for ch, value in self.channels.items()}, replace(self.convention, unit="fractional_frequency"), provenance, cross)


class AEInnerProduct:
    """One fixed band, PSD, timestamp grid and injection norm per comparison."""

    def __init__(self, reference: AESeries, psd: AEPSD, band_hz: tuple[float, float]):
        if reference.convention != psd.convention:
            raise ValueError("Signal and PSD conventions differ")
        low, high = band_hz
        nyquist = .5 / reference.dt
        if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high < nyquist):
            raise ValueError("Band must lie strictly above DC and below Nyquist; refine cadence instead of truncating")
        if low < psd.frequency_hz[0] or high > psd.frequency_hz[-1]:
            raise ValueError("PSD does not cover the full requested band; extrapolation is forbidden")
        self.t = reference.t.copy()
        self.dt = reference.dt
        self.convention = reference.convention
        self.band_hz = (float(low), float(high))
        self.df = 1. / (self.t.size * self.dt)
        frequencies = np.fft.rfftfreq(self.t.size, self.dt)
        self.mask = (frequencies >= low) & (frequencies <= high)
        if not np.any(self.mask):
            raise ValueError("No Fourier bins in requested band")
        self.frequency_hz = frequencies[self.mask]
        values = []
        for ch in ("A", "E"):
            # Check both endpoints used by interpolation, including neighboring
            # input PSD bins; invalid values must never be silently masked away.
            left = max(0, np.searchsorted(psd.frequency_hz, low) - 1)
            right = min(psd.frequency_hz.size, np.searchsorted(psd.frequency_hz, high, side="right") + 1)
            input_values = psd.channels[ch][left:right]
            if np.any(~np.isfinite(input_values)) or np.any(input_values <= 0):
                raise ValueError(f"Nonpositive/nonfinite {ch} PSD in analysis band")
            values.append(np.interp(self.frequency_hz, psd.frequency_hz, psd.channels[ch]))
        self.sqrt_weight = np.sqrt(4. * self.df / np.array(values))
        self.covariance_whitener = None
        self.noise_weighting = 'diagonal_AE' if psd.cross_ae is None else 'full_Hermitian_AE'
        if psd.cross_ae is not None:
            cross_input = psd.cross_ae[left:right]
            if np.any(~np.isfinite(cross_input)) or np.any(
                np.abs(cross_input)**2 >= psd.channels['A'][left:right]*psd.channels['E'][left:right]
            ):
                raise ValueError('A/E PSD covariance must be finite and positive definite in analysis band')
            cross = np.interp(self.frequency_hz, psd.frequency_hz, psd.cross_ae)
            # Factor C=D R D with D=diag(sqrt(S_A),sqrt(S_E)). Scaling
            # first avoids a dimensional tiny determinant. C_AE=E[A E*].
            correlation = np.zeros((cross.size, 2, 2), dtype=complex)
            correlation[:, 0, 0] = correlation[:, 1, 1] = 1.
            correlation[:, 0, 1] = cross / np.sqrt(values[0]*values[1])
            correlation[:, 1, 0] = correlation[:, 0, 1].conj()
            try:
                lower = np.linalg.cholesky(correlation)
            except np.linalg.LinAlgError as error:
                raise ValueError('Interpolated A/E PSD covariance is not positive definite') from error
            self.covariance_whitener = np.linalg.solve(lower, np.broadcast_to(np.eye(2), lower.shape))
        self.reference = self.prepare(reference)
        self.reference_norm = self.norm(self.reference)

    def prepare(self, series: AESeries):
        if series.convention != self.convention:
            raise ValueError("Candidate TDI or unit convention differs")
        if not np.array_equal(series.t, self.t):
            raise ValueError("Candidate timestamps differ; implicit time alignment is forbidden")
        spectra = np.array([self.dt * np.fft.rfft(series.channels[ch])[self.mask] for ch in ("A", "E")])
        weighted = spectra * self.sqrt_weight
        if self.covariance_whitener is not None:
            weighted = np.einsum('fij,jf->if', self.covariance_whitener, weighted)
        return weighted

    @staticmethod
    def inner(a, b):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
            raise ValueError("Invalid prepared A/E arrays")
        return float(np.real(np.vdot(b, a)))

    @classmethod
    def norm(cls, a):
        value = cls.inner(a, a)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Noise-weighted norm must be finite and positive")
        return value

    def compare_prepared(self, candidate):
        norm = self.norm(candidate)
        cross = self.inner(self.reference, candidate)
        overlap = cross / np.sqrt(self.reference_norm * norm)
        if not np.isfinite(overlap) or abs(overlap) > 1. + 1e-12:
            raise ValueError("Invalid normalized overlap")
        amplitude = max(0., cross / norm)
        raw = self.reference - candidate
        residual = self.reference - amplitude * candidate
        return {
            "overlap": float(overlap),
            "positive_amplitude_scale": float(amplitude),
            "raw_relative_residual": float(np.sqrt(self.inner(raw, raw) / self.reference_norm)),
            "fitted_relative_residual": float(np.sqrt(self.inner(residual, residual) / self.reference_norm)),
            "reference_norm": self.reference_norm,
            "candidate_norm": norm,
        }

    def compare(self, candidate: AESeries):
        return self.compare_prepared(self.prepare(candidate))
