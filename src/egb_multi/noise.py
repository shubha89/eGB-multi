"""Noise generation, PSD estimation, and overlap utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch

from .tdi import LINK_LABELS

KARNESIS_2021_CGB_PARAMETERS = {
    ("running_median", 5): {
        "a1": -0.15,
        "ak": -0.34,
        "b1": -2.78,
        "bk": -2.55,
        "amplitude": 1.14e-44,
        "f2": 0.00059,
        "alpha": 1.66,
    },
    ("running_median", 7): {
        "a1": -0.15,
        "ak": -0.37,
        "b1": -2.72,
        "bk": -2.49,
        "amplitude": 1.15e-44,
        "f2": 0.00067,
        "alpha": 1.56,
    },
    ("running_mean", 5): {
        "a1": -0.16,
        "ak": -0.34,
        "b1": -2.78,
        "bk": -2.53,
        "amplitude": 1.15e-44,
        "f2": 0.00059,
        "alpha": 1.66,
    },
    ("running_mean", 7): {
        "a1": -0.25,
        "ak": -0.27,
        "b1": -2.70,
        "bk": -2.47,
        "amplitude": 1.14e-44,
        "f2": 0.00031,
        "alpha": 1.80,
    },
}


@dataclass(frozen=True)
class LisainstrumentNoiseResult:
    """Container for LISA instrument noise projected to TDI channels."""

    t: NDArray[np.float64]
    dt: float
    links: dict[str, NDArray[np.float64]]
    xyz: dict[str, NDArray[np.float64]]
    aet: dict[str, NDArray[np.float64]]
    delays: dict[str, NDArray[np.float64]] | None = None
    central_frequency_hz: float | None = None


def karnesis_2021_cgb_confusion_psd(
    frequency: NDArray[np.float64],
    *,
    t_obs_years: float = 4.0,
    snr_threshold: int = 7,
    smoothing: str = "running_median",
) -> NDArray[np.float64]:
    """Empirical unresolved CGB confusion PSD from Karnesis et al. 2021.

    The model is Eq. (6) with the time-dependent frequency parameters of
    Eq. (7), using Table II coefficients. Frequencies are in Hz and the
    observation time is in years.
    """

    key = (smoothing, int(snr_threshold))
    if key not in KARNESIS_2021_CGB_PARAMETERS:
        raise ValueError(f"unsupported Karnesis 2021 parameter choice {key}")
    if t_obs_years <= 0.0:
        raise ValueError("t_obs_years must be positive")
    params = KARNESIS_2021_CGB_PARAMETERS[key]
    f = np.asarray(frequency, dtype=np.float64)
    out = np.zeros_like(f, dtype=np.float64)
    positive = f > 0.0
    f1 = 10.0 ** (params["a1"] * np.log10(t_obs_years) + params["b1"])
    fknee = 10.0 ** (params["ak"] * np.log10(t_obs_years) + params["bk"])
    fp = f[positive]
    out[positive] = (
        0.5
        * params["amplitude"]
        * fp ** (-7.0 / 3.0)
        * np.exp(-((fp / f1) ** params["alpha"]))
        * (1.0 + np.tanh((fknee - fp) / params["f2"]))
    )
    return out


def pytdi_data_from_lisainstrument_result(
    result,
    *,
    measurement: str = "sci_carrier_fluctuations",
    include_local_metrology: bool = True,
):
    """Build a pyTDI Data object from a `lisainstrument` in-memory result.

    By default this uses `sci_carrier_fluctuations`, i.e. the fluctuating part
    of the science beatnote frequency measurements, and `mprs` as link delays.
    TMI/REF carrier fluctuations are included by default: they are required
    by pyTDI's xi/eta construction for six-laser instrument data. Explicitly
    setting include_local_metrology=False selects the historical science-only
    projection, suitable for GW-only streams rather than instrument noise.
    """

    try:
        from pytdi import Data
    except ImportError as exc:
        raise ImportError("pytdi is required for pytdi_data_from_lisainstrument_result") from exc

    streams = getattr(result.mosa_data, measurement)
    delays_in = result.mosa_data.mprs
    t = np.asarray(result.t, dtype=np.float64)
    if t.size < 2:
        raise ValueError("at least two samples are required")
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1.0e-9, atol=1.0e-12):
        raise ValueError("lisainstrument result must be uniformly sampled")

    zero = np.zeros(t.size, dtype=np.float64)
    measurements = {key: zero.copy() for key in Data.MEASUREMENTS}
    delays = {}
    for label in LINK_LABELS:
        measurements[f"sci_{label}"] = np.asarray(streams[label], dtype=np.float64)
        delays[f"d_{label}"] = np.asarray(delays_in[label], dtype=np.float64)
        if include_local_metrology:
            if measurement != "sci_carrier_fluctuations":
                raise ValueError("Local metrology mapping requires sci_carrier_fluctuations")
            measurements[f"tmi_{label}"] = np.asarray(result.mosa_data.tmi_carrier_fluctuations[label], dtype=np.float64)
            measurements[f"ref_{label}"] = np.asarray(result.mosa_data.ref_carrier_fluctuations[label], dtype=np.float64)

    data = Data(measurements, delays, float(1.0 / dt[0]))
    data.compute_delay_derivatives()
    return data


def xyz_from_pytdi_data(data, *, generation: int = 1, order: int = 45, delay_order: int = 5) -> dict[str, NDArray[np.float64]]:
    """Compute pyTDI Michelson X/Y/Z from a pyTDI Data object."""

    try:
        from pytdi.michelson import compute_factorized_michelson
    except ImportError as exc:
        raise ImportError("pytdi is required for xyz_from_pytdi_data") from exc

    return {
        "X": compute_factorized_michelson(data, rot=0, order=order, delay_order=delay_order, generation=generation, unit="frequency"),
        "Y": compute_factorized_michelson(data, rot=1, order=order, delay_order=delay_order, generation=generation, unit="frequency"),
        "Z": compute_factorized_michelson(data, rot=2, order=order, delay_order=delay_order, generation=generation, unit="frequency"),
    }


def aet_from_xyz(xyz: dict[str, NDArray]) -> dict[str, NDArray]:
    """Convert Michelson X/Y/Z to the common orthogonal A/E/T basis."""

    x = np.asarray(xyz["X"])
    y = np.asarray(xyz["Y"])
    z = np.asarray(xyz["Z"])
    return {
        "A": (z - x) / np.sqrt(2.0),
        "E": (x - 2.0 * y + z) / np.sqrt(6.0),
        "T": (x + y + z) / np.sqrt(3.0),
    }


def simulate_lisainstrument_tdi_noise(
    *,
    size: int,
    dt: float,
    seed: int | None = 1234,
    measurement: str = "sci_carrier_fluctuations",
    generation: int = 1,
    order: int = 45,
    delay_order: int = 5,
    instrument_kwargs: dict | None = None,
    include_local_metrology: bool = True,
    enabled_noises: tuple[str, ...] | None = None,
) -> LisainstrumentNoiseResult:
    """Generate instrument noise and project its SCI/TMI/REF carriers to TDI.

    High interpolation orders are essential for cancelling large laser noise.
    Cadence and filtering still require validation over the intended band.
    This helper uses a synthetic band-limited simulation without antialias
    filtering by default; it does not implement sideband clock calibration.
    enabled_noises selects LISA Instrument's named contributions; None retains
    all defaults. An empty tuple disables all noise and clock imperfections.
    """

    try:
        from lisainstrument import Instrument, SchedulerConfigSerial
    except ImportError as exc:
        raise ImportError("lisainstrument is required for simulate_lisainstrument_tdi_noise") from exc

    kwargs = {
        "size": size,
        "dt": dt,
        "seed": seed,
        "physics_upsampling": 1,
        "aafilter": None,
        "telemetry_downsampling": max(1, size),
        "orbits": "static",
        "interpolation": ("lagrange", 45),
        "noises_f_min_hz": None,
    }
    if instrument_kwargs:
        kwargs.update(instrument_kwargs)

    instrument = Instrument(**kwargs)
    if enabled_noises is not None:
        instrument.disable_all_noises(excluding=list(enabled_noises))
    result = instrument.export_numpy_core(cfgscheduler=SchedulerConfigSerial(chunk_size=size))
    data = pytdi_data_from_lisainstrument_result(result, measurement=measurement, include_local_metrology=include_local_metrology)
    xyz = xyz_from_pytdi_data(data, generation=generation, order=order, delay_order=delay_order)
    aet = aet_from_xyz(xyz)
    links = {label: np.asarray(getattr(result.mosa_data, measurement)[label], dtype=np.float64) for label in LINK_LABELS}
    t = np.asarray(result.t, dtype=np.float64)
    delays = {label: np.asarray(result.mosa_data.mprs[label], dtype=np.float64) for label in LINK_LABELS}
    return LisainstrumentNoiseResult(t=t, dt=float(np.diff(t)[0]), links=links, xyz=xyz, aet=aet,
                                    delays=delays, central_frequency_hz=float(instrument.central_freq))


def estimate_welch_psd(
    series: NDArray,
    dt: float,
    *,
    nperseg: int,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate a one-sided PSD with Welch's method."""

    frequencies, psd = welch(
        np.asarray(series),
        fs=1.0 / dt,
        window=window,
        nperseg=nperseg,
        noverlap=nperseg // 2 if noverlap is None else noverlap,
        detrend=detrend,
        return_onesided=True,
        scaling="density",
    )
    return np.asarray(frequencies, dtype=np.float64), np.asarray(psd, dtype=np.float64)


def estimate_channel_psds(
    channels: dict[str, NDArray],
    dt: float,
    *,
    nperseg: int,
    noverlap: int | None = None,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Estimate Welch PSDs for a channel dictionary."""

    frequencies = None
    psds = {}
    for name, series in channels.items():
        freq, psd = estimate_welch_psd(series, dt, nperseg=nperseg, noverlap=noverlap)
        if frequencies is None:
            frequencies = freq
        elif not np.array_equal(freq, frequencies):
            raise ValueError("Welch returned inconsistent frequency grids")
        psds[name] = psd
    assert frequencies is not None
    return frequencies, psds


def rfft_with_dt(series: NDArray, dt: float) -> NDArray[np.complex128]:
    """Return a one-sided Fourier transform with continuous-transform scaling."""

    return np.fft.rfft(np.asarray(series), axis=-1) * dt


def noise_weighted_inner_product_fd(
    a_fd: NDArray,
    b_fd: NDArray,
    psd: NDArray[np.float64],
    df: float,
    *,
    frequency_mask: NDArray[np.bool_] | None = None,
) -> float:
    """Compute the one-sided noise-weighted inner product from FD arrays."""

    a = np.asarray(a_fd, dtype=np.complex128)
    b = np.asarray(b_fd, dtype=np.complex128)
    s = np.asarray(psd, dtype=np.float64)
    if a.shape != b.shape or a.shape != s.shape:
        raise ValueError("a_fd, b_fd, and psd must have the same shape")
    mask = np.isfinite(s) & (s > 0.0)
    if frequency_mask is not None:
        mask &= np.asarray(frequency_mask, dtype=bool)
    if mask.size:
        mask[0] = False
    return float(4.0 * np.real(np.sum(a[mask] * np.conjugate(b[mask]) / s[mask])) * df)


def normalized_overlap_fd(
    a_fd: NDArray,
    b_fd: NDArray,
    psd: NDArray[np.float64],
    df: float,
    *,
    frequency_mask: NDArray[np.bool_] | None = None,
) -> float:
    """Return the normalized noise-weighted overlap."""

    aa = noise_weighted_inner_product_fd(a_fd, a_fd, psd, df, frequency_mask=frequency_mask)
    bb = noise_weighted_inner_product_fd(b_fd, b_fd, psd, df, frequency_mask=frequency_mask)
    if aa <= 0.0 or bb <= 0.0:
        return 0.0
    ab = noise_weighted_inner_product_fd(a_fd, b_fd, psd, df, frequency_mask=frequency_mask)
    return float(ab / np.sqrt(aa * bb))


def fitting_factor_fixed_grid(
    reference_fd: NDArray,
    template_fd: NDArray,
    psd: NDArray[np.float64],
    df: float,
    *,
    frequency_mask: NDArray[np.bool_] | None = None,
) -> float:
    """Current fixed-parameter fitting factor, i.e. absolute normalized overlap."""

    return abs(normalized_overlap_fd(reference_fd, template_fd, psd, df, frequency_mask=frequency_mask))
