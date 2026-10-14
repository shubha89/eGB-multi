"""Windowed Fast Galactic Binary response prototype.

This module follows the algorithmic structure of Riegger, Robertsson, and
Andersson, Phys. Rev. D 110, 082001 (2024).  It is a CPU reference
implementation intended for testing and benchmarking, not a GPU production
kernel.

Implemented pieces:

* Eq. (3): one-way Doppler measurements for a monochromatic/chirping GB.
* Eq. (14): carrier-stripped slow link streams.
* Eq. (15): slow-bandwidth estimate.
* Eq. (21)--(22): base-frequency assignment on a Fourier grid.
* Eq. (27)--(30): slow sample count and grouping by base frequency.
* Eq. (25) and Eq. (31): symmetric buffer-zone windowing.
* Eq. (12) idea: time-varying delay phase corrections.
* Eq. (9)/(13) structure: first-generation unequal-arm Michelson spectra.
* Eq. (17)--(19): frequency shifting/reduction and inverse FFT.

Approximations relative to a full paper-grade implementation:

* The implementation evaluates Doppler links directly via Eq. (3), rather than
  using the optical-path-length version and Fourier differentiation.
* The base signal is monochromatic; chirping remains in the slow stream.
* The time-varying-delay TDI is implemented through the paper's phase
  correction idea and first-generation Michelson structure, but is not a
  substitute for a full second-generation TDI pipeline.
* The code prioritizes clarity and validation over GPU-oriented memory layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, log2

import numpy as np
from numpy.typing import NDArray

from .constants import JULIAN_YEAR_S
from .external import default_lisaorbits, state_from_lisaorbits
from .geometry import LISAState
from .sources import GalacticBinaryParams


DIRECTED_LINKS: tuple[str, ...] = ("12", "13", "21", "23", "31", "32")
CHANNEL_VERTICES: dict[str, int] = {"X": 1, "Y": 2, "Z": 3}


@dataclass(frozen=True)
class WFGBConfig:
    """Configuration for the windowed FGB reference implementation."""

    duration: float
    dt: float
    buffer_duration: float | None = None
    window_a: float = 0.36
    window_power: int = 4
    oversampling_power: int = 3
    min_slow_samples: int = 1024
    db: int | None = 1
    use_time_varying_delays: bool = True
    nominal_delay: float | None = None


@dataclass(frozen=True)
class WFGBGroup:
    """Metadata for one base-frequency group."""

    base_frequency: float
    source_indices: tuple[int, ...]
    slow_samples: int


@dataclass(frozen=True)
class WFGBResult:
    """Windowed FGB time-domain channels and diagnostic metadata."""

    times: NDArray[np.float64]
    x: NDArray[np.complex128]
    y: NDArray[np.complex128]
    z: NDArray[np.complex128]
    groups: tuple[WFGBGroup, ...]
    db: int
    window_times: NDArray[np.float64]
    x_windowed: NDArray[np.complex128]
    y_windowed: NDArray[np.complex128]
    z_windowed: NDArray[np.complex128]

    @property
    def xyz(self) -> dict[str, NDArray[np.complex128]]:
        return {"X": self.x, "Y": self.y, "Z": self.z}


def source_bandwidth_hz(source: GalacticBinaryParams, duration: float) -> float:
    """Eq. (15): approximate bandwidth of the carrier-stripped slow stream."""

    return float(2.0 * (1.0e-4 * source.f0 + 4.0 / JULIAN_YEAR_S) + abs(source.fdot) * duration)


def slow_sample_count(
    source: GalacticBinaryParams,
    *,
    duration: float,
    window_duration: float,
    oversampling_power: int = 3,
    min_slow_samples: int = 1024,
) -> int:
    """Eq. (27): power-of-two sample count for one source's slow stream."""

    bandwidth = source_bandwidth_hz(source, duration)
    raw = max(bandwidth * window_duration, 1.0)
    exponent = max(0, ceil(log2(raw)) + oversampling_power)
    return max(min_slow_samples, 2**exponent)


def choose_db(
    sources: list[GalacticBinaryParams],
    *,
    duration: float,
    window_duration: float,
    slow_samples: NDArray[np.int64],
) -> int:
    """Eq. (28): choose an integer base-frequency spacing parameter."""

    bounds = []
    for source, samples in zip(sources, slow_samples, strict=True):
        bandwidth = source_bandwidth_hz(source, duration)
        bound = 0.5 * duration * (samples / window_duration - bandwidth)
        bounds.append(bound)
    return max(1, int(floor(min(bounds))))


def base_frequency_hz(source: GalacticBinaryParams, *, duration: float, db: int) -> float:
    """Eq. (21)--(22): assign source to the next smaller base frequency."""

    delta_base = db / duration
    return floor(source.f0 / delta_base) * delta_base


def wfgb_window(
    times: NDArray[np.float64],
    *,
    duration: float,
    buffer_duration: float,
    a: float = 0.36,
    power: int = 4,
) -> NDArray[np.float64]:
    """Eq. (25): buffer-zone taper.

    The OCR text in the PDF renders the Tukey subfunction ambiguously.  The
    implementation below uses the standard cosine taper that satisfies the
    boundary behavior required by Eq. (25): zero at the outer buffer edges and
    one at the plateau boundaries.
    """

    if not (0.0 < a <= 1.0):
        raise ValueError("window parameter a must satisfy 0 < a <= 1")
    if buffer_duration <= 0.0:
        raise ValueError("buffer_duration must be positive")
    taper_width = a * buffer_duration
    left_plateau = -buffer_duration * (1.0 - a)
    right_plateau = duration + buffer_duration * (1.0 - a)

    window = np.zeros_like(times, dtype=np.float64)
    plateau = (times >= left_plateau) & (times < right_plateau)
    window[plateau] = 1.0

    left = (times >= -buffer_duration) & (times < left_plateau)
    left_x = (times[left] + buffer_duration) / taper_width
    window[left] = 0.5 * (1.0 - np.cos(np.pi * left_x))

    right = (times >= right_plateau) & (times < duration + buffer_duration)
    right_x = (duration + buffer_duration - times[right]) / taper_width
    window[right] = 0.5 * (1.0 - np.cos(np.pi * right_x))

    if power != 1:
        window = window**power
    return window


def _carrier(source: GalacticBinaryParams, t: NDArray[np.float64]) -> NDArray[np.complex128]:
    phase = 2.0 * np.pi * source.f0 * t + np.pi * source.fdot * t * t - source.phi0
    return np.exp(1.0j * phase)


def _link_indices(label: str) -> tuple[int, int]:
    if len(label) != 2:
        raise ValueError(f"invalid link label {label!r}")
    return int(label[0]) - 1, int(label[1]) - 1


def _link_lengths(state: LISAState) -> dict[str, NDArray[np.float64]]:
    return {label: state.arm_length(*_link_indices(label), relative=False) for label in DIRECTED_LINKS}


def one_way_doppler_links(
    source: GalacticBinaryParams,
    state: LISAState,
) -> dict[str, NDArray[np.complex128]]:
    """Eq. (3): six one-way Doppler link streams for one source."""

    k = source.propagation_direction()
    e_matrix = source.polarization_matrix()
    links: dict[str, NDArray[np.complex128]] = {}
    for label in DIRECTED_LINKS:
        sender, receiver = _link_indices(label)
        p_sender = state.positions[:, sender, :]
        p_receiver = state.positions[:, receiver, :]
        arm = p_receiver - p_sender
        length = np.linalg.norm(arm, axis=1)
        n_hat = arm / length[:, None]
        antenna = np.einsum("ti,ij,tj->t", n_hat, e_matrix, n_hat)
        denom = 2.0 * (1.0 - np.einsum("ti,i->t", n_hat, k))
        tx_arg = state.t - length - np.einsum("ti,i->t", p_sender, k)
        rx_arg = state.t - np.einsum("ti,i->t", p_receiver, k)
        links[label] = antenna / denom * (_carrier(source, tx_arg) - _carrier(source, rx_arg))
    return links


def slow_doppler_links(
    source: GalacticBinaryParams,
    state: LISAState,
    *,
    base_frequency: float,
) -> dict[str, NDArray[np.complex128]]:
    """Eq. (14): strip the base carrier from one-way Doppler links."""

    base_conjugate = np.exp(-2.0j * np.pi * base_frequency * state.t)
    return {label: base_conjugate * value for label, value in one_way_doppler_links(source, state).items()}


def _delay_fft(
    stream: NDArray[np.complex128],
    *,
    delay_labels: tuple[str, ...],
    base_frequency: float,
    slow_frequencies: NDArray[np.float64],
    lengths: dict[str, NDArray[np.float64]],
    nominal_lengths: dict[str, float],
    window: NDArray[np.float64],
    use_time_varying_delays: bool,
) -> NDArray[np.complex128]:
    """Apply Eq. (12)-style variable-delay phases and constant-delay FFT phases."""

    total_nominal_delay = sum(nominal_lengths[label] for label in delay_labels)
    if use_time_varying_delays and delay_labels:
        delta_delay = np.zeros_like(stream.real)
        for label in delay_labels:
            delta_delay += lengths[label] - nominal_lengths[label]
        variable_phase = np.exp(-2.0j * np.pi * base_frequency * delta_delay)
    else:
        variable_phase = 1.0

    spectrum = np.fft.fft(window * variable_phase * stream)
    full_frequency = base_frequency + slow_frequencies
    return spectrum * np.exp(-2.0j * np.pi * full_frequency * total_nominal_delay)


def _michelson_spectrum(
    grouped_links: dict[str, NDArray[np.complex128]],
    *,
    vertex: int,
    base_frequency: float,
    slow_frequencies: NDArray[np.float64],
    lengths: dict[str, NDArray[np.float64]],
    nominal_lengths: dict[str, float],
    window: NDArray[np.float64],
    use_time_varying_delays: bool,
) -> NDArray[np.complex128]:
    """Eq. (9)/(13) structure for one first-generation Michelson channel."""

    others = [spacecraft for spacecraft in (1, 2, 3) if spacecraft != vertex]
    a, b = others
    va, av = f"{vertex}{a}", f"{a}{vertex}"
    vb, bv = f"{vertex}{b}", f"{b}{vertex}"

    arm_a = _delay_fft(
        grouped_links[va],
        delay_labels=(va,),
        base_frequency=base_frequency,
        slow_frequencies=slow_frequencies,
        lengths=lengths,
        nominal_lengths=nominal_lengths,
        window=window,
        use_time_varying_delays=use_time_varying_delays,
    ) + _delay_fft(
        grouped_links[av],
        delay_labels=(),
        base_frequency=base_frequency,
        slow_frequencies=slow_frequencies,
        lengths=lengths,
        nominal_lengths=nominal_lengths,
        window=window,
        use_time_varying_delays=use_time_varying_delays,
    )
    arm_a_roundtrip = _delay_fft(
        grouped_links[va],
        delay_labels=(va, vb, bv),
        base_frequency=base_frequency,
        slow_frequencies=slow_frequencies,
        lengths=lengths,
        nominal_lengths=nominal_lengths,
        window=window,
        use_time_varying_delays=use_time_varying_delays,
    ) + _delay_fft(
        grouped_links[av],
        delay_labels=(vb, bv),
        base_frequency=base_frequency,
        slow_frequencies=slow_frequencies,
        lengths=lengths,
        nominal_lengths=nominal_lengths,
        window=window,
        use_time_varying_delays=use_time_varying_delays,
    )

    arm_b = _delay_fft(
        grouped_links[vb],
        delay_labels=(vb,),
        base_frequency=base_frequency,
        slow_frequencies=slow_frequencies,
        lengths=lengths,
        nominal_lengths=nominal_lengths,
        window=window,
        use_time_varying_delays=use_time_varying_delays,
    ) + _delay_fft(
        grouped_links[bv],
        delay_labels=(),
        base_frequency=base_frequency,
        slow_frequencies=slow_frequencies,
        lengths=lengths,
        nominal_lengths=nominal_lengths,
        window=window,
        use_time_varying_delays=use_time_varying_delays,
    )
    arm_b_roundtrip = _delay_fft(
        grouped_links[vb],
        delay_labels=(vb, va, av),
        base_frequency=base_frequency,
        slow_frequencies=slow_frequencies,
        lengths=lengths,
        nominal_lengths=nominal_lengths,
        window=window,
        use_time_varying_delays=use_time_varying_delays,
    ) + _delay_fft(
        grouped_links[bv],
        delay_labels=(va, av),
        base_frequency=base_frequency,
        slow_frequencies=slow_frequencies,
        lengths=lengths,
        nominal_lengths=nominal_lengths,
        window=window,
        use_time_varying_delays=use_time_varying_delays,
    )

    return (arm_a_roundtrip - arm_a) - (arm_b_roundtrip - arm_b)


def _nominal_lengths(lengths: dict[str, NDArray[np.float64]], nominal_delay: float | None) -> dict[str, float]:
    if nominal_delay is not None:
        return {label: float(nominal_delay) for label in DIRECTED_LINKS}
    return {label: float(np.mean(value)) for label, value in lengths.items()}


def _add_group_spectrum(
    target: NDArray[np.complex128],
    spectrum: NDArray[np.complex128],
    *,
    base_frequency: float,
    slow_dt: float,
    window_duration: float,
) -> None:
    slow_frequencies = np.fft.fftfreq(spectrum.size, slow_dt)
    df = 1.0 / window_duration
    target_indices = np.rint((base_frequency + slow_frequencies) / df).astype(np.int64) % target.size
    np.add.at(target, target_indices, spectrum * (target.size / spectrum.size))


def synthesize_wfgb(
    sources: GalacticBinaryParams | list[GalacticBinaryParams],
    config: WFGBConfig,
    *,
    orbits=None,
) -> WFGBResult:
    """Algorithm 2: synthesize windowed FGB Michelson X/Y/Z channels."""

    source_list = [sources] if isinstance(sources, GalacticBinaryParams) else list(sources)
    if not source_list:
        raise ValueError("at least one source is required")
    if config.dt <= 0.0 or config.duration <= 0.0:
        raise ValueError("duration and dt must be positive")

    buffer_duration = 2.0 * config.duration if config.buffer_duration is None else config.buffer_duration
    n_obs = int(floor(config.duration / config.dt))
    n_buffer = int(round(buffer_duration / config.dt))
    duration = n_obs * config.dt
    buffer_duration = n_buffer * config.dt
    n_window = n_obs + 2 * n_buffer
    window_duration = n_window * config.dt
    window_times = np.arange(n_window, dtype=np.float64) * config.dt - buffer_duration

    per_source_samples = np.array(
        [
            slow_sample_count(
                source,
                duration=duration,
                window_duration=window_duration,
                oversampling_power=config.oversampling_power,
                min_slow_samples=config.min_slow_samples,
            )
            for source in source_list
        ],
        dtype=np.int64,
    )
    db = choose_db(source_list, duration=duration, window_duration=window_duration, slow_samples=per_source_samples) if config.db is None else config.db
    if db <= 0:
        raise ValueError("db must be positive")

    base_frequencies = np.array([base_frequency_hz(source, duration=duration, db=db) for source in source_list])
    groups: list[WFGBGroup] = []
    x_spectrum = np.zeros(n_window, dtype=np.complex128)
    y_spectrum = np.zeros(n_window, dtype=np.complex128)
    z_spectrum = np.zeros(n_window, dtype=np.complex128)

    if orbits is None:
        orbits = default_lisaorbits("equal")

    for base_frequency in np.unique(base_frequencies):
        indices_array = np.flatnonzero(base_frequencies == base_frequency)
        indices = tuple(int(index) for index in indices_array)
        slow_samples = int(np.max(per_source_samples[indices_array]))
        slow_dt = window_duration / slow_samples
        slow_times = np.arange(slow_samples, dtype=np.float64) * slow_dt - buffer_duration
        state = state_from_lisaorbits(orbits, slow_times)
        lengths = _link_lengths(state)
        nominal_lengths = _nominal_lengths(lengths, config.nominal_delay)
        window = wfgb_window(
            slow_times,
            duration=duration,
            buffer_duration=buffer_duration,
            a=config.window_a,
            power=config.window_power,
        )

        grouped_links = {label: np.zeros(slow_samples, dtype=np.complex128) for label in DIRECTED_LINKS}
        for index in indices:
            slow_links = slow_doppler_links(source_list[index], state, base_frequency=float(base_frequency))
            for label in DIRECTED_LINKS:
                grouped_links[label] += slow_links[label]

        slow_frequencies = np.fft.fftfreq(slow_samples, slow_dt)
        spectra = {
            channel: _michelson_spectrum(
                grouped_links,
                vertex=vertex,
                base_frequency=float(base_frequency),
                slow_frequencies=slow_frequencies,
                lengths=lengths,
                nominal_lengths=nominal_lengths,
                window=window,
                use_time_varying_delays=config.use_time_varying_delays,
            )
            for channel, vertex in CHANNEL_VERTICES.items()
        }
        _add_group_spectrum(
            x_spectrum,
            spectra["X"],
            base_frequency=float(base_frequency),
            slow_dt=slow_dt,
            window_duration=window_duration,
        )
        _add_group_spectrum(
            y_spectrum,
            spectra["Y"],
            base_frequency=float(base_frequency),
            slow_dt=slow_dt,
            window_duration=window_duration,
        )
        _add_group_spectrum(
            z_spectrum,
            spectra["Z"],
            base_frequency=float(base_frequency),
            slow_dt=slow_dt,
            window_duration=window_duration,
        )
        groups.append(WFGBGroup(float(base_frequency), indices, slow_samples))

    x_windowed = np.fft.ifft(x_spectrum)
    y_windowed = np.fft.ifft(y_spectrum)
    z_windowed = np.fft.ifft(z_spectrum)
    start = n_buffer
    stop = n_buffer + n_obs
    times = np.arange(n_obs, dtype=np.float64) * config.dt
    return WFGBResult(
        times=times,
        x=x_windowed[start:stop],
        y=y_windowed[start:stop],
        z=z_windowed[start:stop],
        groups=tuple(groups),
        db=db,
        window_times=window_times,
        x_windowed=x_windowed,
        y_windowed=y_windowed,
        z_windowed=z_windowed,
    )
