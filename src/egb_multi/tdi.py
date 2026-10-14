"""pyTDI-facing exact-link and A/E/T helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .eccentric import (
    EccentricBinaryParams,
    EccentricHarmonicComponent,
    EccentricPhysicsOptions,
    eccentric_harmonic_source_batch,
    eccentric_physics_mode_label,
    recommended_eccentric_order,
)
from .geometry import LISAState
from .response import one_way_doppler
from .sources import GalacticBinaryParams
from .tensors import tracefree_matrix

LINKS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 2),
    (2, 1),
    (1, 0),
)
# Rosetta Stone / pyTDI convention: label ij is received at i, emitted by j.
LINK_LABELS: tuple[str, ...] = ("21", "32", "13", "31", "23", "12")


def _sampling_frequency(t: NDArray[np.float64]) -> float:
    if t.size < 2:
        raise ValueError("at least two time samples are required")
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1e-9, atol=1e-12):
        raise ValueError("pyTDI data requires a uniformly sampled time grid")
    return float(1.0 / dt[0])


def _source_waveform(source: GalacticBinaryParams):
    def h(t):
        tt = np.asarray(t, dtype=np.float64)
        phase = 2.0 * np.pi * source.f0 * tt + np.pi * source.fdot * tt**2 + source.phi0
        return np.exp(1.0j * phase)

    return h


def _source_carrier(source: GalacticBinaryParams, t: NDArray[np.float64], delay: NDArray[np.float64] | float = 0.0) -> NDArray[np.complex128]:
    retarded = t - delay
    phase = 2.0 * np.pi * source.f0 * retarded + np.pi * source.fdot * retarded**2 + source.phi0
    return np.exp(1.0j * phase)


def _harmonic_carrier(
    frequency: float,
    phase0: float,
    t: NDArray[np.float64],
    delay: NDArray[np.float64] | float = 0.0,
) -> NDArray[np.complex128]:
    return np.exp(1.0j * (2.0 * np.pi * frequency * (t - delay) + phase0))


def _transfer_series(z: NDArray[np.complex128], order: int | str) -> NDArray[np.complex128]:
    if order == "exact-transfer":
        return np.exp(z) - 1.0
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer or 'exact-transfer'")
    out = np.zeros_like(z, dtype=np.complex128)
    term = np.ones_like(z, dtype=np.complex128)
    factorial = 1.0
    for power in range(1, order + 1):
        term = term * z
        factorial *= power
        out += term / factorial
    return out


def beta_link_responses(
    state: LISAState,
    source: GalacticBinaryParams | list[GalacticBinaryParams],
    *,
    order: int | str = 1,
    retarded_geometry: bool = False,
    barycentric_phase: bool = True,
) -> dict[str, NDArray[np.complex128]]:
    """Compute six carrier-separated beta-expanded one-way link responses.

    `order` controls the Taylor expansion of the one-way transfer factor
    `exp(-2 pi i f L (1 - n.k)) - 1`. Use `"exact-transfer"` to keep this
    transfer factor exponential while retaining the same slowly varying
    geometry approximation. The default is the fastest accurate setting found
    in the current benchmarks: first-order transfer with barycentric carrier
    phase factoring. If `retarded_geometry` is true, the emitter
    spacecraft is evaluated at the approximate emission time `t - L(t)` when
    forming the link vector and emitter phase. If `barycentric_phase` is true,
    the carrier includes the barycenter delay `p0(t).k`.
    """

    sources = [source] if isinstance(source, GalacticBinaryParams) else source
    responses = {label: np.zeros(state.t.size, dtype=np.complex128) for label in LINK_LABELS}
    relative_positions = state.relative_positions

    for src in sources:
        direction = src.propagation_direction()
        matrix = tracefree_matrix(src.tracefree_coefficients())
        bary_delay = np.einsum("ti,i->t", state.barycenter, direction) if barycentric_phase else 0.0
        carrier = _source_carrier(src, state.t, bary_delay)
        freq_eff = src.f0 + src.fdot * state.t

        for label, (sender, receiver) in zip(LINK_LABELS, LINKS, strict=True):
            if retarded_geometry:
                n, length, p_sender, p_receiver = _retarded_link_geometry(state, sender, receiver)
            else:
                n = state.unit_arm(sender, receiver)
                length = state.arm_length(sender, receiver)
                p_sender = relative_positions[:, sender, :]
                p_receiver = relative_positions[:, receiver, :]
            n_dot_k = np.einsum("ti,i->t", n, direction)
            nen = np.einsum("ti,ij,tj->t", n, matrix, n)
            transfer_arg = -2j * np.pi * freq_eff * length * (1.0 - n_dot_k)
            transfer = _transfer_series(transfer_arg, order) / (1.0 - n_dot_k)
            receiver_phase = np.exp(
                -2j
                * np.pi
                * freq_eff
                * np.einsum("ti,i->t", p_receiver, direction)
            )
            responses[label] += 0.5 * nen * carrier * receiver_phase * transfer

    return responses


def harmonic_beta_link_responses(
    state: LISAState,
    components: EccentricHarmonicComponent | list[EccentricHarmonicComponent],
    *,
    order: int | str = 1,
    retarded_geometry: bool = False,
    barycentric_phase: bool = True,
) -> dict[str, NDArray[np.complex128]]:
    """Compute beta-expanded one-way links from explicit harmonic carriers."""

    component_list = [components] if isinstance(components, EccentricHarmonicComponent) else components
    responses = {label: np.zeros(state.t.size, dtype=np.complex128) for label in LINK_LABELS}
    relative_positions = state.relative_positions

    for component in component_list:
        direction = component.propagation_direction
        matrix = tracefree_matrix(component.tracefree_coefficients)
        bary_delay = np.einsum("ti,i->t", state.barycenter, direction) if barycentric_phase else 0.0
        carrier = _harmonic_carrier(component.frequency, component.phase0, state.t, bary_delay)

        for label, (sender, receiver) in zip(LINK_LABELS, LINKS, strict=True):
            if retarded_geometry:
                n, length, p_sender, p_receiver = _retarded_link_geometry(state, sender, receiver)
            else:
                n = state.unit_arm(sender, receiver)
                length = state.arm_length(sender, receiver)
                p_sender = relative_positions[:, sender, :]
                p_receiver = relative_positions[:, receiver, :]
            n_dot_k = np.einsum("ti,i->t", n, direction)
            nen = np.einsum("ti,ij,tj->t", n, matrix, n)
            transfer_arg = -2j * np.pi * component.frequency * length * (1.0 - n_dot_k)
            transfer = _transfer_series(transfer_arg, order) / (1.0 - n_dot_k)
            receiver_phase = np.exp(
                -2j
                * np.pi
                * component.frequency
                * np.einsum("ti,i->t", p_receiver, direction)
            )
            responses[label] += 0.5 * nen * carrier * receiver_phase * transfer

    return responses


def eccentric_beta_link_responses(
    state: LISAState,
    source: EccentricBinaryParams | list[EccentricBinaryParams],
    *,
    eccentric_order: int = 1,
    harmonic_samples: int = 512,
    include_negative_frequencies: bool = True,
    physics_mode: str | EccentricPhysicsOptions = "1pn",
    evolution_mode: str = "fixed",
    order: int | str = 1,
    retarded_geometry: bool = False,
    barycentric_phase: bool = True,
) -> dict[str, NDArray[np.complex128]]:
    """Compute fixed-eccentric beta-link responses from harmonic carriers."""

    return harmonic_beta_link_responses(
        state,
        eccentric_harmonic_source_batch(
            source,
            order=eccentric_order,
            samples=harmonic_samples,
            include_negative_frequencies=include_negative_frequencies,
            physics_mode=physics_mode,
            evolution_mode=evolution_mode,
        ),
        order=order,
        retarded_geometry=retarded_geometry,
        barycentric_phase=barycentric_phase,
    )


def _resolve_fixed_eccentric_order(
    source: EccentricBinaryParams | list[EccentricBinaryParams],
    eccentric_order: int | None,
) -> int:
    if eccentric_order is not None:
        return eccentric_order
    source_list = [source] if isinstance(source, EccentricBinaryParams) else source
    if not source_list:
        raise ValueError("at least one source is required")
    max_eccentricity = max(float(src.eccentricity) for src in source_list)
    resolved = recommended_eccentric_order(max_eccentricity)
    if resolved is None:
        raise ValueError(
            "automatic fixed-eccentric trilinear order selection is only recommended for eccentricity < 0.1; "
            "pass eccentric_order explicitly to run the experimental high-e harmonic truncation"
        )
    return resolved


def _grouped_jax_physics_mode_label(physics_mode: str | EccentricPhysicsOptions) -> str:
    return eccentric_physics_mode_label(physics_mode)


def fixed_eccentric_triliear_links(
    state: LISAState,
    source: EccentricBinaryParams | list[EccentricBinaryParams],
    *,
    eccentric_order: int | None = None,
    harmonic_samples: int = 512,
    include_negative_frequencies: bool = True,
    physics_mode: str | EccentricPhysicsOptions = "1pn",
    order: int | str = 1,
    retarded_geometry: bool = False,
    barycentric_phase: bool = True,
) -> dict[str, NDArray[np.complex128]]:
    """Compute fixed-eccentric trilinear one-way links.

    The source polarizations are decomposed over one radial period with
    `evolution_mode="fixed"`, then each harmonic carrier is evaluated by the
    beta/trilinear link response. Set `order="exact-transfer"` to keep the same
    fixed-eccentric harmonic model while using the exact finite-arm transfer.
    """

    resolved_order = _resolve_fixed_eccentric_order(source, eccentric_order)
    return eccentric_beta_link_responses(
        state,
        source,
        eccentric_order=resolved_order,
        harmonic_samples=harmonic_samples,
        include_negative_frequencies=include_negative_frequencies,
        physics_mode=physics_mode,
        evolution_mode="fixed",
        order=order,
        retarded_geometry=retarded_geometry,
        barycentric_phase=barycentric_phase,
    )


def _interp_columns(t_new: NDArray[np.float64], t: NDArray[np.float64], values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.stack([np.interp(t_new, t, values[:, i]) for i in range(values.shape[1])], axis=1)


def _retarded_link_geometry(
    state: LISAState,
    sender: int,
    receiver: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    p_receiver = state.relative_positions[:, receiver, :]
    zeroth_vector = state.relative_positions[:, receiver, :] - state.relative_positions[:, sender, :]
    zeroth_length = np.linalg.norm(zeroth_vector, axis=1)
    t_emit = np.clip(state.t - zeroth_length, state.t[0], state.t[-1])
    p_sender = _interp_columns(t_emit, state.t, state.relative_positions[:, sender, :])
    vector = p_receiver - p_sender
    length = np.linalg.norm(vector, axis=1)
    if np.any(length == 0.0):
        raise ValueError("retarded link geometry produced a zero-length arm")
    return vector / length[:, None], length, p_sender, p_receiver


def link_responses(
    state: LISAState,
    source: GalacticBinaryParams | list[GalacticBinaryParams],
) -> dict[str, NDArray[np.complex128]]:
    """Compute six exact one-way GW link responses keyed by pyTDI link labels.

    The outputs correspond to GW-only `sci_ij` measurements. Multiple sources
    are summed link-by-link. This path uses the non-expanded one-way Doppler
    response rather than the low-frequency triliear tensor. Eccentric source
    evolution is intentionally not included yet.
    """

    sources = [source] if isinstance(source, GalacticBinaryParams) else source
    responses = {label: np.zeros(state.t.size, dtype=np.complex128) for label in LINK_LABELS}

    for src in sources:
        coeffs = src.tracefree_coefficients()
        direction = src.propagation_direction()
        waveform = _source_waveform(src)
        for label, (sender, receiver) in zip(LINK_LABELS, LINKS, strict=True):
            responses[label] += one_way_doppler(state, sender, receiver, direction, coeffs, waveform)

    return responses


def pytdi_data_from_links(
    state: LISAState,
    links: dict[str, NDArray[np.complex128]],
    *,
    compute_delay_derivatives: bool = True,
    link_delays: dict[str, NDArray[np.float64]] | None = None,
):
    """Create a `pytdi.Data` object from six GW-only link responses.

    The link responses are assigned to `sci_ij`. All local metrology and
    reference measurements are set to zero, so pyTDI's intermediary variables
    reduce to the GW link content. Optional link_delays (seconds, labels ij)
    supplies explicit analysis delays. Clean ideal-clock/ranging comparisons
    should use deterministic orbit-derived delays rather than noisy MPRS.
    The default remains the instantaneous geometric arm lengths.
    """

    try:
        from pytdi import Data
    except ImportError as exc:
        raise ImportError("pytdi is required for pytdi_data_from_links") from exc

    missing = sorted(set(LINK_LABELS) - set(links))
    if missing:
        raise ValueError(f"missing link responses for {missing}")

    fs = _sampling_frequency(state.t)
    zero = np.zeros(state.t.size, dtype=np.complex128)
    measurements = {key: zero for key in Data.MEASUREMENTS}
    for label in LINK_LABELS:
        measurements[f"sci_{label}"] = np.asarray(links[label], dtype=np.complex128)
        if measurements[f"sci_{label}"].shape != (state.t.size,):
            raise ValueError(f"link {label} has shape {measurements[f'sci_{label}'].shape}, expected {(state.t.size,)}")

    delays = {}
    for label, (sender, receiver) in zip(LINK_LABELS, LINKS, strict=True):
        values = state.arm_length(sender, receiver) if link_delays is None else np.asarray(link_delays[label], dtype=float)
        if values.shape != (state.t.size,) or np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"Invalid delay samples for link {label}")
        delays[f"d_{label}"] = values

    data = Data(measurements, delays, fs)
    if compute_delay_derivatives:
        data.compute_delay_derivatives()
    return data


class PreparedGWTDIMichelson:
    """Cache native PyTDI factorized delays for GW-only real link responses.

    With zero local metrology, eta_ij=sci_ij. This uses the same arm/round-trip
    factorization as pytdi.michelson.compute_factorized_michelson, retaining
    native nested delays and frequency Doppler factors. Instrument noise must
    use the SCI/TMI/REF path instead. Delay preparation is fixed; measurements
    are supplied afresh at every evaluation.
    """

    def __init__(self, state, link_delays, *, generation=2, measurement_order=45, delay_order=5):
        from pytdi.core import LISATDICombination
        if generation not in (1,2) or any(v<1 or v%2!=1 for v in (measurement_order,delay_order)):
            raise ValueError('Valid TDI generation and positive odd interpolation orders required')
        self.size, self.generation, self.order = state.t.size, generation, measurement_order
        data = pytdi_data_from_links(state,{label:np.zeros(self.size) for label in LINK_LABELS},link_delays=link_delays)
        def build(components):
            return LISATDICombination(components).build(**data.args,order=delay_order)
        self.operators = []
        for labels in (('12','21','13','31'),('23','32','21','12'),('31','13','32','23')):
            a,b,c,d = [f'D_{label}' for label in labels]
            arms = (build({f'eta_{labels[0]}':[(1,[])],f'eta_{labels[1]}':[(1,[a])]}),
                    build({f'eta_{labels[2]}':[(1,[])],f'eta_{labels[3]}':[(1,[c])]}))
            if generation == 1:
                final = build({'x_arm_1':[(-1,[]),(1,[c,d])],
                    'x_arm_2':[(1,[]),(-1,[a,b])]})
                rounds = None
            else:
                rounds = (build({'arm_a':[(1,[])],'arm_b':[(1,[a,b])]}),
                          build({'arm_a':[(1,[c,d])],'arm_b':[(1,[])]}))
                final = build({'roundtrip_a':[(-1,[]),(1,[c,d,a,b])],
                    'roundtrip_b':[(1,[]),(-1,[a,b,c,d])]})
            self.operators.append((arms,rounds,final))

    def __call__(self, links):
        etas = {}
        for label in LINK_LABELS:
            value = np.asarray(links[label])
            if value.shape!=(self.size,) or np.any(~np.isfinite(value)) or (np.iscomplexobj(value) and np.any(value.imag!=0)):
                raise ValueError('Prepared GW TDI requires finite real samples on its fixed grid')
            etas[f'eta_{label}'] = value.real
        result = {}
        for channel,(arms,rounds,final) in zip(('X','Y','Z'),self.operators,strict=True):
            values = [op(etas,order=self.order,unit='frequency') for op in arms]
            if rounds is None:
                arguments = dict(zip(('x_arm_1','x_arm_2'),values,strict=True))
            else:
                arguments = {'arm_a':values[0],'arm_b':values[1]}
                values = [op(arguments,order=self.order,unit='frequency') for op in rounds]
                arguments = dict(zip(('roundtrip_a','roundtrip_b'),values,strict=True))
            result[channel] = final(arguments,order=self.order,unit='frequency')
        return result


def compute_aet(
    state: LISAState,
    source: GalacticBinaryParams | list[GalacticBinaryParams],
    *,
    generation: int = 1,
    measurement_order: int = 7,
    delay_order: int = 5,
) -> dict[str, NDArray[np.complex128]]:
    """Compute pyTDI Michelson A/E/T from six exact-link responses."""

    try:
        from pytdi.ortho import compute_factorized_ortho
    except ImportError as exc:
        raise ImportError("pytdi is required for compute_aet") from exc

    data = pytdi_data_from_links(state, link_responses(state, source))
    return compute_factorized_ortho(
        data,
        kind="Michelson",
        order=measurement_order,
        delay_order=delay_order,
        generation=generation,
        unit="frequency",
    )


def compute_xyz(
    state: LISAState,
    source: GalacticBinaryParams | list[GalacticBinaryParams],
    *,
    generation: int = 1,
    measurement_order: int = 7,
    delay_order: int = 5,
) -> dict[str, NDArray[np.complex128]]:
    """Compute pyTDI Michelson X/Y/Z from six exact-link responses."""

    try:
        from pytdi.michelson import compute_factorized_michelson
    except ImportError as exc:
        raise ImportError("pytdi is required for compute_xyz") from exc

    data = pytdi_data_from_links(state, link_responses(state, source))
    return {
        "X": compute_factorized_michelson(
            data,
            rot=0,
            order=measurement_order,
            delay_order=delay_order,
            generation=generation,
            unit="frequency",
        ),
        "Y": compute_factorized_michelson(
            data,
            rot=1,
            order=measurement_order,
            delay_order=delay_order,
            generation=generation,
            unit="frequency",
        ),
        "Z": compute_factorized_michelson(
            data,
            rot=2,
            order=measurement_order,
            delay_order=delay_order,
            generation=generation,
            unit="frequency",
        ),
    }


def compute_beta_xyz(
    state: LISAState,
    source: GalacticBinaryParams | list[GalacticBinaryParams],
    *,
    order: int | str = 1,
    retarded_geometry: bool = False,
    barycentric_phase: bool = True,
    generation: int = 1,
    measurement_order: int = 3,
    delay_order: int = 3,
) -> dict[str, NDArray[np.complex128]]:
    """Compute pyTDI Michelson X/Y/Z from beta-expanded six-link responses.

    Defaults are the current recommended triliear settings: `order=1`,
    `barycentric_phase=True`, `measurement_order=3`, and `delay_order=3`.
    Other beta orders, exact transfer, retarded geometry, and interpolation
    orders remain available as explicit options.
    """

    try:
        from pytdi.michelson import compute_factorized_michelson
    except ImportError as exc:
        raise ImportError("pytdi is required for compute_beta_xyz") from exc

    data = pytdi_data_from_links(
        state,
        beta_link_responses(
            state,
            source,
            order=order,
            retarded_geometry=retarded_geometry,
            barycentric_phase=barycentric_phase,
        ),
    )
    return {
        "X": compute_factorized_michelson(data, rot=0, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
        "Y": compute_factorized_michelson(data, rot=1, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
        "Z": compute_factorized_michelson(data, rot=2, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
    }


def compute_triliear_xyz_eccentric(
    state: LISAState,
    source: EccentricBinaryParams | list[EccentricBinaryParams],
    *,
    eccentric_order: int = 1,
    harmonic_samples: int = 512,
    include_negative_frequencies: bool = True,
    physics_mode: str | EccentricPhysicsOptions = "1pn",
    evolution_mode: str = "fixed",
    order: int | str = 1,
    retarded_geometry: bool = False,
    barycentric_phase: bool = True,
    generation: int = 1,
    measurement_order: int = 3,
    delay_order: int = 3,
) -> dict[str, NDArray[np.complex128]]:
    """Compute experimental fixed-eccentric small-e triliear pyTDI X/Y/Z."""

    try:
        from pytdi.michelson import compute_factorized_michelson
    except ImportError as exc:
        raise ImportError("pytdi is required for compute_triliear_xyz_eccentric") from exc

    data = pytdi_data_from_links(
        state,
        eccentric_beta_link_responses(
            state,
            source,
            eccentric_order=eccentric_order,
            harmonic_samples=harmonic_samples,
            include_negative_frequencies=include_negative_frequencies,
            physics_mode=physics_mode,
            evolution_mode=evolution_mode,
            order=order,
            retarded_geometry=retarded_geometry,
            barycentric_phase=barycentric_phase,
        ),
    )
    return {
        "X": compute_factorized_michelson(data, rot=0, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
        "Y": compute_factorized_michelson(data, rot=1, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
        "Z": compute_factorized_michelson(data, rot=2, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
    }


def compute_fixed_eccentric_triliear_xyz(
    state: LISAState,
    source: EccentricBinaryParams | list[EccentricBinaryParams],
    *,
    eccentric_order: int | None = None,
    harmonic_samples: int = 512,
    include_negative_frequencies: bool = True,
    physics_mode: str | EccentricPhysicsOptions = "1pn",
    order: int | str = 1,
    retarded_geometry: bool = False,
    barycentric_phase: bool = True,
    generation: int = 1,
    measurement_order: int = 3,
    delay_order: int = 3,
) -> dict[str, NDArray[np.complex128]]:
    """Compute fixed-eccentric trilinear pyTDI X/Y/Z.

    This is the named fixed-eccentric model: source parameters are held fixed,
    harmonic carriers are extracted from one radial period, and the harmonics
    are passed through the beta/trilinear response. Use `order="exact-transfer"`
    for the exact finite-arm transfer diagnostic.
    """

    try:
        from pytdi.michelson import compute_factorized_michelson
    except ImportError as exc:
        raise ImportError("pytdi is required for compute_fixed_eccentric_triliear_xyz") from exc

    data = pytdi_data_from_links(
        state,
        fixed_eccentric_triliear_links(
            state,
            source,
            eccentric_order=eccentric_order,
            harmonic_samples=harmonic_samples,
            include_negative_frequencies=include_negative_frequencies,
            physics_mode=physics_mode,
            order=order,
            retarded_geometry=retarded_geometry,
            barycentric_phase=barycentric_phase,
        ),
    )
    return {
        "X": compute_factorized_michelson(data, rot=0, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
        "Y": compute_factorized_michelson(data, rot=1, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
        "Z": compute_factorized_michelson(data, rot=2, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
    }


def fixed_eccentric_triliear_links_jax(
    state: LISAState,
    source: EccentricBinaryParams | list[EccentricBinaryParams],
    *,
    geometry: dict[str, NDArray[np.float64]] | None = None,
    basis: str = "harmonic",
    eccentric_order: int | None = None,
    sideband_q_max: int | None = None,
    harmonic_samples: int = 512,
    fit_cycles: float = 4.0,
    include_radial: bool = True,
    include_quadrupole_minus: bool = True,
    include_negative_frequencies: bool = True,
    coefficient_tolerance: float | None = None,
    regularization: float = 0.0,
    physics_mode: str | EccentricPhysicsOptions = "1pn",
    order: int | str = 1,
    batch_size: int = 128,
    barycentric_phase: bool = True,
) -> dict[str, NDArray[np.complex128]]:
    """Compute fast fixed-eccentric trilinear links with grouped JAX kernels.

    `basis="harmonic"` is the original small-eccentricity harmonic ansatz.
    `basis="sideband"` uses the explicit radial/periastron sideband basis and
    fits it inside the grouped JAX response kernel.
    """

    basis_normalized = basis.lower()
    if basis_normalized not in {"harmonic", "sideband"}:
        raise ValueError("basis must be 'harmonic' or 'sideband'")
    physics_mode_label = _grouped_jax_physics_mode_label(physics_mode)
    if physics_mode_label not in {"newtonian", "1pn_no_periastron", "1pn"}:
        raise ValueError("physics_mode must be 'newtonian', '1pn_no_periastron', '1pn', or '1pn_periastron'")
    if order == "exact-transfer":
        response_mode = "fastgb_transfer"
    elif order == 1:
        response_mode = "triliear_beta"
    else:
        raise ValueError("grouped JAX fixed-eccentric trilinear links support order=1 or order='exact-transfer'")

    from .eccentric_jax import eccentric_harmonic_links_grouped_jax, eccentric_sideband_links_grouped_jax, precompute_jax_link_geometry

    resolved_geometry = precompute_jax_link_geometry(state) if geometry is None else geometry
    if basis_normalized == "harmonic":
        resolved_order = _resolve_fixed_eccentric_order(source, eccentric_order)
        return eccentric_harmonic_links_grouped_jax(
            source,
            resolved_geometry,
            eccentric_order=resolved_order,
            harmonic_samples=harmonic_samples,
            batch_size=batch_size,
            response_mode=response_mode,
            barycentric_phase=barycentric_phase,
            include_negative_frequencies=include_negative_frequencies,
            physics_mode=physics_mode_label,
        )

    resolved_q_max = sideband_q_max if sideband_q_max is not None else eccentric_order
    if resolved_q_max is None:
        raise ValueError("sideband_q_max is required when basis='sideband'")
    if physics_mode_label != "1pn":
        raise ValueError("basis='sideband' in the grouped JAX path currently supports physics_mode='1pn' or '1pn_periastron'")
    return eccentric_sideband_links_grouped_jax(
        source,
        resolved_geometry,
        q_max=resolved_q_max,
        harmonic_samples=harmonic_samples,
        fit_cycles=fit_cycles,
        batch_size=batch_size,
        response_mode=response_mode,
        barycentric_phase=barycentric_phase,
        include_radial=include_radial,
        include_quadrupole_minus=include_quadrupole_minus,
        include_negative_frequencies=include_negative_frequencies,
        coefficient_tolerance=coefficient_tolerance,
        regularization=regularization,
        physics_mode=physics_mode_label,
    )


def compute_fixed_eccentric_triliear_xyz_jax(
    state: LISAState,
    source: EccentricBinaryParams | list[EccentricBinaryParams],
    *,
    geometry: dict[str, NDArray[np.float64]] | None = None,
    basis: str = "harmonic",
    eccentric_order: int | None = None,
    sideband_q_max: int | None = None,
    harmonic_samples: int = 512,
    fit_cycles: float = 4.0,
    include_radial: bool = True,
    include_quadrupole_minus: bool = True,
    include_negative_frequencies: bool = True,
    coefficient_tolerance: float | None = None,
    regularization: float = 0.0,
    physics_mode: str | EccentricPhysicsOptions = "1pn",
    order: int | str = 1,
    batch_size: int = 128,
    barycentric_phase: bool = True,
    generation: int = 1,
    measurement_order: int = 3,
    delay_order: int = 3,
) -> dict[str, NDArray[np.complex128]]:
    """Compute fast fixed-eccentric trilinear pyTDI X/Y/Z with grouped JAX links."""

    try:
        from pytdi.michelson import compute_factorized_michelson
    except ImportError as exc:
        raise ImportError("pytdi is required for compute_fixed_eccentric_triliear_xyz_jax") from exc

    data = pytdi_data_from_links(
        state,
        fixed_eccentric_triliear_links_jax(
            state,
            source,
            geometry=geometry,
            basis=basis,
            eccentric_order=eccentric_order,
            sideband_q_max=sideband_q_max,
            harmonic_samples=harmonic_samples,
            fit_cycles=fit_cycles,
            include_radial=include_radial,
            include_quadrupole_minus=include_quadrupole_minus,
            include_negative_frequencies=include_negative_frequencies,
            coefficient_tolerance=coefficient_tolerance,
            regularization=regularization,
            physics_mode=physics_mode,
            order=order,
            batch_size=batch_size,
            barycentric_phase=barycentric_phase,
        ),
    )
    return {
        "X": compute_factorized_michelson(data, rot=0, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
        "Y": compute_factorized_michelson(data, rot=1, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
        "Z": compute_factorized_michelson(data, rot=2, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
    }


def fixed_eccentric_sideband_links_jax(
    state: LISAState,
    source: EccentricBinaryParams | list[EccentricBinaryParams],
    *,
    geometry: dict[str, NDArray[np.float64]] | None = None,
    q_max: int,
    harmonic_samples: int = 2048,
    fit_cycles: float = 4.0,
    include_radial: bool = True,
    include_quadrupole_minus: bool = True,
    include_negative_frequencies: bool = True,
    coefficient_tolerance: float | None = None,
    regularization: float = 0.0,
    physics_mode: str | EccentricPhysicsOptions = "1pn",
    batch_size: int = 128,
    barycentric_phase: bool = True,
) -> dict[str, NDArray[np.complex128]]:
    """Compute fixed-eccentric exact-transfer links from fitted sidebands.

    This is the explicit sideband convenience wrapper. For the default 1PN
    fixed source model, it uses the grouped source-batched sideband JAX kernel.
    """

    physics_arg = _grouped_jax_physics_mode_label(physics_mode)
    is_supported_physics = physics_arg == "1pn"

    from .eccentric_jax import eccentric_sideband_links_grouped_jax, eccentric_sideband_links_jax, precompute_jax_link_geometry

    resolved_geometry = precompute_jax_link_geometry(state) if geometry is None else geometry
    if is_supported_physics:
        return eccentric_sideband_links_grouped_jax(
            source,
            resolved_geometry,
            q_max=q_max,
            harmonic_samples=harmonic_samples,
            fit_cycles=fit_cycles,
            batch_size=batch_size,
            response_mode="fastgb_transfer",
            barycentric_phase=barycentric_phase,
            include_radial=include_radial,
            include_quadrupole_minus=include_quadrupole_minus,
            include_negative_frequencies=include_negative_frequencies,
            coefficient_tolerance=coefficient_tolerance,
            regularization=regularization,
            physics_mode="1pn",
        )

    return eccentric_sideband_links_jax(
        source,
        resolved_geometry,
        q_max=q_max,
        harmonic_samples=harmonic_samples,
        fit_cycles=fit_cycles,
        batch_size=batch_size,
        response_mode="fastgb_transfer",
        barycentric_phase=barycentric_phase,
        include_radial=include_radial,
        include_quadrupole_minus=include_quadrupole_minus,
        include_negative_frequencies=include_negative_frequencies,
        coefficient_tolerance=coefficient_tolerance,
        regularization=regularization,
        physics_mode=physics_arg,
    )


def compute_fixed_eccentric_sideband_xyz_jax(
    state: LISAState,
    source: EccentricBinaryParams | list[EccentricBinaryParams],
    *,
    geometry: dict[str, NDArray[np.float64]] | None = None,
    q_max: int,
    harmonic_samples: int = 2048,
    fit_cycles: float = 4.0,
    include_radial: bool = True,
    include_quadrupole_minus: bool = True,
    include_negative_frequencies: bool = True,
    coefficient_tolerance: float | None = None,
    regularization: float = 0.0,
    physics_mode: str | EccentricPhysicsOptions = "1pn",
    batch_size: int = 128,
    barycentric_phase: bool = True,
    generation: int = 1,
    measurement_order: int = 3,
    delay_order: int = 3,
) -> dict[str, NDArray[np.complex128]]:
    """Compute fixed-eccentric exact-transfer pyTDI X/Y/Z from fitted sidebands."""

    try:
        from pytdi.michelson import compute_factorized_michelson
    except ImportError as exc:
        raise ImportError("pytdi is required for compute_fixed_eccentric_sideband_xyz_jax") from exc

    data = pytdi_data_from_links(
        state,
        fixed_eccentric_sideband_links_jax(
            state,
            source,
            geometry=geometry,
            q_max=q_max,
            harmonic_samples=harmonic_samples,
            fit_cycles=fit_cycles,
            include_radial=include_radial,
            include_quadrupole_minus=include_quadrupole_minus,
            include_negative_frequencies=include_negative_frequencies,
            coefficient_tolerance=coefficient_tolerance,
            regularization=regularization,
            physics_mode=physics_mode,
            batch_size=batch_size,
            barycentric_phase=barycentric_phase,
        ),
    )
    return {
        "X": compute_factorized_michelson(data, rot=0, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
        "Y": compute_factorized_michelson(data, rot=1, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
        "Z": compute_factorized_michelson(data, rot=2, order=measurement_order, delay_order=delay_order, generation=generation, unit="frequency"),
    }


def triliear_links(
    state: LISAState,
    source: GalacticBinaryParams | list[GalacticBinaryParams],
    **kwargs,
) -> dict[str, NDArray[np.complex128]]:
    """Recommended beta-link triliear response; options forward to `beta_link_responses`."""

    return beta_link_responses(state, source, **kwargs)


def compute_triliear_xyz(
    state: LISAState,
    source: GalacticBinaryParams | list[GalacticBinaryParams],
    **kwargs,
) -> dict[str, NDArray[np.complex128]]:
    """Recommended triliear pyTDI X/Y/Z response; options forward to `compute_beta_xyz`."""

    return compute_beta_xyz(state, source, **kwargs)
