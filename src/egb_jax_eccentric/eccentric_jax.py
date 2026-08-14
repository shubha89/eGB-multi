"""Optional JAX kernels for batched eccentric source generation."""

from __future__ import annotations

from functools import partial

import numpy as np
from numpy.typing import NDArray

from .constants import C_M_PER_S, G_M3_KG_S2, KILOPARSEC_M, SOLAR_MASS_KG
from .eccentric import (
    EccentricBinaryParams,
    EccentricHarmonicComponent,
    eccentric_harmonic_indices,
    eccentric_harmonic_source_batch,
    eccentric_physics_mode_label,
)
from .geometry import LISAState

JAX_LINKS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 2),
    (2, 1),
    (1, 0),
)
# Rosetta Stone / pyTDI convention: label ij is received at i, emitted by j.
JAX_LINK_LABELS: tuple[str, ...] = ("21", "32", "13", "31", "23", "12")
INTERPOLATION_HALO_SAMPLES = 128

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    jax = None
    jnp = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def require_jax() -> None:
    if _IMPORT_ERROR is not None:
        raise ImportError("JAX is required for egb_jax_eccentric.eccentric_jax") from _IMPORT_ERROR


def _physics_switches(physics_mode: str) -> tuple[bool, bool]:
    physics_mode = eccentric_physics_mode_label(physics_mode)
    if physics_mode == "newtonian":
        return False, False
    if physics_mode == "1pn_no_periastron":
        return True, False
    if physics_mode == "1pn":
        return True, True
    raise ValueError("physics_mode must be 'newtonian', '1pn_no_periastron', '1pn', or '1pn_periastron'")


def pack_eccentric_sources(sources: EccentricBinaryParams | list[EccentricBinaryParams]) -> NDArray[np.float64]:
    """Pack source dataclasses into a JAX-friendly numeric array."""

    source_list = [sources] if isinstance(sources, EccentricBinaryParams) else sources
    return np.array(
        [
            [
                source.mean_motion,
                source.eccentricity,
                source.m1_solar,
                source.m2_solar,
                source.distance_m,
                source.inclination,
                source.phi0,
                source.t0,
                source.fdot,
            ]
            for source in source_list
        ],
        dtype=np.float64,
    )


def pack_eccentric_response_sources(sources: EccentricBinaryParams | list[EccentricBinaryParams]) -> NDArray[np.float64]:
    """Pack eccentric source and sky parameters for JAX link-response kernels."""

    source_list = [sources] if isinstance(sources, EccentricBinaryParams) else sources
    return np.array(
        [
            [
                source.mean_motion,
                source.eccentricity,
                source.m1_solar,
                source.m2_solar,
                source.distance_m,
                source.inclination,
                source.beta,
                source.lambda_,
                source.psi,
                source.phi0,
                source.t0,
                source.fdot,
            ]
            for source in source_list
        ],
        dtype=np.float64,
    )


def pad_packed_sources(params: NDArray[np.float64], batch_size: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pad packed source arrays to a fixed batch size and return a source mask."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    packed = np.asarray(params, dtype=np.float64)
    if packed.ndim != 2:
        raise ValueError("params must have shape (source, parameter)")
    if packed.shape[0] > batch_size:
        raise ValueError("params has more rows than batch_size")
    out = np.zeros((batch_size, packed.shape[1]), dtype=np.float64)
    mask = np.zeros(batch_size, dtype=np.float64)
    out[: packed.shape[0]] = packed
    if packed.shape[0]:
        out[packed.shape[0] :] = packed[0]
    mask[: packed.shape[0]] = 1.0
    return out, mask


def precompute_jax_link_geometry(
    state: LISAState,
    *,
    position_origin: str = "ssb",
) -> dict[str, NDArray[np.float64]]:
    """Precompute link geometry arrays used by the JAX brute-force kernel."""

    if position_origin not in {"ssb", "barycenter"}:
        raise ValueError("position_origin must be 'ssb' or 'barycenter'")
    n = []
    length = []
    p_sender = []
    p_receiver = []
    positions = state.positions if position_origin == "ssb" else state.relative_positions
    for sender, receiver in JAX_LINKS:
        n.append(state.unit_arm(sender, receiver))
        length.append(state.arm_length(sender, receiver))
        p_sender.append(positions[:, sender, :])
        p_receiver.append(positions[:, receiver, :])
    return {
        "times": np.asarray(state.t, dtype=np.float64),
        "n": np.stack(n, axis=0),
        "length": np.stack(length, axis=0),
        "p_sender": np.stack(p_sender, axis=0),
        "p_receiver": np.stack(p_receiver, axis=0),
        "barycenter": np.asarray(state.barycenter, dtype=np.float64),
        "position_origin": position_origin,
    }


def pack_harmonic_components(components: EccentricHarmonicComponent | list[EccentricHarmonicComponent]) -> NDArray[np.float64]:
    """Pack harmonic components for JAX beta-link kernels."""

    component_list = [components] if isinstance(components, EccentricHarmonicComponent) else components
    return np.array(
        [
            [
                component.frequency,
                *np.real(component.tracefree_coefficients),
                *np.imag(component.tracefree_coefficients),
                *component.propagation_direction,
                component.phase0,
            ]
            for component in component_list
        ],
        dtype=np.float64,
    )


if jax is not None:

    def _mikkola_kepler_jax(mean_anomaly, eccentricity):
        wrapped = (mean_anomaly + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        offset = mean_anomaly - wrapped

        alpha = (1.0 - eccentricity) / (4.0 * eccentricity + 0.5)
        beta = 0.5 * wrapped / (4.0 * eccentricity + 0.5)
        root = jnp.sqrt(beta * beta + alpha**3)
        z = jnp.cbrt(jnp.where(wrapped >= 0.0, beta + root, beta - root))
        s = z - alpha / z
        omega = s - 0.078 * s**5 / (1.0 + eccentricity)
        u0 = wrapped + eccentricity * (3.0 * omega - 4.0 * omega**3)

        sin_u = jnp.sin(u0)
        cos_u = jnp.cos(u0)
        f = u0 - eccentricity * sin_u - wrapped
        f1 = 1.0 - eccentricity * cos_u
        f2 = eccentricity * sin_u
        f3 = eccentricity * cos_u
        f4 = -eccentricity * sin_u
        du1 = -f / f1
        du2 = -f / (f1 + 0.5 * f2 * du1)
        du3 = -f / (f1 + 0.5 * f2 * du2 + f3 * du2**2 / 6.0)
        du4 = -f / (f1 + 0.5 * f2 * du3 + f3 * du3**2 / 6.0 + f4 * du3**3 / 24.0)
        solved = u0 + du4 + offset
        return jnp.where(eccentricity == 0.0, mean_anomaly, solved)

    def _single_source_complex_strain_with_options(params, times, include_1pn_orbital_corrections: bool, include_periastron_advance: bool):
        azimuthal_frequency0, eccentricity, m1_solar, m2_solar, distance_m, inclination, phi0, t0, fdot = params
        dt = times - t0
        azimuthal_frequency_dot = jnp.pi * fdot
        azimuthal_frequency_eff = azimuthal_frequency0 + azimuthal_frequency_dot * dt
        total_solar = m1_solar + m2_solar
        total_mass = total_solar * SOLAR_MASS_KG
        eta = m1_solar * m2_solar / total_solar**2
        gm = G_M3_KG_S2 * total_mass
        x_full = (gm * azimuthal_frequency_eff / C_M_PER_S**3) ** (2.0 / 3.0)
        x = x_full if include_1pn_orbital_corrections else 0.0

        one_minus_e2 = 1.0 - eccentricity * eccentricity
        x0 = (gm * azimuthal_frequency0 / C_M_PER_S**3) ** (2.0 / 3.0)
        k0 = 3.0 * x0 / one_minus_e2 if include_periastron_advance else 0.0
        radial_frequency0 = azimuthal_frequency0 * (1.0 - k0)
        radial_frequency_dot = azimuthal_frequency_dot * (1.0 - k0)
        radial_frequency_eff = radial_frequency0 + radial_frequency_dot * dt
        mean_anomaly = radial_frequency0 * dt + 0.5 * radial_frequency_dot * dt**2
        u = _mikkola_kepler_jax(mean_anomaly, eccentricity)
        one_minus_ecosu = 1.0 - eccentricity * jnp.cos(u)
        sqrt_one_minus_e2 = jnp.sqrt(one_minus_e2)

        if include_1pn_orbital_corrections:
            nu = one_minus_ecosu
            r_1pn = (
                -24.0
                + 9.0 * eta
                + nu * (18.0 - 7.0 * eta)
                + eccentricity
                * eccentricity
                * (24.0 - 9.0 * eta + nu * (-6.0 + 7.0 * eta))
            ) / (6.0 * nu * one_minus_e2)
            r = gm * one_minus_ecosu * (1.0 + x * r_1pn) / (C_M_PER_S**2 * x_full)
            rdot_1pn = (
                -7.0 * eta + eccentricity * eccentricity * (-6.0 + 7.0 * eta)
            ) / (6.0 * one_minus_e2)
            rdot = (
                C_M_PER_S
                * jnp.sqrt(x_full)
                * eccentricity
                * jnp.sin(u)
                / one_minus_ecosu
                * (1.0 + x * rdot_1pn)
            )
            phidot_1pn = ((-1.0 + nu + eccentricity * eccentricity) * (-4.0 + eta)) / (nu * one_minus_e2)
        else:
            semimajor_scale = (gm / radial_frequency_eff**2) ** (1.0 / 3.0)
            r = semimajor_scale * one_minus_ecosu
            rdot = eccentricity * (gm * radial_frequency_eff) ** (1.0 / 3.0) * jnp.sin(u) / one_minus_ecosu
            phidot_1pn = 0.0

        safe_e = jnp.where(jnp.abs(eccentricity) > 1.0e-14, eccentricity, 1.0)
        beta_newtonian = (1.0 - sqrt_one_minus_e2) / safe_e
        beta_1pn = (
            -4.0
            + eta
            + eccentricity * eccentricity * (8.0 - 2.0 * eta)
            + (4.0 - eta) * sqrt_one_minus_e2
        ) / (safe_e * sqrt_one_minus_e2)
        beta_phi = jnp.where(
            jnp.abs(eccentricity) > 1.0e-14,
            beta_newtonian + x * beta_1pn,
            0.0,
        )
        v_minus_u = jnp.unwrap(
            2.0 * jnp.arctan2(
                beta_phi * jnp.sin(u),
                1.0 - beta_phi * jnp.cos(u),
            )
        )
        w = v_minus_u + eccentricity * jnp.sin(u)
        if include_1pn_orbital_corrections:
            w = w + x * 3.0 * (eccentricity * jnp.sin(u) + v_minus_u) / one_minus_e2
        lam = (
            azimuthal_frequency0 * dt + 0.5 * azimuthal_frequency_dot * dt**2 + phi0
            if include_periastron_advance
            else mean_anomaly + phi0
        )
        phi = lam + w
        phidot = (
            azimuthal_frequency_eff
            * sqrt_one_minus_e2
            / one_minus_ecosu**2
            * (1.0 + x * phidot_1pn)
        )

        cos_i = jnp.cos(inclination)
        sin_i = jnp.sin(inclination)
        sin_2phi = jnp.sin(2.0 * phi)
        cos_2phi = jnp.cos(2.0 * phi)
        radial = gm / r
        tangential = r * r * phidot * phidot
        radial_velocity = rdot * rdot
        a = radial + tangential - radial_velocity
        b = 2.0 * r * rdot * phidot

        strain_scale = G_M3_KG_S2 * total_mass * eta / (C_M_PER_S**4 * distance_m)
        h_cross = -2.0 * strain_scale * cos_i * (a * sin_2phi - b * cos_2phi)
        h_plus = -strain_scale * (
            (1.0 + cos_i * cos_i) * (a * cos_2phi + b * sin_2phi)
            + sin_i * sin_i * (radial - tangential - radial_velocity)
        )
        return h_plus - 1.0j * h_cross

    def _single_source_complex_strain(params, times):
        return _single_source_complex_strain_with_options(params, times, True, True)

    def _single_source_polarizations_with_options(response_params, times, include_1pn_orbital_corrections: bool, include_periastron_advance: bool):
        source_params = jnp.array(
            [
                response_params[0],
                response_params[1],
                response_params[2],
                response_params[3],
                response_params[4],
                response_params[5],
                response_params[9],
                response_params[10],
                response_params[11],
            ]
        )
        complex_strain = _single_source_complex_strain_with_options(
            source_params,
            times,
            include_1pn_orbital_corrections,
            include_periastron_advance,
        )
        return jnp.real(complex_strain), -jnp.imag(complex_strain)

    def _single_source_polarizations(response_params, times):
        return _single_source_polarizations_with_options(response_params, times, True, True)

    def _source_direction(response_params):
        beta = response_params[6]
        lambda_ = response_params[7]
        cos_beta = jnp.cos(beta)
        return jnp.array(
            [
                -cos_beta * jnp.cos(lambda_),
                -cos_beta * jnp.sin(lambda_),
                -jnp.sin(beta),
            ],
            dtype=jnp.float64,
        )

    def _source_rotated_basis(response_params):
        beta = response_params[6]
        lambda_ = response_params[7]
        psi = response_params[8]
        theta = jnp.pi / 2.0 - beta
        sinth, costh = jnp.sin(theta), jnp.cos(theta)
        sinph, cosph = jnp.sin(lambda_), jnp.cos(lambda_)

        u = jnp.array([costh * cosph, costh * sinph, -sinth], dtype=jnp.float64)
        v = jnp.array([sinph, -cosph, 0.0], dtype=jnp.float64)
        eplus = jnp.outer(v, v) - jnp.outer(u, u)
        ecross = jnp.outer(u, v) + jnp.outer(v, u)
        cosps = jnp.cos(2.0 * psi)
        sinps = jnp.sin(2.0 * psi)
        return cosps * eplus - sinps * ecross, sinps * eplus + cosps * ecross

    def _single_source_links(response_params, times, n, length, p_sender, p_receiver):
        direction = _source_direction(response_params)
        eplus, ecross = _source_rotated_basis(response_params)
        denominator = 1.0 - jnp.einsum("lti,i->lt", n, direction)
        delay_sender = times[None, :] - length - jnp.einsum("lti,i->lt", p_sender, direction)
        delay_receiver = times[None, :] - jnp.einsum("lti,i->lt", p_receiver, direction)

        hplus_sender, hcross_sender = _single_source_polarizations(response_params, delay_sender)
        hplus_receiver, hcross_receiver = _single_source_polarizations(response_params, delay_receiver)
        qplus = jnp.einsum("lti,ij,ltj->lt", n, eplus, n)
        qcross = jnp.einsum("lti,ij,ltj->lt", n, ecross, n)
        h_sender = qplus * hplus_sender + qcross * hcross_sender
        h_receiver = qplus * hplus_receiver + qcross * hcross_receiver
        return 0.5 * (h_sender - h_receiver) / denominator

    def _single_source_links_with_options(
        response_params,
        times,
        n,
        length,
        p_sender,
        p_receiver,
        include_1pn_orbital_corrections: bool,
        include_periastron_advance: bool,
    ):
        direction = _source_direction(response_params)
        eplus, ecross = _source_rotated_basis(response_params)
        denominator = 1.0 - jnp.einsum("lti,i->lt", n, direction)
        delay_sender = times[None, :] - length - jnp.einsum("lti,i->lt", p_sender, direction)
        delay_receiver = times[None, :] - jnp.einsum("lti,i->lt", p_receiver, direction)

        hplus_sender, hcross_sender = _single_source_polarizations_with_options(
            response_params,
            delay_sender,
            include_1pn_orbital_corrections,
            include_periastron_advance,
        )
        hplus_receiver, hcross_receiver = _single_source_polarizations_with_options(
            response_params,
            delay_receiver,
            include_1pn_orbital_corrections,
            include_periastron_advance,
        )
        qplus = jnp.einsum("lti,ij,ltj->lt", n, eplus, n)
        qcross = jnp.einsum("lti,ij,ltj->lt", n, ecross, n)
        h_sender = qplus * hplus_sender + qcross * hcross_sender
        h_receiver = qplus * hplus_receiver + qcross * hcross_receiver
        return 0.5 * (h_sender - h_receiver) / denominator

    def _interpolate_on_grid(times, values, query_times, interpolation_order: int):
        dt = times[1] - times[0]
        floating_index = (query_times - times[0]) / dt
        left = jnp.floor(floating_index).astype(jnp.int32)
        if interpolation_order == 1:
            start = jnp.clip(left, 0, values.shape[0] - 2)
        else:
            start = jnp.clip(left - interpolation_order // 2, 0, values.shape[0] - interpolation_order - 1)
        offsets = jnp.arange(interpolation_order + 1, dtype=jnp.int32)
        indices = start[..., None] + offsets
        nodes = start[..., None].astype(jnp.float64) + offsets.astype(jnp.float64)

        weights = []
        for j in range(interpolation_order + 1):
            weight = jnp.ones_like(floating_index, dtype=jnp.float64)
            for m in range(interpolation_order + 1):
                if m != j:
                    weight = weight * (floating_index - nodes[..., m]) / (nodes[..., j] - nodes[..., m])
            weights.append(weight)
        weights_array = jnp.stack(weights, axis=-1)
        return jnp.sum(values[indices] * weights_array, axis=-1)

    def _single_source_links_interpolated(response_params, times, n, length, p_sender, p_receiver, interpolation_order: int):
        direction = _source_direction(response_params)
        eplus, ecross = _source_rotated_basis(response_params)
        denominator = 1.0 - jnp.einsum("lti,i->lt", n, direction)
        delay_sender = times[None, :] - length - jnp.einsum("lti,i->lt", p_sender, direction)
        delay_receiver = times[None, :] - jnp.einsum("lti,i->lt", p_receiver, direction)

        dt = times[1] - times[0]
        left_halo = times[0] - dt * jnp.arange(INTERPOLATION_HALO_SAMPLES, 0, -1, dtype=jnp.float64)
        right_halo = times[-1] + dt * jnp.arange(1, INTERPOLATION_HALO_SAMPLES + 1, dtype=jnp.float64)
        waveform_times = jnp.concatenate((left_halo, times, right_halo))
        hplus_grid, hcross_grid = _single_source_polarizations(response_params, waveform_times)
        hplus_sender = _interpolate_on_grid(waveform_times, hplus_grid, delay_sender, interpolation_order)
        hcross_sender = _interpolate_on_grid(waveform_times, hcross_grid, delay_sender, interpolation_order)
        hplus_receiver = _interpolate_on_grid(waveform_times, hplus_grid, delay_receiver, interpolation_order)
        hcross_receiver = _interpolate_on_grid(waveform_times, hcross_grid, delay_receiver, interpolation_order)

        qplus = jnp.einsum("lti,ij,ltj->lt", n, eplus, n)
        qcross = jnp.einsum("lti,ij,ltj->lt", n, ecross, n)
        h_sender = qplus * hplus_sender + qcross * hcross_sender
        h_receiver = qplus * hplus_receiver + qcross * hcross_receiver
        return 0.5 * (h_sender - h_receiver) / denominator

    def _tracefree_matrix_from_coeffs(coeffs):
        return jnp.array(
            [
                [coeffs[0], coeffs[1], coeffs[2]],
                [coeffs[1], coeffs[3], coeffs[4]],
                [coeffs[2], coeffs[4], -coeffs[0] - coeffs[3]],
            ],
            dtype=jnp.complex128,
        )

    def _matrix_to_tracefree5_jax(matrix):
        return jnp.array(
            [
                matrix[0, 0],
                matrix[0, 1],
                matrix[0, 2],
                matrix[1, 1],
                matrix[1, 2],
            ],
            dtype=jnp.complex128,
        )

    def _single_source_harmonic_component_params(response_params, harmonic_indices, phase_grid, coefficient_scale):
        mean_motion = response_params[0]
        sample_times = response_params[10] + phase_grid / mean_motion
        hplus, hcross = _single_source_polarizations(response_params, sample_times)
        hplus_coeffs = jnp.fft.fft(hplus) / phase_grid.size
        hcross_coeffs = jnp.fft.fft(hcross) / phase_grid.size
        hplus_harmonics = coefficient_scale * hplus_coeffs[harmonic_indices]
        hcross_harmonics = coefficient_scale * hcross_coeffs[harmonic_indices]

        eplus, ecross = _source_rotated_basis(response_params)
        matrices = hplus_harmonics[:, None, None] * eplus[None, :, :] + hcross_harmonics[:, None, None] * ecross[None, :, :]
        coeffs = jax.vmap(_matrix_to_tracefree5_jax)(matrices)

        frequencies = harmonic_indices.astype(jnp.float64) * mean_motion / (2.0 * jnp.pi)
        direction = _source_direction(response_params)
        repeated_direction = jnp.broadcast_to(direction[None, :], (harmonic_indices.size, 3))
        phase0 = jnp.zeros((harmonic_indices.size, 1), dtype=jnp.float64)
        return jnp.concatenate(
            (
                frequencies[:, None],
                jnp.real(coeffs),
                jnp.imag(coeffs),
                repeated_direction,
                phase0,
            ),
            axis=1,
        )

    def _single_harmonic_links(component_params, times, n, length, p_receiver, barycenter):
        frequency = component_params[0]
        coeffs = component_params[1:6] + 1.0j * component_params[6:11]
        direction = component_params[11:14]
        phase0 = component_params[14]
        matrix = _tracefree_matrix_from_coeffs(coeffs)

        n_dot_k = jnp.einsum("lti,i->lt", n, direction)
        nen = jnp.einsum("lti,ij,ltj->lt", n, matrix, n)
        transfer_arg = -2.0j * jnp.pi * frequency * length * (1.0 - n_dot_k)
        transfer = transfer_arg / (1.0 - n_dot_k)
        bary_delay = jnp.einsum("ti,i->t", barycenter, direction)
        carrier = jnp.exp(1.0j * (2.0 * jnp.pi * frequency * (times - bary_delay) + phase0))
        receiver_phase = jnp.exp(-2.0j * jnp.pi * frequency * jnp.einsum("lti,i->lt", p_receiver, direction))
        return 0.5 * nen * carrier[None, :] * receiver_phase * transfer

    def _single_harmonic_links_fastgb_transfer(component_params, times, n, length, p_receiver, barycenter):
        frequency = component_params[0]
        coeffs = component_params[1:6] + 1.0j * component_params[6:11]
        direction = component_params[11:14]
        phase0 = component_params[14]
        matrix = _tracefree_matrix_from_coeffs(coeffs)

        n_dot_k = jnp.einsum("lti,i->lt", n, direction)
        nen = jnp.einsum("lti,ij,ltj->lt", n, matrix, n)
        omega_l = 2.0 * jnp.pi * frequency * length
        transfer_phase = 0.5 * omega_l * (1.0 - n_dot_k)
        transfer = -1.0j * omega_l * jnp.sinc(transfer_phase / jnp.pi) * jnp.exp(-1.0j * transfer_phase)
        bary_delay = jnp.einsum("ti,i->t", barycenter, direction)
        carrier = jnp.exp(1.0j * (2.0 * jnp.pi * frequency * (times - bary_delay) + phase0))
        receiver_phase = jnp.exp(-2.0j * jnp.pi * frequency * jnp.einsum("lti,i->lt", p_receiver, direction))
        return 0.5 * nen * carrier[None, :] * receiver_phase * transfer

    @jax.jit
    def eccentric_complex_strain_batch_jax_array(params, times):
        """Return complex strain shaped `(source, time)` for packed params."""

        return jax.vmap(_single_source_complex_strain, in_axes=(0, None))(params, times)

    @partial(jax.jit, static_argnames=("include_1pn_orbital_corrections", "include_periastron_advance"))
    def eccentric_complex_strain_batch_jax_array_options(
        params,
        times,
        *,
        include_1pn_orbital_corrections: bool,
        include_periastron_advance: bool,
    ):
        """Return source strains with selectable source-physics switches."""

        return jax.vmap(
            lambda p: _single_source_complex_strain_with_options(
                p,
                times,
                include_1pn_orbital_corrections,
                include_periastron_advance,
            )
        )(params)

    @jax.jit
    def eccentric_links_batch_jax_array(params, mask, times, n, length, p_sender, p_receiver):
        """Return summed eccentric one-way links shaped `(6, time)` for a fixed batch."""

        per_source = jax.vmap(_single_source_links, in_axes=(0, None, None, None, None, None))(
            params,
            times,
            n,
            length,
            p_sender,
            p_receiver,
        )
        return jnp.sum(per_source * mask[:, None, None], axis=0)

    @partial(jax.jit, static_argnames=("include_1pn_orbital_corrections", "include_periastron_advance"))
    def eccentric_links_batch_jax_array_options(
        params,
        mask,
        times,
        n,
        length,
        p_sender,
        p_receiver,
        *,
        include_1pn_orbital_corrections: bool,
        include_periastron_advance: bool,
    ):
        """Return summed links with selectable source-physics switches."""

        per_source = jax.vmap(
            lambda p: _single_source_links_with_options(
                p,
                times,
                n,
                length,
                p_sender,
                p_receiver,
                include_1pn_orbital_corrections,
                include_periastron_advance,
            )
        )(params)
        return jnp.sum(per_source * mask[:, None, None], axis=0)

    @partial(jax.jit, static_argnames=("interpolation_order",))
    def eccentric_links_interpolated_batch_jax_array(params, mask, times, n, length, p_sender, p_receiver, interpolation_order: int):
        """Return summed links using one waveform grid plus retarded-time interpolation."""

        per_source = jax.vmap(_single_source_links_interpolated, in_axes=(0, None, None, None, None, None, None))(
            params,
            times,
            n,
            length,
            p_sender,
            p_receiver,
            interpolation_order,
        )
        return jnp.sum(per_source * mask[:, None, None], axis=0)

    @jax.jit
    def harmonic_links_batch_jax_array(params, mask, times, n, length, p_receiver, barycenter):
        """Return summed order-1 beta/triliear harmonic links shaped `(6, time)`."""

        per_component = jax.vmap(_single_harmonic_links, in_axes=(0, None, None, None, None, None))(
            params,
            times,
            n,
            length,
            p_receiver,
            barycenter,
        )
        return jnp.sum(per_component * mask[:, None, None], axis=0)

    @jax.jit
    def harmonic_links_fastgb_transfer_batch_jax_array(params, mask, times, n, length, p_receiver, barycenter):
        """Return summed harmonic links with a sinc/exact finite-arm transfer."""

        per_component = jax.vmap(_single_harmonic_links_fastgb_transfer, in_axes=(0, None, None, None, None, None))(
            params,
            times,
            n,
            length,
            p_receiver,
            barycenter,
        )
        return jnp.sum(per_component * mask[:, None, None], axis=0)

    @jax.jit
    def eccentric_harmonic_component_params_batch_jax_array(params, harmonic_indices, phase_grid, coefficient_scale):
        """Return packed harmonic component rows shaped `(source, harmonic, parameter)`."""

        return jax.vmap(_single_source_harmonic_component_params, in_axes=(0, None, None, None))(
            params,
            harmonic_indices,
            phase_grid,
            coefficient_scale,
        )


def eccentric_complex_strain_batch_jax(
    sources: EccentricBinaryParams | list[EccentricBinaryParams],
    times: NDArray[np.float64],
    *,
    physics_mode: str = "1pn",
) -> NDArray[np.complex128]:
    """Evaluate a batch of eccentric source strains with a JIT-compiled JAX kernel."""

    require_jax()
    params = pack_eccentric_sources(sources)
    include_1pn, include_periastron = _physics_switches(physics_mode)
    if include_1pn and include_periastron:
        result = eccentric_complex_strain_batch_jax_array(jnp.asarray(params), jnp.asarray(times, dtype=jnp.float64))
    else:
        result = eccentric_complex_strain_batch_jax_array_options(
            jnp.asarray(params),
            jnp.asarray(times, dtype=jnp.float64),
            include_1pn_orbital_corrections=include_1pn,
            include_periastron_advance=include_periastron,
        )
    return np.asarray(result)


def eccentric_links_jax(
    sources: EccentricBinaryParams | list[EccentricBinaryParams],
    geometry: dict[str, NDArray[np.float64]],
    *,
    batch_size: int = 1024,
    physics_mode: str = "1pn",
) -> dict[str, NDArray[np.complex128]]:
    """Compute summed brute-force eccentric links with fixed-shape JAX batches."""

    require_jax()
    include_1pn, include_periastron = _physics_switches(physics_mode)
    params = pack_eccentric_response_sources(sources)
    if params.shape[0] == 0:
        raise ValueError("at least one source is required")
    times = jnp.asarray(geometry["times"], dtype=jnp.float64)
    n = jnp.asarray(geometry["n"], dtype=jnp.float64)
    length = jnp.asarray(geometry["length"], dtype=jnp.float64)
    p_sender = jnp.asarray(geometry["p_sender"], dtype=jnp.float64)
    p_receiver = jnp.asarray(geometry["p_receiver"], dtype=jnp.float64)

    total = jnp.zeros((len(JAX_LINK_LABELS), np.asarray(geometry["times"]).size), dtype=jnp.complex128)
    for start in range(0, params.shape[0], batch_size):
        chunk = params[start : start + batch_size]
        padded, mask = pad_packed_sources(chunk, batch_size)
        if include_1pn and include_periastron:
            total = total + eccentric_links_batch_jax_array(
                jnp.asarray(padded),
                jnp.asarray(mask),
                times,
                n,
                length,
                p_sender,
                p_receiver,
            )
        else:
            total = total + eccentric_links_batch_jax_array_options(
                jnp.asarray(padded),
                jnp.asarray(mask),
                times,
                n,
                length,
                p_sender,
                p_receiver,
                include_1pn_orbital_corrections=include_1pn,
                include_periastron_advance=include_periastron,
            )
    out = np.asarray(total)
    return {label: out[index] for index, label in enumerate(JAX_LINK_LABELS)}


def eccentric_links_interpolated_jax(
    sources: EccentricBinaryParams | list[EccentricBinaryParams],
    geometry: dict[str, NDArray[np.float64]],
    *,
    batch_size: int = 1024,
    interpolation_order: int = 3,
) -> dict[str, NDArray[np.complex128]]:
    """Compute eccentric links using grid waveforms interpolated to retarded times."""

    require_jax()
    if interpolation_order not in {1, 3, 5, 7}:
        raise ValueError("interpolation_order must be one of 1, 3, 5, or 7")
    if interpolation_order > 2 * INTERPOLATION_HALO_SAMPLES - 1:
        raise ValueError("interpolation_order is larger than the interpolation halo supports")
    params = pack_eccentric_response_sources(sources)
    if params.shape[0] == 0:
        raise ValueError("at least one source is required")
    times = jnp.asarray(geometry["times"], dtype=jnp.float64)
    n = jnp.asarray(geometry["n"], dtype=jnp.float64)
    length = jnp.asarray(geometry["length"], dtype=jnp.float64)
    p_sender = jnp.asarray(geometry["p_sender"], dtype=jnp.float64)
    p_receiver = jnp.asarray(geometry["p_receiver"], dtype=jnp.float64)

    total = jnp.zeros((len(JAX_LINK_LABELS), np.asarray(geometry["times"]).size), dtype=jnp.complex128)
    for start in range(0, params.shape[0], batch_size):
        chunk = params[start : start + batch_size]
        padded, mask = pad_packed_sources(chunk, batch_size)
        total = total + eccentric_links_interpolated_batch_jax_array(
            jnp.asarray(padded),
            jnp.asarray(mask),
            times,
            n,
            length,
            p_sender,
            p_receiver,
            interpolation_order,
        )
    out = np.asarray(total)
    return {label: out[index] for index, label in enumerate(JAX_LINK_LABELS)}


def harmonic_links_jax(
    components: EccentricHarmonicComponent | list[EccentricHarmonicComponent],
    geometry: dict[str, NDArray[np.float64]],
    *,
    batch_size: int = 64,
    response_mode: str = "triliear_beta",
    barycentric_phase: bool = True,
) -> dict[str, NDArray[np.complex128]]:
    """Compute harmonic links from packed carriers.

    `response_mode="triliear_beta"` uses the first-order beta transfer.
    `response_mode="fastgb_transfer"` uses the sinc form of the exact
    monochromatic finite-arm transfer, inspired by FastGB's slow response.
    """

    require_jax()
    if response_mode not in {"triliear_beta", "fastgb_transfer"}:
        raise ValueError("response_mode must be 'triliear_beta' or 'fastgb_transfer'")
    params = pack_harmonic_components(components)
    return harmonic_links_from_packed_jax(
        params,
        geometry,
        batch_size=batch_size,
        response_mode=response_mode,
        barycentric_phase=barycentric_phase,
    )


def harmonic_links_from_packed_jax(
    params: NDArray[np.float64],
    geometry: dict[str, NDArray[np.float64]],
    *,
    batch_size: int = 64,
    response_mode: str = "triliear_beta",
    barycentric_phase: bool = True,
) -> dict[str, NDArray[np.complex128]]:
    """Compute harmonic links from packed component rows."""

    require_jax()
    if response_mode not in {"triliear_beta", "fastgb_transfer"}:
        raise ValueError("response_mode must be 'triliear_beta' or 'fastgb_transfer'")
    params = np.asarray(params, dtype=np.float64)
    if params.shape[0] == 0:
        raise ValueError("at least one harmonic component is required")
    times = jnp.asarray(geometry["times"], dtype=jnp.float64)
    n = jnp.asarray(geometry["n"], dtype=jnp.float64)
    length = jnp.asarray(geometry["length"], dtype=jnp.float64)
    p_receiver = jnp.asarray(geometry["p_receiver"], dtype=jnp.float64)
    if barycentric_phase and geometry.get("position_origin", "ssb") == "barycenter":
        barycenter_array = geometry["barycenter"]
    else:
        barycenter_array = np.zeros_like(geometry["barycenter"])
    barycenter = jnp.asarray(barycenter_array, dtype=jnp.float64)

    total = jnp.zeros((len(JAX_LINK_LABELS), np.asarray(geometry["times"]).size), dtype=jnp.complex128)
    for start in range(0, params.shape[0], batch_size):
        chunk = params[start : start + batch_size]
        padded, mask = pad_packed_sources(chunk, batch_size)
        if response_mode == "fastgb_transfer":
            total = total + harmonic_links_fastgb_transfer_batch_jax_array(
                jnp.asarray(padded),
                jnp.asarray(mask),
                times,
                n,
                length,
                p_receiver,
                barycenter,
            )
        else:
            total = total + harmonic_links_batch_jax_array(
                jnp.asarray(padded),
                jnp.asarray(mask),
                times,
                n,
                length,
                p_receiver,
                barycenter,
            )
    out = np.asarray(total)
    return {label: out[index] for index, label in enumerate(JAX_LINK_LABELS)}


def packed_eccentric_harmonic_components_jax(
    sources: EccentricBinaryParams | list[EccentricBinaryParams],
    *,
    order: int = 1,
    samples: int = 512,
    include_negative_frequencies: bool = True,
    source_batch_size: int | None = None,
) -> NDArray[np.float64]:
    """Return packed eccentric harmonic component rows using batched JAX FFTs."""

    require_jax()
    if samples < 32:
        raise ValueError("samples must be at least 32")
    source_params = pack_eccentric_response_sources(sources)
    if source_params.shape[0] == 0:
        raise ValueError("at least one source is required")
    if source_batch_size is not None and source_batch_size <= 0:
        raise ValueError("source_batch_size must be positive")
    harmonic_indices = np.asarray(eccentric_harmonic_indices(order), dtype=np.int32)
    phase_grid = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False, dtype=np.float64)
    coefficient_scale = 1.0 if include_negative_frequencies else 2.0

    def add_negative_components(positive_rows: NDArray[np.float64]) -> NDArray[np.float64]:
        if not include_negative_frequencies:
            return positive_rows
        negative_rows = positive_rows.copy()
        negative_rows[:, 0] *= -1.0
        negative_rows[:, 6:11] *= -1.0
        interleaved_rows = np.empty((positive_rows.shape[0] * 2, positive_rows.shape[1]), dtype=np.float64)
        interleaved_rows[0::2] = positive_rows
        interleaved_rows[1::2] = negative_rows
        return interleaved_rows

    def compute_positive(params: NDArray[np.float64]) -> NDArray[np.float64]:
        positive = eccentric_harmonic_component_params_batch_jax_array(
            jnp.asarray(params),
            jnp.asarray(harmonic_indices),
            jnp.asarray(phase_grid),
            jnp.asarray(coefficient_scale, dtype=jnp.float64),
        )
        return np.asarray(positive)

    if source_batch_size is None:
        positive_np = compute_positive(source_params).reshape((-1, 15))
        return add_negative_components(positive_np)

    positive_chunks = []
    for start in range(0, source_params.shape[0], source_batch_size):
        chunk = source_params[start : start + source_batch_size]
        padded, _ = pad_packed_sources(chunk, source_batch_size)
        positive_chunk = compute_positive(padded)[: chunk.shape[0]]
        positive_chunks.append(positive_chunk.reshape((-1, 15)))
    positive_np = np.concatenate(positive_chunks, axis=0)
    return add_negative_components(positive_np)


def eccentric_harmonic_links_jax(
    sources: EccentricBinaryParams | list[EccentricBinaryParams],
    geometry: dict[str, NDArray[np.float64]],
    *,
    eccentric_order: int = 1,
    harmonic_samples: int = 512,
    batch_size: int = 64,
    response_mode: str = "triliear_beta",
    barycentric_phase: bool = True,
    include_negative_frequencies: bool = True,
    component_backend: str = "jax",
    component_source_batch_size: int | None = None,
) -> dict[str, NDArray[np.complex128]]:
    """Compute experimental eccentric triliear links with a batched JAX kernel."""

    if component_backend not in {"jax", "python"}:
        raise ValueError("component_backend must be 'jax' or 'python'")
    if component_backend == "jax":
        component_params = packed_eccentric_harmonic_components_jax(
            sources,
            order=eccentric_order,
            samples=harmonic_samples,
            include_negative_frequencies=include_negative_frequencies,
            source_batch_size=component_source_batch_size,
        )
        return harmonic_links_from_packed_jax(
            component_params,
            geometry,
            batch_size=batch_size,
            response_mode=response_mode,
            barycentric_phase=barycentric_phase,
        )

    components = eccentric_harmonic_source_batch(
        sources,
        order=eccentric_order,
        samples=harmonic_samples,
        include_negative_frequencies=include_negative_frequencies,
    )
    return harmonic_links_jax(
        components,
        geometry,
        batch_size=batch_size,
        response_mode=response_mode,
        barycentric_phase=barycentric_phase,
    )
