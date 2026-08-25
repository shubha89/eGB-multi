"""Eccentric compact-binary source waveforms.

This module implements time-domain quadrupole polarizations evaluated on
Newtonian or 1PN quasi-Keplerian eccentric orbital dynamics following Tessmer
and Gopakumar
(gr-qc/0610139). The expensive LISA response is intentionally kept separate:
these functions only produce the source polarizations that can later be fed
through the existing triliear/pyTDI response path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from .constants import C_M_PER_S, G_M3_KG_S2, KILOPARSEC_M, SOLAR_MASS_KG
from .tensors import matrix_to_tracefree5, normalize_direction


@dataclass(frozen=True)
class EccentricBinaryParams:
    """Eccentric compact-binary source parameters.

    `mean_motion` is the source carrier angular frequency in rad/s. It equals
    the radial mean motion `n` in the Newtonian model and is interpreted as the
    averaged azimuthal frequency `omega` when PN periastron advance is enabled.
    The dominant nearly circular gravitational-wave frequency is approximately
    `mean_motion / pi`.
    """

    mean_motion: float
    eccentricity: float
    m1_solar: float = 0.5
    m2_solar: float = 0.5
    distance_m: float = KILOPARSEC_M
    inclination: float = 0.0
    beta: float = 0.0
    lambda_: float = 0.0
    psi: float = 0.0
    phi0: float = 0.0
    t0: float = 0.0
    fdot: float = 0.0

    @property
    def total_mass_kg(self) -> float:
        return (self.m1_solar + self.m2_solar) * SOLAR_MASS_KG

    @property
    def symmetric_mass_ratio(self) -> float:
        total = self.m1_solar + self.m2_solar
        return self.m1_solar * self.m2_solar / total**2

    def propagation_direction(self) -> NDArray[np.float64]:
        cos_beta = np.cos(self.beta)
        return normalize_direction(
            [
                -cos_beta * np.cos(self.lambda_),
                -cos_beta * np.sin(self.lambda_),
                -np.sin(self.beta),
            ]
        )

    def polarization_basis(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        theta = np.pi / 2.0 - self.beta
        sinth, costh = np.sin(theta), np.cos(theta)
        sinph, cosph = np.sin(self.lambda_), np.cos(self.lambda_)

        u = np.array([costh * cosph, costh * sinph, -sinth], dtype=np.float64)
        v = np.array([sinph, -cosph, 0.0], dtype=np.float64)
        eplus = np.outer(v, v) - np.outer(u, u)
        ecross = np.outer(u, v) + np.outer(v, u)
        return eplus, ecross


@dataclass(frozen=True)
class EccentricHarmonicComponent:
    """A single carrier component for the eccentric triliear response."""

    frequency: float
    tracefree_coefficients: NDArray[np.complex128]
    propagation_direction: NDArray[np.float64]
    harmonic: int
    source_index: int = 0
    phase0: float = 0.0


@dataclass(frozen=True)
class EccentricPhysicsOptions:
    """Switches for eccentric source physics terms.

    `include_1pn_orbital_corrections` controls the explicit 1PN corrections to
    r, rdot, e_phi, and phidot. `include_periastron_advance` controls the 1PN
    periastron advance factor in the orbital phase. The default matches the
    model previously implemented in this package.
    """

    include_1pn_orbital_corrections: bool = True
    include_periastron_advance: bool = True


@dataclass(frozen=True)
class PetersMathewsSwitchPoint:
    """One calibration point for the automatic Peters--Mathews switch."""

    f0_hz: float
    eccentricity: float
    total_mass_solar: float
    symmetric_mass_ratio: float
    duration_s: float
    mismatch: float


@dataclass(frozen=True)
class PetersMathewsSwitchRule:
    """Empirical rule for deciding when Peters--Mathews evolution is needed.

    The rule stores fixed-1PN-vs-full-PM source-level mismatches over a sparse
    calibration grid. Predictions use inverse-distance interpolation in
    ``log(f0)``, eccentricity, ``log(total_mass)``, symmetric mass ratio, and
    ``log(T_obs)``.
    """

    points: tuple[PetersMathewsSwitchPoint, ...]
    mismatch_tolerance: float = 1.0e-2
    neighbors: int = 4

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("PetersMathewsSwitchRule requires at least one calibration point")
        _validate_mismatch_tolerance(self.mismatch_tolerance)
        if self.neighbors < 1:
            raise ValueError("neighbors must be positive")
        for point in self.points:
            _validate_switch_point(point)

    def predict_mismatch(self, source: EccentricBinaryParams, duration_s: float) -> float:
        """Predict fixed-1PN-vs-full-PM mismatch for ``source`` and duration."""

        target = _switch_features_from_source(source, duration_s)
        features = np.array([_switch_features(point) for point in self.points], dtype=np.float64)
        scales = np.ptp(features, axis=0)
        scales = np.where(scales > 0.0, scales, 1.0)
        distances = np.linalg.norm((features - target) / scales, axis=1)
        exact = np.nonzero(distances < 1.0e-12)[0]
        if exact.size:
            return float(self.points[int(exact[0])].mismatch)

        count = min(self.neighbors, len(self.points))
        indices = np.argsort(distances)[:count]
        weights = 1.0 / np.maximum(distances[indices], 1.0e-12) ** 2
        mismatches = np.array([self.points[int(index)].mismatch for index in indices], dtype=np.float64)
        return float(np.sum(weights * mismatches) / np.sum(weights))

    def evolution_mode_for(
        self,
        source: EccentricBinaryParams,
        duration_s: float,
        *,
        mismatch_tolerance: float | None = None,
    ) -> str:
        """Return ``fixed`` or ``peters_mathews`` from the calibrated rule."""

        tolerance = self.mismatch_tolerance if mismatch_tolerance is None else mismatch_tolerance
        _validate_mismatch_tolerance(tolerance)
        predicted = self.predict_mismatch(source, duration_s)
        return "peters_mathews" if predicted >= tolerance else "fixed"


ECCENTRIC_PHYSICS_MODES: dict[str, EccentricPhysicsOptions] = {
    "newtonian": EccentricPhysicsOptions(False, False),
    "1pn_no_periastron": EccentricPhysicsOptions(True, False),
    "1pn": EccentricPhysicsOptions(True, True),
    "1pn_periastron": EccentricPhysicsOptions(True, True),
}

ECCENTRIC_EVOLUTION_MODES = (
    "fixed",
    "peters_mathews",
    "peters_mathews_orbital_only",
    "peters_mathews_eccentricity_only",
    "auto",
)

ECCENTRIC_EVOLUTION_MODE_ALIASES: dict[str, str] = {
    "pm": "peters_mathews",
    "peters-mathews": "peters_mathews",
    "adaptive": "auto",
    "pm_auto": "auto",
}

ECCENTRIC_PHYSICS_MODE_ALIASES: dict[str, str] = {
    "1pn": "1pn",
    "1pn_periastron": "1pn",
    "1pn+periastron": "1pn",
    "1pn_with_periastron": "1pn",
    "1pn_no_precession": "1pn_no_periastron",
}


def eccentric_physics_mode_label(mode: str | EccentricPhysicsOptions = "1pn") -> str:
    """Return the canonical physics-mode label for eccentric source terms."""

    if isinstance(mode, EccentricPhysicsOptions):
        if mode == EccentricPhysicsOptions(False, False):
            return "newtonian"
        if mode == EccentricPhysicsOptions(True, False):
            return "1pn_no_periastron"
        if mode == EccentricPhysicsOptions(True, True):
            return "1pn"
        raise ValueError("custom eccentric physics switches do not have a canonical mode label")
    return ECCENTRIC_PHYSICS_MODE_ALIASES.get(mode, mode)


def eccentric_physics_options(mode: str | EccentricPhysicsOptions = "1pn") -> EccentricPhysicsOptions:
    """Return normalized eccentric physics switches."""

    if isinstance(mode, EccentricPhysicsOptions):
        return mode
    mode = eccentric_physics_mode_label(mode)
    try:
        return ECCENTRIC_PHYSICS_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"unknown eccentric physics mode {mode!r}") from exc


def eccentric_evolution_mode_label(evolution_mode: str) -> str:
    """Return the canonical evolution-mode label for eccentric source parameters."""

    return ECCENTRIC_EVOLUTION_MODE_ALIASES.get(evolution_mode, evolution_mode)


def _validate_evolution_mode(evolution_mode: str) -> str:
    evolution_mode = eccentric_evolution_mode_label(evolution_mode)
    if evolution_mode not in ECCENTRIC_EVOLUTION_MODES:
        raise ValueError(
            "evolution_mode must be one of 'fixed', 'peters_mathews', "
            "'peters_mathews_orbital_only', 'peters_mathews_eccentricity_only', or 'auto'"
        )
    return evolution_mode


def _validate_mismatch_tolerance(tolerance: float) -> None:
    if not np.isfinite(tolerance) or tolerance < 0.0 or tolerance >= 1.0:
        raise ValueError("pm_mismatch_tolerance must satisfy 0 <= tolerance < 1")


def _validate_switch_point(point: PetersMathewsSwitchPoint) -> None:
    if point.f0_hz <= 0.0:
        raise ValueError("switch-rule f0_hz must be positive")
    _validate_eccentricity(point.eccentricity)
    if point.total_mass_solar <= 0.0:
        raise ValueError("switch-rule total_mass_solar must be positive")
    if not 0.0 < point.symmetric_mass_ratio <= 0.25:
        raise ValueError("switch-rule symmetric_mass_ratio must satisfy 0 < eta <= 0.25")
    if point.duration_s <= 0.0:
        raise ValueError("switch-rule duration_s must be positive")
    if not np.isfinite(point.mismatch) or point.mismatch < 0.0 or point.mismatch >= 1.0:
        raise ValueError("switch-rule mismatch must satisfy 0 <= mismatch < 1")


def _switch_features(point: PetersMathewsSwitchPoint) -> NDArray[np.float64]:
    return np.array(
        [
            np.log(point.f0_hz),
            point.eccentricity,
            np.log(point.total_mass_solar),
            point.symmetric_mass_ratio,
            np.log(point.duration_s),
        ],
        dtype=np.float64,
    )


def _switch_features_from_source(source: EccentricBinaryParams, duration_s: float) -> NDArray[np.float64]:
    if source.mean_motion <= 0.0:
        raise ValueError("mean_motion must be positive")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    total_mass_solar = source.m1_solar + source.m2_solar
    if total_mass_solar <= 0.0:
        raise ValueError("total source mass must be positive")
    return np.array(
        [
            np.log(source.mean_motion / np.pi),
            source.eccentricity,
            np.log(total_mass_solar),
            source.symmetric_mass_ratio,
            np.log(duration_s),
        ],
        dtype=np.float64,
    )


def _duration_from_times(t: ArrayLike) -> float:
    tt = np.asarray(t, dtype=np.float64)
    if tt.size < 2:
        raise ValueError("automatic Peters-Mathews switching requires at least two time samples")
    return float(np.max(tt) - np.min(tt))


def _validate_eccentricity(eccentricity: float | NDArray[np.float64]) -> None:
    e = np.asarray(eccentricity, dtype=np.float64)
    if np.any(~np.isfinite(e)) or np.any(e < 0.0) or np.any(e >= 1.0):
        raise ValueError("eccentricity must be finite and satisfy 0 <= e < 1")


def _validate_peters_mathews_state(state: NDArray[np.float64]) -> None:
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("Peters-Mathews state must have shape (3,)")
    if np.any(~np.isfinite(values)):
        raise ValueError("Peters-Mathews state must be finite")
    if values[0] <= 0.0:
        raise ValueError("Peters-Mathews mean_motion state must be positive")
    _validate_eccentricity(values[1])


def _pn_x_from_azimuthal_frequency(
    gm: float,
    azimuthal_frequency: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    return (gm * np.asarray(azimuthal_frequency, dtype=np.float64) / C_M_PER_S**3) ** (2.0 / 3.0)


def _periastron_advance_1pn(x: float | NDArray[np.float64], eccentricity: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    one_minus_e2 = 1.0 - np.asarray(eccentricity, dtype=np.float64) ** 2
    return 3.0 * np.asarray(x, dtype=np.float64) / one_minus_e2


def _radial_from_azimuthal_frequency_1pn(
    azimuthal_frequency: float | NDArray[np.float64],
    x: float | NDArray[np.float64],
    eccentricity: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    return np.asarray(azimuthal_frequency, dtype=np.float64) * (1.0 - _periastron_advance_1pn(x, eccentricity))


def _azimuthal_from_radial_frequency_1pn(
    gm: float,
    radial_frequency: float | NDArray[np.float64],
    eccentricity: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    # Using x(n) here is equivalent to the omega-based relation through 1PN.
    x_radial = _pn_x_from_azimuthal_frequency(gm, radial_frequency)
    return np.asarray(radial_frequency, dtype=np.float64) * (1.0 + _periastron_advance_1pn(x_radial, eccentricity))


def _integrate_rate_from_t0(
    t: NDArray[np.float64],
    rate: float | NDArray[np.float64],
    *,
    t0: float,
    rate_at_t0: float,
) -> NDArray[np.float64]:
    tt = np.asarray(t, dtype=np.float64)
    rr = np.broadcast_to(np.asarray(rate, dtype=np.float64), tt.shape)
    flat_t = tt.reshape(-1)
    flat_r = rr.reshape(-1)
    out = np.empty_like(flat_t)

    def integrate_order(indices: NDArray[np.int64], *, reverse: bool) -> None:
        if indices.size == 0:
            return
        order = indices[np.argsort(flat_t[indices])]
        if reverse:
            order = order[::-1]
        ordered_t = flat_t[order]
        ordered_r = flat_r[order]
        step_t = np.concatenate(([t0], ordered_t))
        step_r = np.concatenate(([rate_at_t0], ordered_r))
        increments = 0.5 * (step_r[1:] + step_r[:-1]) * (step_t[1:] - step_t[:-1])
        out[order] = np.cumsum(increments)

    integrate_order(np.nonzero(flat_t >= t0)[0], reverse=False)
    integrate_order(np.nonzero(flat_t < t0)[0], reverse=True)
    return out.reshape(tt.shape)


def _safe_beta_phi_1pn(
    eta: float,
    eccentricity: float | NDArray[np.float64],
    x: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    e = np.asarray(eccentricity, dtype=np.float64)
    safe_e = np.where(np.abs(e) > 1.0e-14, e, 1.0)
    sqrt_one_minus_e2 = np.sqrt(np.maximum(1.0 - e * e, 0.0))
    beta_newtonian = (1.0 - sqrt_one_minus_e2) / safe_e
    beta_1pn = (
        -4.0
        + eta
        + e * e * (8.0 - 2.0 * eta)
        + (4.0 - eta) * sqrt_one_minus_e2
    ) / (safe_e * sqrt_one_minus_e2)
    beta = beta_newtonian + np.asarray(x, dtype=np.float64) * beta_1pn
    return np.where(np.abs(e) > 1.0e-14, beta, 0.0)


def _v_minus_u_from_beta(
    u: NDArray[np.float64],
    beta_phi: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    v_minus_u = 2.0 * np.arctan2(beta_phi * np.sin(u), 1.0 - beta_phi * np.cos(u))
    return np.unwrap(v_minus_u, axis=-1) if np.ndim(v_minus_u) > 0 else v_minus_u


def mikkola_kepler(mean_anomaly: ArrayLike, eccentricity: float) -> NDArray[np.float64]:
    """Solve `u - e sin(u) = mean_anomaly` with Mikkola's starter.

    The starter is followed by one fourth-order Danby correction. The function
    is vectorized over the mean anomaly array and returns a continuous eccentric
    anomaly with the same 2-pi branch as the input mean anomaly.
    """

    _validate_eccentricity(eccentricity)
    mean_anomaly = np.asarray(mean_anomaly, dtype=np.float64)
    e = np.asarray(eccentricity, dtype=np.float64)
    if np.all(e == 0.0):
        return mean_anomaly.copy()

    wrapped = (mean_anomaly + np.pi) % (2.0 * np.pi) - np.pi
    offset = mean_anomaly - wrapped

    alpha = (1.0 - e) / (4.0 * e + 0.5)
    beta = 0.5 * wrapped / (4.0 * e + 0.5)
    root = np.sqrt(beta * beta + alpha**3)
    z = np.cbrt(np.where(wrapped >= 0.0, beta + root, beta - root))
    s = z - alpha / z
    omega = s - 0.078 * s**5 / (1.0 + e)
    u = wrapped + e * (3.0 * omega - 4.0 * omega**3)

    sin_u = np.sin(u)
    cos_u = np.cos(u)
    f = u - e * sin_u - wrapped
    f1 = 1.0 - e * cos_u
    f2 = e * sin_u
    f3 = e * cos_u
    f4 = -e * sin_u
    du1 = -f / f1
    du2 = -f / (f1 + 0.5 * f2 * du1)
    du3 = -f / (f1 + 0.5 * f2 * du2 + f3 * du2**2 / 6.0)
    du4 = -f / (f1 + 0.5 * f2 * du3 + f3 * du3**2 / 6.0 + f4 * du3**3 / 24.0)
    solved = u + du4 + offset
    return np.where(e == 0.0, mean_anomaly, solved)


def peters_mathews_derivatives(
    mean_motion: ArrayLike,
    eccentricity: ArrayLike,
    total_mass_kg: float,
    symmetric_mass_ratio: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return leading radiation-reaction derivatives `(dn/dt, de/dt)`.

    This is the Peters-Mathews quadrupole-order secular evolution, expressed
    using the radial mean motion `n` instead of the semi-major axis.
    """

    n = np.asarray(mean_motion, dtype=np.float64)
    e = np.asarray(eccentricity, dtype=np.float64)
    _validate_eccentricity(e)
    if np.any(n <= 0.0):
        raise ValueError("mean_motion must be positive")

    one_minus_e2 = 1.0 - e * e
    mass_factor = symmetric_mass_ratio * (G_M3_KG_S2 * total_mass_kg) ** (5.0 / 3.0) / C_M_PER_S**5
    eccentric_frequency_boost = (1.0 + 73.0 * e * e / 24.0 + 37.0 * e**4 / 96.0) / one_minus_e2 ** (7.0 / 2.0)
    eccentric_decay_boost = e * (1.0 + 121.0 * e * e / 304.0) / one_minus_e2 ** (5.0 / 2.0)
    ndot = (96.0 / 5.0) * mass_factor * n ** (11.0 / 3.0) * eccentric_frequency_boost
    edot = -(304.0 / 15.0) * mass_factor * n ** (8.0 / 3.0) * eccentric_decay_boost
    return ndot, edot


def _peters_mathews_state_derivative(
    state: NDArray[np.float64],
    total_mass_kg: float,
    symmetric_mass_ratio: float,
) -> NDArray[np.float64]:
    _validate_peters_mathews_state(state)
    n = state[0]
    e = state[1]
    ndot, edot = peters_mathews_derivatives(n, e, total_mass_kg, symmetric_mass_ratio)
    return np.array([float(ndot), float(edot), n], dtype=np.float64)


def _rk4_peters_mathews_step(
    state: NDArray[np.float64],
    dt: float,
    total_mass_kg: float,
    symmetric_mass_ratio: float,
) -> NDArray[np.float64]:
    k1 = _peters_mathews_state_derivative(state, total_mass_kg, symmetric_mass_ratio)
    k2 = _peters_mathews_state_derivative(state + 0.5 * dt * k1, total_mass_kg, symmetric_mass_ratio)
    k3 = _peters_mathews_state_derivative(state + 0.5 * dt * k2, total_mass_kg, symmetric_mass_ratio)
    k4 = _peters_mathews_state_derivative(state + dt * k3, total_mass_kg, symmetric_mass_ratio)
    next_state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    _validate_peters_mathews_state(next_state)
    return next_state


def peters_mathews_evolution(
    source: EccentricBinaryParams,
    t: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """Integrate Peters-Mathews secular evolution from the source values at `t0`.

    Returns arrays for `mean_motion`, `eccentricity`, and mean anomaly `l`.
    The integration follows the sorted requested times, including support for
    times before `t0` through negative RK4 steps.
    """

    _validate_eccentricity(source.eccentricity)
    if source.mean_motion <= 0.0:
        raise ValueError("mean_motion must be positive")
    tt = np.asarray(t, dtype=np.float64)
    flat_times = tt.reshape(-1)
    n_out = np.empty_like(flat_times)
    e_out = np.empty_like(flat_times)
    l_out = np.empty_like(flat_times)

    base_state = np.array([source.mean_motion, source.eccentricity, 0.0], dtype=np.float64)

    def integrate_indices(indices: NDArray[np.int64]) -> None:
        if indices.size == 0:
            return
        order = indices[np.argsort(flat_times[indices])]
        eval_times = flat_times[order]
        if np.all(eval_times == source.t0):
            values = np.repeat(base_state[:, None], eval_times.size, axis=1)
        else:
            tau = np.abs(eval_times - source.t0)
            unique_tau, inverse = np.unique(tau, return_inverse=True)
            direction = 1.0 if eval_times[-1] >= source.t0 else -1.0

            def rhs(_tau: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
                return direction * _peters_mathews_state_derivative(state, source.total_mass_kg, source.symmetric_mass_ratio)

            solution = solve_ivp(
                rhs,
                (0.0, float(unique_tau[-1])),
                base_state,
                t_eval=unique_tau,
                rtol=1.0e-10,
                atol=[1.0e-18, 1.0e-14, 1.0e-8],
            )
            if not solution.success:
                raise RuntimeError(f"Peters-Mathews evolution failed: {solution.message}")
            values = solution.y[:, inverse]
        try:
            _validate_eccentricity(values[1])
        except ValueError as exc:
            raise RuntimeError("Peters-Mathews evolution left the physical eccentricity domain") from exc
        if np.any(~np.isfinite(values[0])) or np.any(values[0] <= 0.0):
            raise RuntimeError("Peters-Mathews evolution left the physical mean-motion domain")
        n_out[order] = values[0]
        e_out[order] = values[1]
        l_out[order] = values[2]

    forward = np.nonzero(flat_times >= source.t0)[0]
    backward = np.nonzero(flat_times < source.t0)[0]
    integrate_indices(forward)
    integrate_indices(backward)
    return {
        "mean_motion": n_out.reshape(tt.shape),
        "eccentricity": e_out.reshape(tt.shape),
        "mean_anomaly": l_out.reshape(tt.shape),
    }


def complex_strain_fitting_factor(
    reference: ArrayLike,
    candidate: ArrayLike,
) -> float:
    """Return the normalized complex overlap between two source strains."""

    ref = np.asarray(reference, dtype=np.complex128)
    cand = np.asarray(candidate, dtype=np.complex128)
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate strains must have the same shape")
    rr = float(np.vdot(ref, ref).real)
    cc = float(np.vdot(cand, cand).real)
    if rr <= 0.0 or cc <= 0.0:
        return 0.0
    ff = float(abs(np.vdot(ref, cand)) / np.sqrt(rr * cc))
    return min(ff, 1.0)


def complex_strain_mismatch(
    reference: ArrayLike,
    candidate: ArrayLike,
) -> float:
    """Return ``1 - FF`` for two source-level complex strains."""

    return max(1.0 - complex_strain_fitting_factor(reference, candidate), 0.0)


def peters_mathews_source_mismatch(
    source: EccentricBinaryParams,
    t: ArrayLike,
    *,
    physics_mode: str | EccentricPhysicsOptions = "1pn",
) -> float:
    """Compare fixed 1PN source strain to full Peters--Mathews evolution.

    This is the quantity used by ``evolution_mode="auto"`` when no calibrated
    rule is provided.
    """

    fixed = eccentric_complex_strain(source, t, physics_mode=physics_mode, evolution_mode="fixed")
    full_pm = eccentric_complex_strain(source, t, physics_mode=physics_mode, evolution_mode="peters_mathews")
    return complex_strain_mismatch(full_pm, fixed)


def calibrate_peters_mathews_switch_rule(
    sources: EccentricBinaryParams | list[EccentricBinaryParams],
    t: ArrayLike,
    *,
    mismatch_tolerance: float = 1.0e-2,
    physics_mode: str | EccentricPhysicsOptions = "1pn",
    neighbors: int = 4,
) -> PetersMathewsSwitchRule:
    """Build a tolerance rule from fixed-1PN-vs-full-PM source mismatches."""

    _validate_mismatch_tolerance(mismatch_tolerance)
    source_list = [sources] if isinstance(sources, EccentricBinaryParams) else list(sources)
    if not source_list:
        raise ValueError("at least one source is required")
    duration_s = _duration_from_times(t)
    points = []
    for source in source_list:
        mismatch = peters_mathews_source_mismatch(source, t, physics_mode=physics_mode)
        points.append(
            PetersMathewsSwitchPoint(
                f0_hz=float(source.mean_motion / np.pi),
                eccentricity=float(source.eccentricity),
                total_mass_solar=float(source.m1_solar + source.m2_solar),
                symmetric_mass_ratio=float(source.symmetric_mass_ratio),
                duration_s=duration_s,
                mismatch=mismatch,
            )
        )
    return PetersMathewsSwitchRule(tuple(points), mismatch_tolerance=mismatch_tolerance, neighbors=neighbors)


def select_peters_mathews_evolution_mode(
    source: EccentricBinaryParams,
    t: ArrayLike,
    *,
    physics_mode: str | EccentricPhysicsOptions = "1pn",
    pm_mismatch_tolerance: float = 1.0e-2,
    pm_switch_rule: PetersMathewsSwitchRule | None = None,
) -> str:
    """Return ``fixed`` or ``peters_mathews`` for a requested PM tolerance."""

    _validate_mismatch_tolerance(pm_mismatch_tolerance)
    duration_s = _duration_from_times(t)
    if pm_switch_rule is not None:
        return pm_switch_rule.evolution_mode_for(
            source,
            duration_s,
            mismatch_tolerance=pm_mismatch_tolerance,
        )
    mismatch = peters_mathews_source_mismatch(source, t, physics_mode=physics_mode)
    return "peters_mathews" if mismatch >= pm_mismatch_tolerance else "fixed"


def eccentric_dynamics(
    source: EccentricBinaryParams,
    t: ArrayLike,
    *,
    physics_mode: str | EccentricPhysicsOptions = "1pn",
    evolution_mode: str = "fixed",
    pm_mismatch_tolerance: float = 1.0e-2,
    pm_switch_rule: PetersMathewsSwitchRule | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Return the 1PN quasi-Keplerian orbital variables."""

    _validate_eccentricity(source.eccentricity)
    evolution_mode = _validate_evolution_mode(evolution_mode)
    if source.mean_motion <= 0.0:
        raise ValueError("mean_motion must be positive")
    if source.distance_m <= 0.0:
        raise ValueError("distance_m must be positive")

    tt = np.asarray(t, dtype=np.float64)
    if evolution_mode == "auto":
        evolution_mode = select_peters_mathews_evolution_mode(
            source,
            tt,
            physics_mode=physics_mode,
            pm_mismatch_tolerance=pm_mismatch_tolerance,
            pm_switch_rule=pm_switch_rule,
        )
    options = eccentric_physics_options(physics_mode)
    m = source.total_mass_kg
    eta = source.symmetric_mass_ratio
    gm = G_M3_KG_S2 * m
    omega0 = source.mean_motion
    x0 = _pn_x_from_azimuthal_frequency(gm, omega0)
    radial_n0 = (
        float(_radial_from_azimuthal_frequency_1pn(omega0, x0, source.eccentricity))
        if options.include_periastron_advance
        else omega0
    )

    if evolution_mode in {"peters_mathews", "peters_mathews_orbital_only", "peters_mathews_eccentricity_only"}:
        evolution_source = replace(source, mean_motion=radial_n0)
        evolution = peters_mathews_evolution(evolution_source, tt)
        if evolution_mode == "peters_mathews":
            e = evolution["eccentricity"]
            n = evolution["mean_motion"]
            mean_anomaly = evolution["mean_anomaly"]
        elif evolution_mode == "peters_mathews_orbital_only":
            e = source.eccentricity
            n = evolution["mean_motion"]
            mean_anomaly = evolution["mean_anomaly"]
        else:
            e = evolution["eccentricity"]
            omega = np.broadcast_to(omega0, tt.shape)
            x_for_radial = _pn_x_from_azimuthal_frequency(gm, omega)
            n = (
                _radial_from_azimuthal_frequency_1pn(omega, x_for_radial, e)
                if options.include_periastron_advance
                else omega
            )
            mean_anomaly = _integrate_rate_from_t0(tt, n, t0=source.t0, rate_at_t0=radial_n0)
        if evolution_mode != "peters_mathews_eccentricity_only":
            omega = (
                _azimuthal_from_radial_frequency_1pn(gm, n, e)
                if options.include_periastron_advance
                else n
            )
        lam = (
            _integrate_rate_from_t0(tt, omega, t0=source.t0, rate_at_t0=omega0) + source.phi0
            if options.include_periastron_advance
            else mean_anomaly + source.phi0
        )
    else:
        e = source.eccentricity
        omega = np.broadcast_to(omega0, tt.shape)
        x_for_radial = _pn_x_from_azimuthal_frequency(gm, omega)
        n = (
            _radial_from_azimuthal_frequency_1pn(omega, x_for_radial, e)
            if options.include_periastron_advance
            else omega
        )
        mean_anomaly = n * (tt - source.t0)
        lam = omega * (tt - source.t0) + source.phi0

    x_full = _pn_x_from_azimuthal_frequency(gm, omega)
    x = x_full if options.include_1pn_orbital_corrections else 0.0

    u = mikkola_kepler(mean_anomaly, e)
    one_minus_ecosu = 1.0 - e * np.cos(u)
    one_minus_e2 = 1.0 - e * e
    sqrt_one_minus_e2 = np.sqrt(one_minus_e2)

    if options.include_1pn_orbital_corrections:
        nu = one_minus_ecosu
        r_1pn = (
            -24.0
            + 9.0 * eta
            + nu * (18.0 - 7.0 * eta)
            + e * e * (24.0 - 9.0 * eta + nu * (-6.0 + 7.0 * eta))
        ) / (6.0 * nu * one_minus_e2)
        r = gm * one_minus_ecosu * (1.0 + x * r_1pn) / (C_M_PER_S**2 * x_full)
        rdot_1pn = (-7.0 * eta + e * e * (-6.0 + 7.0 * eta)) / (6.0 * one_minus_e2)
        rdot = (
            C_M_PER_S
            * np.sqrt(x_full)
            * e
            * np.sin(u)
            / one_minus_ecosu
            * (1.0 + x * rdot_1pn)
        )
        phidot_1pn = ((-1.0 + nu + e * e) * (-4.0 + eta)) / (nu * one_minus_e2)
    else:
        semimajor_scale = (gm / n**2) ** (1.0 / 3.0)
        r = semimajor_scale * one_minus_ecosu
        rdot = e * (gm * n) ** (1.0 / 3.0) * np.sin(u) / one_minus_ecosu
        phidot_1pn = 0.0

    beta_phi = _safe_beta_phi_1pn(eta, e, x)
    v_minus_u = _v_minus_u_from_beta(u, beta_phi)
    w = v_minus_u + e * np.sin(u)
    if options.include_1pn_orbital_corrections:
        w = w + x * 3.0 * (e * np.sin(u) + v_minus_u) / one_minus_e2
    phi = lam + w
    phidot = (
        omega
        * sqrt_one_minus_e2
        / one_minus_ecosu**2
        * (1.0 + x * phidot_1pn)
    )

    return {
        "u": u,
        "r": r,
        "rdot": rdot,
        "phi": phi,
        "phidot": phidot,
        "mean_anomaly": mean_anomaly,
        "radial_mean_motion": n,
        "azimuthal_frequency": omega,
    }


def eccentric_polarizations(
    source: EccentricBinaryParams,
    t: ArrayLike,
    *,
    physics_mode: str | EccentricPhysicsOptions = "1pn",
    evolution_mode: str = "fixed",
    pm_mismatch_tolerance: float = 1.0e-2,
    pm_switch_rule: PetersMathewsSwitchRule | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return `(h_plus, h_cross)` from the paper's quadrupole expressions."""

    tt = np.asarray(t, dtype=np.float64)
    dyn = eccentric_dynamics(
        source,
        tt,
        physics_mode=physics_mode,
        evolution_mode=evolution_mode,
        pm_mismatch_tolerance=pm_mismatch_tolerance,
        pm_switch_rule=pm_switch_rule,
    )
    m = source.total_mass_kg
    eta = source.symmetric_mass_ratio
    gm = G_M3_KG_S2 * m
    c = C_M_PER_S

    r = dyn["r"]
    rdot = dyn["rdot"]
    phi = dyn["phi"]
    phidot = dyn["phidot"]

    cos_i = np.cos(source.inclination)
    sin_i = np.sin(source.inclination)
    sin_2phi = np.sin(2.0 * phi)
    cos_2phi = np.cos(2.0 * phi)
    radial = gm / r
    tangential = r * r * phidot * phidot
    radial_velocity = rdot * rdot
    a = radial + tangential - radial_velocity
    b = 2.0 * r * rdot * phidot

    strain_scale = G_M3_KG_S2 * m * eta / (c**4 * source.distance_m)
    h_cross = -2.0 * strain_scale * cos_i * (a * sin_2phi - b * cos_2phi)
    h_plus = -strain_scale * (
        (1.0 + cos_i * cos_i) * (a * cos_2phi + b * sin_2phi)
        + sin_i * sin_i * (radial - tangential - radial_velocity)
    )
    return h_plus, h_cross


def eccentric_complex_strain(
    source: EccentricBinaryParams,
    t: ArrayLike,
    *,
    physics_mode: str | EccentricPhysicsOptions = "1pn",
    evolution_mode: str = "fixed",
    pm_mismatch_tolerance: float = 1.0e-2,
    pm_switch_rule: PetersMathewsSwitchRule | None = None,
) -> NDArray[np.complex128]:
    """Return `h_plus - i h_cross`, matching the package's complex convention."""

    h_plus, h_cross = eccentric_polarizations(
        source,
        t,
        physics_mode=physics_mode,
        evolution_mode=evolution_mode,
        pm_mismatch_tolerance=pm_mismatch_tolerance,
        pm_switch_rule=pm_switch_rule,
    )
    return h_plus.astype(np.complex128) - 1.0j * h_cross.astype(np.complex128)


def eccentric_harmonic_indices(order: int) -> tuple[int, ...]:
    """Return the small-e harmonic set used by the experimental response."""

    if order == 0:
        return (2,)
    if order == 1:
        return (1, 2, 3)
    if order == 2:
        return (1, 2, 3, 4)
    raise ValueError("eccentric_order must be 0, 1, or 2")


def recommended_eccentric_order(eccentricity: float) -> int | None:
    """Return the current small-e harmonic tier.

    `None` means the current small-harmonic approximation is not recommended
    and the brute-force eccentric path should be used.
    """

    _validate_eccentricity(eccentricity)
    if eccentricity == 0.0:
        return 0
    if eccentricity < 0.03:
        return 1
    if eccentricity < 0.1:
        return 2
    return None


def _rotated_polarization_matrix(
    source: EccentricBinaryParams,
    hplus_coeff: complex,
    hcross_coeff: complex,
) -> NDArray[np.complex128]:
    eplus, ecross = source.polarization_basis()
    cosps = np.cos(2.0 * source.psi)
    sinps = np.sin(2.0 * source.psi)
    plus_weight = hplus_coeff * cosps + hcross_coeff * sinps
    cross_weight = -hplus_coeff * sinps + hcross_coeff * cosps
    return plus_weight * eplus + cross_weight * ecross


def eccentric_harmonic_components(
    source: EccentricBinaryParams,
    *,
    order: int = 1,
    samples: int = 512,
    source_index: int = 0,
    include_negative_frequencies: bool = False,
) -> list[EccentricHarmonicComponent]:
    """Decompose a small-e eccentric source into positive-frequency carriers.

    The coefficients are obtained from one radial period of the paper waveform.
    Positive Fourier coefficients are doubled to form an analytic carrier
    convention compatible with the existing complex triliear response. Set
    `include_negative_frequencies=True` to return the conjugate negative
    carriers as well; this reconstructs the real time-domain polarizations.
    """

    if samples < 32:
        raise ValueError("samples must be at least 32")
    harmonics = eccentric_harmonic_indices(order)
    period = 2.0 * np.pi / source.mean_motion
    t = source.t0 + np.linspace(0.0, period, samples, endpoint=False)
    hplus, hcross = eccentric_polarizations(source, t)
    hplus_coeffs = np.fft.fft(hplus) / samples
    hcross_coeffs = np.fft.fft(hcross) / samples
    direction = source.propagation_direction()
    coefficient_scale = 1.0 if include_negative_frequencies else 2.0

    components = []
    for harmonic in harmonics:
        frequency = harmonic * source.mean_motion / (2.0 * np.pi)
        matrix = _rotated_polarization_matrix(
            source,
            coefficient_scale * hplus_coeffs[harmonic],
            coefficient_scale * hcross_coeffs[harmonic],
        )
        components.append(
            EccentricHarmonicComponent(
                frequency=frequency,
                tracefree_coefficients=matrix_to_tracefree5(matrix),
                propagation_direction=direction,
                harmonic=harmonic,
                source_index=source_index,
            )
        )
        if include_negative_frequencies:
            components.append(
                EccentricHarmonicComponent(
                    frequency=-frequency,
                    tracefree_coefficients=np.conjugate(matrix_to_tracefree5(matrix)),
                    propagation_direction=direction,
                    harmonic=-harmonic,
                    source_index=source_index,
                )
            )
    return components


def eccentric_harmonic_source_batch(
    sources: EccentricBinaryParams | list[EccentricBinaryParams],
    *,
    order: int = 1,
    samples: int = 512,
    include_negative_frequencies: bool = False,
) -> list[EccentricHarmonicComponent]:
    """Return harmonic carrier components for one or more eccentric sources."""

    source_list = [sources] if isinstance(sources, EccentricBinaryParams) else sources
    components: list[EccentricHarmonicComponent] = []
    for index, source in enumerate(source_list):
        components.extend(
            eccentric_harmonic_components(
                source,
                order=order,
                samples=samples,
                source_index=index,
                include_negative_frequencies=include_negative_frequencies,
            )
        )
    return components
