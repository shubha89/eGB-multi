"""Coherent physical amplitude/phase/orientation fitting at fixed intrinsic parameters.

Six basis responses retain the phase-independent circular 1PN chirp term.
Only three physical angles and one common positive scale are varied. Basis
coefficients are shared by A/E; six independent amplitudes are never fitted.
Frequency, chirp, sky, masses, timestamps and detector configuration stay fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .comparison import AESeries, AEInnerProduct
from .eccentric import EccentricBinaryParams


def circular_coefficients(cos_inclination: float, psi: float, phase_shift: float):
    """Coefficients in (Pcos, Psin, Qcos, Qsin, Pradial, Qradial).

    P=e_plus(psi=0), Q=e_cross(psi=0). Each temporal basis has the
    source's common -GM*eta/(c^4*d) scale. phase_shift is orbital phase.
    The polarization tensors rotate as P'=cos(2psi)P-sin(2psi)Q.
    """
    u = float(cos_inclination)
    if not np.isfinite(u) or not -1. <= u <= 1. or not np.isfinite(psi) or not np.isfinite(phase_shift):
        raise ValueError("Invalid physical orientation")
    c, s = np.cos(2. * phase_shift), np.sin(2. * phase_shift)
    p, q = np.cos(2. * psi), np.sin(2. * psi)
    plus, cross, radial = 1. + u*u, 2.*u, 1. - u*u
    return np.array([plus*p*c + cross*q*s, -plus*p*s + cross*q*c,
                     -plus*q*c + cross*p*s, plus*q*s + cross*p*c,
                     radial*p, -radial*q])


def circular_coefficient_derivatives(cos_inclination: float, psi: float, phase_shift: float):
    """Exact Jacobian of the existing six physical coefficients, columns u/psi/phase."""
    # Validate with the public coefficient mapping rather than a separate domain.
    circular_coefficients(cos_inclination, psi, phase_shift)
    u = cos_inclination
    a, b, r = 1.+u*u, 2.*u, 1.-u*u
    p, q, c, s = np.cos(2*psi), np.sin(2*psi), np.cos(2*phase_shift), np.sin(2*phase_shift)
    return np.array([
        [2*u*p*c+2*q*s, -2*a*q*c+2*b*p*s, -2*a*p*s+2*b*q*c],
        [-2*u*p*s+2*q*c, 2*a*q*s+2*b*p*c, -2*a*p*c-2*b*q*s],
        [-2*u*q*c+2*p*s, -2*a*p*c-2*b*q*s, 2*a*q*s+2*b*p*c],
        [2*u*q*s+2*p*c, 2*a*p*s-2*b*q*c, 2*a*q*c-2*b*p*s],
        [-2*u*p, -2*r*q, 0.], [2*u*q, -2*r*p, 0.]])


def orientation_seed_from_oscillatory_coefficients(coefficients):
    """Physical angles from the SVD of the four oscillatory coefficients.

    C=R(-2psi) diag(1+u^2,2u) R(2phase), up to a positive scale.
    This is only an optimizer start: the objective retains all six responses.
    """
    matrix = np.asarray(coefficients, dtype=float).reshape(2,2)
    if np.any(~np.isfinite(matrix)) or not np.any(matrix):
        return None
    left, singular, right = np.linalg.svd(matrix)
    signed_ratio = singular[1]/singular[0]
    if np.linalg.det(left) < 0:
        left[:,1] *= -1.
        signed_ratio *= -1.
    if np.linalg.det(right) < 0:
        right[1,:] *= -1.
        signed_ratio *= -1.
    # A singular-value ratio is <=1 by construction; do not clip an invalid one.
    if abs(signed_ratio) > 1.:
        raise ValueError('Invalid oscillatory singular-value ratio')
    u = signed_ratio/(1.+np.sqrt(1.-signed_ratio**2))
    psi = (-.5*np.arctan2(left[1,0],left[0,0])) % np.pi
    phase = (.5*np.arctan2(right[1,0],right[0,0])) % np.pi
    return np.array([u,psi,phase])


def _orientation_statistics(metric, basis):
    prepared = basis.prepared if isinstance(basis,PreparedCircularAmplitudeBasis) else np.array([metric.prepare(item) for item in basis.responses])
    if prepared.shape!=(6,*metric.reference.shape) or not np.all(np.isfinite(prepared)):
        raise ValueError('Six aligned finite prepared circular responses required')
    if type(metric) is AEInnerProduct:
        # The stationary metric is Euclidean after whitening. Validate once
        # above, then use one small Gram contraction instead of rescanning the
        # same large vectors for every pair. Other metrics retain their own
        # bilinear form (notably the finite covariance solve-pair metric).
        flat=prepared.reshape(6,-1)
        gram=(flat@flat.conj().T).real
        projections=(flat@metric.reference.conj().ravel()).real
    else:
        gram = np.array([[metric.inner(a,b) for b in prepared] for a in prepared])
        projections = np.array([metric.inner(metric.reference,b) for b in prepared])
    scale = float(np.max(np.diag(gram)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Circular basis has no weighted power')
    g, p = gram/scale, projections/np.sqrt(metric.reference_norm*scale)
    if np.linalg.eigvalsh(g)[0] < -1e-10:
        raise ValueError('Circular Gram matrix is not positive semidefinite')
    return prepared, gram, g, p


@dataclass
class PreparedCircularAmplitudeBasis:
    """Six responses prepared under the same immutable comparison metric."""
    source: EccentricBinaryParams
    prepared: np.ndarray

    def __post_init__(self):
        if self.source.eccentricity!=0:
            raise ValueError('Prepared circular basis requires eccentricity=0')
        self.prepared=np.asarray(self.prepared)


def profile_circular_orientation(metric: AEInnerProduct, basis: CircularAmplitudeBasis, *, local_starts=8, gtol=1e-10):
    """Deterministic physical six-response profile for repeated intrinsic trials.

    Grid starts and a four-coefficient SVD start feed analytic-gradient local
    fits. Both phase/polarization branches of the oscillatory SVD are included.
    No radial response or physical coefficient constraint is dropped. Accuracy
    must be checked against the independent global orientation routine.
    """
    if local_starts < 1 or not np.isfinite(gtol) or gtol <= 0:
        raise ValueError('Positive local-start count and gradient tolerance required')
    prepared, gram, g, p = _orientation_statistics(metric,basis)
    evaluations = 0

    def objective(angles):
        nonlocal evaluations
        evaluations += 1
        coeff = circular_coefficients(*angles)
        jac = circular_coefficient_derivatives(*angles)
        norm = float(coeff @ g @ coeff)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('Physical orientation has nonpositive norm')
        score = float(coeff @ p/np.sqrt(norm))
        gradient = jac.T @ p/np.sqrt(norm) - score*(jac.T @ g @ coeff)/norm
        return -score, -gradient

    grid = [(u,psi,phase) for u in (-1.,-.75,0.,.75,1.)
        for psi in np.arange(8)*np.pi/8 for phase in np.arange(8)*np.pi/8]
    ranked = sorted(((objective(point)[0],np.array(point)) for point in grid),key=lambda item:item[0])
    starts = [point for _,point in ranked[:local_starts]]
    seed_status = 'singular_four_coefficient_Gram; grid starts retained'
    try:
        seed = orientation_seed_from_oscillatory_coefficients(np.linalg.solve(g[:4,:4],p[:4]))
    except np.linalg.LinAlgError:
        seed = None
    if seed is not None:
        starts.extend([seed, np.array([seed[0],(seed[1]+np.pi/2)%np.pi,(seed[2]+np.pi/2)%np.pi])])
        seed_status = 'two_SVD_phase_polarization_branches_used_as_starts_only'
    candidates = [(ranked[0][0],ranked[0][1],'grid')]
    runs = []
    for start in starts:
        fit = minimize(objective,start,jac=True,method='L-BFGS-B',
            bounds=[(-1.,1.),(0.,np.pi),(0.,np.pi)],
            options={'gtol':gtol,'ftol':1e-15,'maxiter':500})
        candidates.append((float(fit.fun),fit.x,'analytic_gradient_local_refinement'))
        runs.append({'start':start.tolist(),'overlap':-float(fit.fun),
            'success':bool(fit.success),'iterations':int(fit.nit),'message':str(fit.message)})
    _, angles, method = min(candidates,key=lambda item:item[0])
    coeff = circular_coefficients(*angles)
    metrics = metric.compare_prepared(np.einsum('i,ijk->jk',coeff,prepared))
    return dict(metrics, quantity='physical_extrinsic_profile_at_fixed_frequency_chirp_sky_masses',
        cos_inclination=float(angles[0]),inclination_rad=float(np.arccos(angles[0])),
        polarization_rad=float(angles[1]%np.pi),
        orbital_phase_rad=float((basis.source.phi0+angles[2])%np.pi),
        orbital_phase_shift_rad=float(angles[2]%np.pi),coefficients=coeff.tolist(),
        basis_gram=gram.tolist(),objective_evaluations=evaluations,local_results=runs,
        SVD_seed_status=seed_status,best_method=method,
        amplitude_treatment='one_common_positive_scale_in_A_and_E',
        limitation='Local multi-start physical profile; compare against global orientation validation before production')


@dataclass
class CircularAmplitudeBasis:
    source: EccentricBinaryParams
    responses: tuple[AESeries, ...]

    @classmethod
    def generate(cls, source: EccentricBinaryParams, response: Callable[[EccentricBinaryParams], AESeries]):
        if source.eccentricity != 0.:
            raise ValueError("Physical circular amplitude basis requires eccentricity=0")
        direct = getattr(response, 'circular_basis', None)
        if direct is not None:
            basis = direct(source)
            if not isinstance(basis, cls) or basis.source != source or len(basis.responses) != 6:
                raise ValueError('Direct circular basis must retain the exact source and six responses')
            prototype = basis.responses[0]
            if any(item.convention != prototype.convention or not np.array_equal(item.t, prototype.t)
                   for item in basis.responses):
                raise ValueError('Basis responses do not share detector conventions/timestamps')
            return basis
        vectors = []
        prototype = None
        # Edge-on sources eliminate h_cross. psi=-pi/4 turns e_plus into Q.
        for psi in (0., -.25 * np.pi):
            series = [response(replace(source, inclination=.5*np.pi, psi=psi, phi0=source.phi0+shift)) for shift in (0., .25*np.pi, .5*np.pi)]
            prototype = series[0] if prototype is None else prototype
            for item in series:
                if item.convention != prototype.convention or not np.array_equal(item.t, prototype.t):
                    raise ValueError("Basis responses do not share detector conventions/timestamps")
            radial = {ch: .5*(series[0].channels[ch]+series[2].channels[ch]) for ch in ("A", "E")}
            cosine = {ch: .5*(series[0].channels[ch]-series[2].channels[ch]) for ch in ("A", "E")}
            sine = {ch: radial[ch]-series[1].channels[ch] for ch in ("A", "E")}
            vectors.extend((cosine, sine, radial))
        ordered = [vectors[index] for index in (0, 1, 3, 4, 2, 5)]
        return cls(source, tuple(AESeries(prototype.t, value, prototype.convention) for value in ordered))

    def reconstruct(self, *, inclination: float, psi: float, phi0: float, amplitude_scale: float = 1.):
        if not np.isfinite(inclination) or not 0 <= inclination <= np.pi:
            raise ValueError("Inclination must lie in [0, pi]")
        if not np.isfinite(amplitude_scale) or amplitude_scale <= 0:
            raise ValueError("Common amplitude scale must be positive")
        coeff = amplitude_scale*circular_coefficients(np.cos(inclination), psi, phi0-self.source.phi0)
        prototype = self.responses[0]
        values = {ch: sum(c*item.channels[ch] for c, item in zip(coeff, self.responses, strict=True)) for ch in ("A", "E")}
        return AESeries(prototype.t, values, prototype.convention)


def maximize_circular_orientation(metric: AEInnerProduct, basis: CircularAmplitudeBasis, *, seeds=(1234, 5678), maxiter=180):
    """Numerically maximize physical inclination, polarization and orbital phase.

    The amplitude is profiled analytically and constrained nonnegative.
    A coarse full-domain grid, independent DE seeds and bounded local refinements
    provide reproducible candidates. This is an extrinsic-only maximum, not FF.
    Search accuracy is reported, not assumed from an optimizer success flag.
    """
    if not seeds or maxiter < 1:
        raise ValueError("At least one deterministic seed and positive iteration count required")
    prepared, gram, g, p = _orientation_statistics(metric,basis)
    evaluations = 0

    def objective(angles):
        nonlocal evaluations
        evaluations += 1
        coeff = circular_coefficients(*angles)
        norm = float(coeff @ g @ coeff)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Physical orientation has nonpositive norm")
        return -float(coeff @ p / np.sqrt(norm))

    bounds = [(-1., 1.), (0., np.pi), (0., np.pi)]
    grid = [(u, psi, phase) for u in (-1., -.75, 0., .75, 1.) for psi in np.arange(8)*np.pi/8 for phase in np.arange(8)*np.pi/8]
    initial = sorted(((objective(point), np.array(point)) for point in grid), key=lambda item: item[0])
    candidates = [(initial[0][0], initial[0][1], "grid")]
    stochastic = []
    local_starts = [point for _, point in initial[:12]]
    for seed in seeds:
        result = differential_evolution(objective, bounds, rng=np.random.default_rng(seed), popsize=12, maxiter=maxiter, tol=1e-10, atol=1e-12, polish=False)
        stochastic.append({"seed": int(seed), "overlap": -float(result.fun), "success": bool(result.success), "iterations": int(result.nit)})
        candidates.append((float(result.fun), result.x, f"DE seed {seed}"))
        local_starts.append(result.x)
    local_results = []
    for start in local_starts:
        result = minimize(objective, start, method="L-BFGS-B", bounds=bounds, jac="3-point", options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 500})
        candidates.append((float(result.fun), result.x, "local refinement"))
        local_results.append(-float(result.fun))
    value, angles, method = min(candidates, key=lambda item: item[0])
    u, psi, phase = angles
    coeff = circular_coefficients(u, psi, phase)
    template = np.einsum("i,ijk->jk", coeff, prepared)
    metrics = metric.compare_prepared(template)
    result = dict(metrics,
        quantity="physical_extrinsic_maximum_at_fixed_frequency_chirp_sky_masses",
        cos_inclination=float(u), inclination_rad=float(np.arccos(u)),
        polarization_rad=float(psi % np.pi),
        orbital_phase_rad=float((basis.source.phi0+phase) % np.pi),
        orbital_phase_shift_rad=float(phase % np.pi),
        coefficients=coeff.tolist(),
        basis_gram=gram.tolist(),
        grid_best_overlap=-float(initial[0][0]),
        independent_seed_results=stochastic,
        local_refinement_best_overlap=max(local_results),
        best_method=method, objective_evaluations=evaluations,
        amplitude_treatment="one_common_positive_scale_in_A_and_E",
        limitation="No frequency, chirp, sky or mass search; full search validation remains Task 2.7")
    return result
