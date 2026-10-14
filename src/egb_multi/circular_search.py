"""Continuous circular frequency/chirp fitting at fixed sky and component masses.

Coordinates are alpha=(f_ref-f_center)*T, gamma=(fdot-fdot_center)*T^2.
The reference epoch is the observation midpoint. The fixed six-response physical
orientation/amplitude profile is evaluated at every intrinsic candidate. This
bounded stage is not a full sky/mass/harmonic fitting-factor search.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .circular_fit import CircularAmplitudeBasis, profile_circular_orientation
from .comparison import AESeries, AEInnerProduct
from .eccentric import EccentricBinaryParams


def circular_source_at_epoch(source: EccentricBinaryParams, epoch_s: float):
    """Change source frequency/phase coordinates, keeping detector times fixed."""
    if source.eccentricity != 0. or not np.isfinite(epoch_s):
        raise ValueError('Finite epoch and exactly circular source required')
    shift = epoch_s-source.t0
    f0 = source.mean_motion/np.pi
    return replace(source,mean_motion=np.pi*(f0+source.fdot*shift),t0=float(epoch_s),
        phi0=float((source.phi0+np.pi*f0*shift+.5*np.pi*source.fdot*shift**2)%np.pi))


class CircularFrequencyChirpSearch:
    """One immutable objective/response contract and explicitly bounded search.

    Evaluation-time bounds must cover every source time used by the response,
    including retarded SSB/link times outside the retained detector interval.
    Cached coordinates are exact floating-point tuples, without rounding.
    """

    def __init__(self, metric: AEInnerProduct, prototype: EccentricBinaryParams,
        response: Callable[[EccentricBinaryParams], AESeries], *,
        frequency_scaled_bounds: tuple[float,float], chirp_scaled_bounds: tuple[float,float],
        evaluation_time_bounds_s: tuple[float,float], orientation_starts: int=8,
        progress: Callable[[dict],None] | None=None):
        if prototype.eccentricity != 0. or prototype.m1_solar <= 0 or prototype.m2_solar <= 0:
            raise ValueError('Exactly circular templates with positive masses required')
        self.metric, self.response = metric, response
        self.duration_s = metric.t.size*metric.dt
        self.reference_epoch_s = float(.5*(metric.t[0]+metric.t[-1]))
        self.prototype = circular_source_at_epoch(prototype,self.reference_epoch_s)
        self.bounds = (tuple(frequency_scaled_bounds),tuple(chirp_scaled_bounds))
        if any(len(pair)!=2 or not np.all(np.isfinite(pair)) or pair[0]>=pair[1] for pair in self.bounds):
            raise ValueError('Finite increasing frequency/chirp bounds required')
        low_t, high_t = evaluation_time_bounds_s
        if not np.all(np.isfinite((low_t,high_t))) or low_t>metric.t[0] or high_t<metric.t[-1]:
            raise ValueError('Evaluation-time bounds must cover the retained detector interval')
        self.evaluation_time_bounds_s = (float(low_t),float(high_t))
        self.orientation_starts, self.progress = orientation_starts, progress
        self.cache, self.history = {}, []
        # Frequency is linear in time, so rectangle vertices and time endpoints
        # bound every possible trial. Reject an invalid domain before generating
        # waveforms, instead of hiding invalid trials in an optimizer penalty.
        for alpha in self.bounds[0]:
            for gamma in self.bounds[1]:
                source = self.source_at((alpha,gamma))
                instantaneous = source.mean_motion/np.pi + source.fdot*(np.array((low_t,high_t))-source.t0)
                if np.any(instantaneous<=0):
                    raise ValueError('Search domain includes nonpositive instantaneous frequency')
                if np.any(instantaneous<=metric.band_hz[0]) or np.any(instantaneous>=metric.band_hz[1]):
                    raise ValueError('Entire circular-carrier search must fit inside the fixed physical analysis band')

    def source_at(self, coordinates):
        alpha, gamma = np.asarray(coordinates,dtype=float)
        return replace(self.prototype,mean_motion=self.prototype.mean_motion+np.pi*alpha/self.duration_s,
            fdot=self.prototype.fdot+gamma/self.duration_s**2)

    def coordinates_of(self, source):
        source = circular_source_at_epoch(source,self.reference_epoch_s)
        return np.array([(source.mean_motion-self.prototype.mean_motion)/np.pi*self.duration_s,
            (source.fdot-self.prototype.fdot)*self.duration_s**2])

    def evaluate(self, coordinates):
        point = np.asarray(coordinates,dtype=float)
        if point.shape!=(2,) or np.any(~np.isfinite(point)) or any(
            value<pair[0] or value>pair[1] for value,pair in zip(point,self.bounds,strict=True)
        ):
            raise ValueError('Frequency/chirp candidate lies outside the declared search domain')
        key = tuple(point.tolist())
        if key not in self.cache:
            source = self.source_at(point)
            basis = CircularAmplitudeBasis.generate(source,self.response)
            profile = profile_circular_orientation(self.metric,basis,local_starts=self.orientation_starts)
            recovered = replace(source,inclination=profile['inclination_rad'],psi=profile['polarization_rad'],phi0=profile['orbital_phase_rad'])
            entry = {'coordinates':list(key),'frequency_hz':float(source.mean_motion/np.pi),
                'fdot_hz_s':float(source.fdot),'source':asdict(recovered),'profile':profile}
            self.cache[key] = entry
            self.history.append({'coordinates':list(key),'overlap':profile['overlap'],
                'unique_intrinsic_evaluation':len(self.cache)})
            if self.progress:
                self.progress(self.history[-1])
        return self.cache[key]

    def _objective(self, coordinates):
        return -self.evaluate(coordinates)['profile']['overlap']

    def run(self, *, coarse_shape=(9,5), starts=3, xtol=2e-5, ftol=1e-10, maxiter=60):
        if any(n<3 for n in coarse_shape) or starts<1 or xtol<=0 or ftol<=0 or maxiter<1:
            raise ValueError('Valid grid, starts and optimizer settings required')
        grid = [np.array([a,g]) for a in np.linspace(*self.bounds[0],coarse_shape[0])
            for g in np.linspace(*self.bounds[1],coarse_shape[1])]
        ranked = sorted(((self._objective(point),point) for point in grid),key=lambda item:item[0])
        # Include previously evaluated restricted-grid candidates: expansion of
        # this same objective cannot discard a better existing physical fit.
        retained = min(self.cache.values(),key=lambda entry:-entry['profile']['overlap'])
        starting_points = [np.asarray(retained['coordinates'])]+[p for _,p in ranked[:starts]]
        distinct = list({tuple(point):point for point in starting_points}.values())
        runs = []
        for start in distinct:
            fit = minimize(self._objective,start,method='Powell',bounds=self.bounds,
                options={'xtol':xtol,'ftol':ftol,'maxiter':maxiter})
            self.evaluate(fit.x)
            runs.append({'start':start.tolist(),'coordinates':fit.x.tolist(),
                'overlap':-float(fit.fun),'success':bool(fit.success),
                'message':str(fit.message),'iterations':int(fit.nit),'function_calls':int(fit.nfev)})
        result = self.result()
        return dict(result,coarse_grid_best_overlap=-float(ranked[0][0]),
            coarse_shape=list(coarse_shape),independent_start_results=runs,
            optimizer={'method':'Powell','xtol_scaled':xtol,'ftol':ftol,'maxiter':maxiter})

    def refine(self, result, *, half_width=.02, xtol=2e-6, ftol=1e-12):
        """Tighter fit plus an explicit local 3x3 grid under the same metric."""
        center = np.asarray(result['best']['coordinates'])
        grid = []
        for da in (-half_width,0.,half_width):
            for dg in (-half_width,0.,half_width):
                point = center+np.array([da,dg])
                if all(pair[0]<=value<=pair[1] for value,pair in zip(point,self.bounds,strict=True)):
                    grid.append(self.evaluate(point))
        best = max(grid,key=lambda entry:entry['profile']['overlap'])
        fit = minimize(self._objective,best['coordinates'],method='Powell',bounds=self.bounds,
            options={'xtol':xtol,'ftol':ftol,'maxiter':90})
        self.evaluate(fit.x)
        return dict(self.result(),refinement={'xtol_scaled':xtol,'ftol':ftol,
            'grid_half_width_scaled':half_width,'grid_points':len(grid),
            'grid_best_overlap':best['profile']['overlap'],
            'optimizer_overlap':-float(fit.fun),'success':bool(fit.success),
            'coordinates':fit.x.tolist(),'function_calls':int(fit.nfev)})

    def result(self):
        if not self.cache:
            raise ValueError('No candidates evaluated')
        best = max(self.cache.values(),key=lambda entry:entry['profile']['overlap'])
        distances = [min(value-pair[0],pair[1]-value) for value,pair in zip(best['coordinates'],self.bounds,strict=True)]
        return {'quantity':'bounded_frequency_chirp_and_physical_orientation_maximum_at_fixed_sky_and_masses',
            'best':best,'unique_intrinsic_evaluations':len(self.cache),
            'physical_basis_response_evaluations':6*len(self.cache),
            'duration_s':self.duration_s,'reference_epoch_s':self.reference_epoch_s,
            'frequency_scaled_bounds':list(self.bounds[0]),'chirp_scaled_bounds':list(self.bounds[1]),
            'minimum_scaled_distance_to_boundary':float(min(distances)),
            'fixed_sky_rad':{'beta':self.prototype.beta,'lambda':self.prototype.lambda_},
            'fixed_masses_solar':[self.prototype.m1_solar,self.prototype.m2_solar],
            'band_hz':list(self.metric.band_hz),'reference_norm':self.metric.reference_norm,
            'noise_weighting':self.metric.noise_weighting,
            'limitation':'No sky/mass/harmonic-basin search; optimizer and production convergence remain later gates'}
