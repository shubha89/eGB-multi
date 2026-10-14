"""Bounded multi-basin circular fitting over frequency, chirp, sky and masses.

This expands the exact physical circular family, with one immutable A/E metric.
Spectrum-informed basin selection is for clean-signal fitting, not a detection
algorithm. Full optimizer and production convergence remain separate checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks

from .circular_fit import CircularAmplitudeBasis, profile_circular_orientation
from .circular_search import circular_source_at_epoch
from .comparison import AESeries, AEInnerProduct
from .eccentric import EccentricBinaryParams


@dataclass(frozen=True)
class FrequencyBasin:
    center_hz: float
    half_width_cycles: float
    label: str = 'spectral_peak'


def ordered_masses(coordinates, bounds=(.1, 1.4)):
    """Map the unit square onto lo <= m2 <= m1 <= hi, in solar masses."""
    u, v = np.asarray(coordinates, dtype=float)
    lo, hi = bounds
    if not 0 < lo < hi or not np.all(np.isfinite((u, v))) or not (0 <= u <= 1 and 0 <= v <= 1):
        raise ValueError('Invalid ordered-mass coordinates or bounds')
    m2 = lo + (hi-lo)*u
    m1 = m2 + (hi-m2)*v
    return float(m1), float(m2)


def mass_coordinates(m1, m2, bounds=(.1, 1.4)):
    lo, hi = bounds
    if not 0 < lo < hi or not np.all(np.isfinite((m1, m2))) or not lo <= m2 <= m1 <= hi:
        raise ValueError('Masses must lie in the ordered domain')
    return np.array([(m2-lo)/(hi-lo), (m1-m2)/(hi-m2) if m2 < hi else 0.])


def equal_area_sky(count):
    """Deterministic Fibonacci sphere plus both poles; longitude is periodic."""
    if count < 4:
        raise ValueError('At least four equal-area sky points required')
    index = np.arange(count)
    sine_latitude = 1.-2.*(index+.5)/count
    longitude = (index*np.pi*(3.-np.sqrt(5.))) % (2.*np.pi)
    return np.vstack((np.column_stack((sine_latitude, longitude)), [[-1., 0.], [1., 0.]]))


def _unscale_optimizer_coordinates(coordinates, scales, bounds, scaled_bounds):
    """Invert the optimizer chart while preserving its exact box endpoints.

    Multiplication then division by a positive curvature scale need not round
    back to the same float. In particular (lower*scale)/scale can lie one ulp
    below lower. A solver coordinate equal to a transformed endpoint denotes
    that physical endpoint exactly. Interior division is unchanged; any
    residual endpoint overshoot must be within four ulps, a conservative
    envelope for the multiply/divide round trip. Genuine scaled-domain
    violations still raise, and the public evaluate() guard stays strict.
    """
    y = np.asarray(coordinates, dtype=float)
    scales, bounds, scaled_bounds = [np.asarray(a, dtype=float) for a in (scales, bounds, scaled_bounds)]
    if (y.ndim != 1 or scales.shape != y.shape or bounds.shape != (y.size, 2)
        or scaled_bounds.shape != bounds.shape or not all(np.all(np.isfinite(a)) for a in (y, scales, bounds, scaled_bounds))
        or np.any(scales <= 0) or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError('Finite positive optimizer scales and aligned box bounds required')
    if np.any(y < scaled_bounds[:, 0]) or np.any(y > scaled_bounds[:, 1]):
        raise ValueError('Optimizer coordinate outside its declared scaled domain')
    raw = y/scales
    physical = raw.copy()
    at_lower, at_upper = y == scaled_bounds[:, 0], y == scaled_bounds[:, 1]
    physical[at_lower] = bounds[at_lower, 0]
    physical[at_upper] = bounds[at_upper, 1]
    ulps = np.spacing(np.maximum(abs(bounds[:, 0]), abs(bounds[:, 1])))
    outside = (physical < bounds[:, 0]) | (physical > bounds[:, 1])
    distance = np.maximum(bounds[:, 0]-physical, physical-bounds[:, 1])
    if np.any(outside & (distance > 4*ulps)):
        raise ValueError('Optimizer unscaling exceeds the floating-point endpoint allowance')
    physical = np.maximum(bounds[:, 0], np.minimum(bounds[:, 1], physical))
    correction = abs(physical-raw)
    return physical, {'corrected_axes':np.flatnonzero(correction).tolist(),
        'maximum_correction_in_bound_ulps':float(np.max(correction/ulps)),
        'maximum_absolute_correction_scaled_physical_coordinates':float(correction.max())}


def select_frequency_basins(metric: AEInnerProduct, signal: AESeries, *,
    half_width_cycles=4., minimum_peak_fraction=.005, target_coverage=.999,
    extra_frequencies=()):
    """Select actual noise-weighted response peaks, retaining quantified coverage.

    A Hann window is used ONLY for seed selection, never in the fit objective.
    Coverage is of this tapered seed spectrum, not a bound on FF search error.
    All peaks above the relative threshold are retained; weaker peaks are added
    until the declared coverage is reached. A requested band-edge peak fails
    explicitly rather than being silently dropped or shifting the fit band.
    """
    if half_width_cycles <= 0 or not 0 < minimum_peak_fraction <= 1 or not 0 < target_coverage <= 1:
        raise ValueError('Invalid spectral selection settings')
    window = np.hanning(metric.t.size)
    tapered = AESeries(signal.t, {ch:signal.channels[ch]*window for ch in ('A', 'E')}, signal.convention)
    weighted = metric.prepare(tapered)
    power = np.sum(np.abs(weighted)**2, axis=0)
    if not np.all(np.isfinite(power)) or power.sum() <= 0:
        raise ValueError('No finite spectral power for basin selection')
    peaks = list(find_peaks(power)[0])
    if power.size == 1 or power[0] >= power[1]:
        peaks.append(0)
    if power.size > 1 and power[-1] >= power[-2]:
        peaks.append(power.size-1)
    peaks = sorted(set(peaks), key=lambda i:power[i], reverse=True)
    selected, mask = [], np.zeros(power.size, dtype=bool)
    frequency, df = metric.frequency_hz, metric.df
    def add(center, label):
        if any(abs(center-b.center_hz) <= half_width_cycles*df for b in selected):
            return
        if center-half_width_cycles*df <= metric.band_hz[0] or center+half_width_cycles*df >= metric.band_hz[1]:
            raise ValueError('Relevant frequency basin reaches the fixed band edge; expand the analysis band')
        selected.append(FrequencyBasin(float(center), float(half_width_cycles), label))
        mask[:] |= np.abs(frequency-center) <= half_width_cycles*df
    for center in extra_frequencies:
        add(float(center), 'retained_carrier')
    strongest = power[peaks[0]]
    for index in peaks:
        fraction = float(power[mask].sum()/power.sum())
        if power[index] < minimum_peak_fraction*strongest and fraction >= target_coverage:
            break
        add(frequency[index], 'weighted_response_peak')
    coverage = float(power[mask].sum()/power.sum())
    if coverage < target_coverage:
        raise ValueError('Selected basins cannot cover the requested seed-spectrum power; enlarge basin widths')
    return tuple(selected), {'method':'Hann_tapered_AE_weighted_response_peaks_for_seeding_only',
        'minimum_peak_fraction':minimum_peak_fraction, 'target_coverage':target_coverage,
        'covered_seed_spectrum_power_fraction':coverage,
        'uncovered_seed_spectrum_power_fraction':1.-coverage,
        'fit_window':'none', 'basins':[asdict(b) for b in selected],
        'limitation':'Seed-spectrum coverage is not an optimizer-error bound or detection calibration'}


class CircularFamilySearch:
    """Continuous six-coordinate search, with the physical orientation profiled.

    Coordinates are (delta_f*T, fdot*T^2, sin(beta), lambda, mass_u, mass_v).
    Each basin has its own frequency origin; longitude is canonical modulo 2pi
    in evaluation and searched on a full-period chart around each local start.
    Evaluation-time bounds must cover source times for EVERY sky direction.
    """

    def __init__(self, metric: AEInnerProduct, prototype: EccentricBinaryParams,
        response: Callable[[EccentricBinaryParams], AESeries], *, basins,
        chirp_scaled_bounds, evaluation_time_bounds_s, mass_bounds_solar=(.1, 1.4),
        orientation_starts=8, progress=None):
        if prototype.eccentricity != 0.:
            raise ValueError('Exactly circular prototype required')
        self.metric, self.response = metric, response
        self.duration_s = metric.t.size*metric.dt
        self.epoch_s = float(.5*(metric.t[0]+metric.t[-1]))
        self.prototype = circular_source_at_epoch(prototype, self.epoch_s)
        self.mass_bounds_solar = tuple(mass_bounds_solar)
        mass_coordinates(prototype.m1_solar, prototype.m2_solar, self.mass_bounds_solar)
        self.basins = tuple(basins)
        self.chirp_scaled_bounds = tuple(chirp_scaled_bounds)
        self.evaluation_time_bounds_s = tuple(evaluation_time_bounds_s)
        self.orientation_starts, self.progress = orientation_starts, progress
        self.cache, self.history, self.stage_results = {}, [], []
        if not self.basins or len(self.chirp_scaled_bounds) != 2 or not np.all(np.isfinite(self.chirp_scaled_bounds)) or self.chirp_scaled_bounds[0] >= self.chirp_scaled_bounds[1]:
            raise ValueError('Nonempty basins and increasing finite chirp bounds required')
        low_t, high_t = self.evaluation_time_bounds_s
        if not np.all(np.isfinite((low_t, high_t))) or low_t > metric.t[0] or high_t < metric.t[-1]:
            raise ValueError('All-sky source time bounds must cover detector timestamps')
        for basin in self.basins:
            if not np.isfinite(basin.center_hz) or not np.isfinite(basin.half_width_cycles) or basin.half_width_cycles <= 0:
                raise ValueError('Finite basin center and positive width required')
            for alpha in (-basin.half_width_cycles, basin.half_width_cycles):
                for gamma in self.chirp_scaled_bounds:
                    frequencies = basin.center_hz+alpha/self.duration_s + gamma/self.duration_s**2*(np.array([low_t,high_t])-self.epoch_s)
                    if np.any(frequencies <= 0):
                        raise ValueError('Search domain includes nonpositive instantaneous frequency')
                    if np.any(frequencies <= metric.band_hz[0]) or np.any(frequencies >= metric.band_hz[1]):
                        raise ValueError('Frequency/chirp basin reaches the fixed band edge')

    def coordinates_of(self, source, basin_index):
        source = circular_source_at_epoch(source, self.epoch_s)
        return np.r_[(source.mean_motion/np.pi-self.basins[basin_index].center_hz)*self.duration_s,
            source.fdot*self.duration_s**2, np.sin(source.beta), source.lambda_ % (2*np.pi),
            mass_coordinates(source.m1_solar, source.m2_solar, self.mass_bounds_solar)]

    def source_at(self, coordinates, basin_index):
        alpha, gamma, sine_latitude, longitude, mass_u, mass_v = np.asarray(coordinates, dtype=float)
        m1, m2 = ordered_masses((mass_u, mass_v), self.mass_bounds_solar)
        return replace(self.prototype, mean_motion=np.pi*(self.basins[basin_index].center_hz+alpha/self.duration_s),
            fdot=gamma/self.duration_s**2, beta=float(np.arcsin(sine_latitude)),
            lambda_=float(longitude % (2*np.pi)), m1_solar=m1, m2_solar=m2)

    def evaluate(self, coordinates, basin_index=0):
        point = np.asarray(coordinates, dtype=float)
        if not isinstance(basin_index, (int, np.integer)) or not 0 <= basin_index < len(self.basins):
            raise ValueError('Invalid frequency basin index')
        basin = self.basins[basin_index]
        if point.shape != (6,) or not np.all(np.isfinite(point)) or not (
            -basin.half_width_cycles <= point[0] <= basin.half_width_cycles and
            self.chirp_scaled_bounds[0] <= point[1] <= self.chirp_scaled_bounds[1] and
            -1 <= point[2] <= 1 and np.all((point[4:] >= 0) & (point[4:] <= 1))):
            raise ValueError('Candidate outside declared circular-family domain')
        source = self.source_at(point, basin_index)
        # Cache only exact physical coordinates. No rounding of frequencies,
        # chirps, sky or masses is allowed to alter the objective.
        key = (source.mean_motion, source.fdot, source.beta, source.lambda_, source.m1_solar, source.m2_solar)
        if key not in self.cache:
            direct_profile=getattr(self.response,'profile_source',None)
            if direct_profile is None:
                basis = CircularAmplitudeBasis.generate(source, self.response)
                profile = profile_circular_orientation(self.metric, basis, local_starts=self.orientation_starts)
            else:
                profile=direct_profile(self.metric,source,local_starts=self.orientation_starts)
            recovered = replace(source, inclination=profile['inclination_rad'], psi=profile['polarization_rad'], phi0=profile['orbital_phase_rad'])
            entry = {'source':asdict(recovered), 'profile':profile,
                'frequency_hz':float(source.mean_motion/np.pi), 'fdot_hz_s':float(source.fdot),
                'first_basin_index':int(basin_index)}
            self.cache[key] = entry
            history = {'unique_intrinsic_evaluation':len(self.cache), 'basin_index':int(basin_index),
                'coordinates':point.tolist(), 'source':asdict(source), 'overlap':profile['overlap']}
            self.history.append(history)
            if self.progress:
                self.progress(history)
        return self.cache[key]

    def retain(self, source):
        """Re-evaluate a previous candidate with this identical fixed objective."""
        for index, basin in enumerate(self.basins):
            point = self.coordinates_of(source, index)
            if abs(point[0]) <= basin.half_width_cycles:
                return self.evaluate(point, index)
        raise ValueError('Retained source outside every declared frequency basin')

    def _bounds(self, basin_index, longitude):
        width = self.basins[basin_index].half_width_cycles
        return [(-width,width), self.chirp_scaled_bounds, (-1.,1.),
            (longitude-np.pi,longitude+np.pi), (0.,1.), (0.,1.)]

    def _local_fit(self, point, basin_index, *, ftol, maxiter, finite_difference_step):
        """Precondition SLSQP using measured diagonal objective curvature.

        Frequency is sharp while short-observation sky/mass directions can be
        almost flat. y_i=sqrt(|d2(-overlap)/dx_i2|)*x_i makes local curvature
        comparable. A 1e-6 curvature floor caps inverse scaling along flat or
        roundoff-dominated directions; it changes no physical bound or score.
        Finite differences remain the declared step in PHYSICAL coordinates.
        """
        bounds = np.array(self._bounds(basin_index,point[3]),dtype=float)
        self.evaluate(point, basin_index)  # Reject invalid starts before preconditioning.
        objective = lambda x:-self.evaluate(x,basin_index)['profile']['overlap']
        curvature = []
        for axis,(low,high) in enumerate(bounds):
            step = min(.02,.25*(high-low))
            offsets = (-step,0.,step)
            if point[axis]-step < low:
                offsets = (0.,step,2*step)
            elif point[axis]+step > high:
                offsets = (-2*step,-step,0.)
            values = []
            for offset in offsets:
                trial = point.copy();trial[axis] += offset
                values.append(objective(trial))
            curvature.append((values[0]-2*values[1]+values[2])/step**2)
        scales = np.sqrt(np.maximum(np.abs(curvature),1e-6))
        scaled_bounds = bounds*scales[:,None]
        corrections = []
        def physical_coordinates(y):
            x, record = _unscale_optimizer_coordinates(y, scales, bounds, scaled_bounds)
            if record['corrected_axes']:
                corrections.append(record)
            return x
        fit = minimize(lambda y:objective(physical_coordinates(y)),point*scales,method='SLSQP',
            bounds=scaled_bounds,options={'ftol':ftol,'maxiter':maxiter,
                'eps':finite_difference_step*scales})
        coordinates = physical_coordinates(fit.x)
        self.evaluate(coordinates,basin_index)
        return fit,coordinates,{'measured_diagonal_curvature':curvature,
            'coordinate_scales':scales.tolist(),'curvature_floor':1e-6,
            'curvature_probe_step_scaled':.02,'physical_finite_difference_step':finite_difference_step,
            'optimizer_chart_endpoint_roundoff_control':{
                'rule':'Exact transformed endpoints map to exact physical endpoints; public domain remains strict',
                'roundtrip_allowance_in_bound_ulps':4,'correction_events':len(corrections),
                'corrected_axes':sorted({axis for r in corrections for axis in r['corrected_axes']}),
                'maximum_correction_in_bound_ulps':max((r['maximum_correction_in_bound_ulps'] for r in corrections),default=0.)}}

    def run(self, *, sky_points=24, coarse_shape=(5,3), starts_per_basin=2,
        ftol=1e-11, maxiter=90, finite_difference_step=2e-5):
        if starts_per_basin < 1 or any(n < 3 for n in coarse_shape) or ftol <= 0 or maxiter < 1 or finite_difference_step <= 0:
            raise ValueError('Valid coarse grid and local search settings required')
        sky = equal_area_sky(sky_points)
        base_masses = mass_coordinates(self.prototype.m1_solar, self.prototype.m2_solar, self.mass_bounds_solar)
        mass_seeds = [(0.,0.), (0.,1.), (1.,0.), tuple(base_masses), (.5,.5)]
        stages = []
        for index, basin in enumerate(self.basins):
            candidates = []
            def trial(point):
                item = self.evaluate(point,index)
                candidates.append((item['profile']['overlap'], np.asarray(point,dtype=float)))
                return item
            # Find a useful intrinsic seed before sampling sky. No injection
            # sky, chirp or mass values enter the automatic sky/mass starts.
            for a in np.linspace(-basin.half_width_cycles,basin.half_width_cycles,coarse_shape[0]):
                for g in np.linspace(*self.chirp_scaled_bounds,coarse_shape[1]):
                    trial(np.r_[a,g,np.sin(self.prototype.beta),self.prototype.lambda_ % (2*np.pi),base_masses])
            center = max(candidates,key=lambda pair:pair[0])[1]
            for s, longitude in sky:
                point = center.copy(); point[2:4] = (s,longitude)
                trial(point)
            # Include retained candidates in this basin without discarding their
            # objective values when expanding the search or changing starts.
            for item in list(self.cache.values()):
                source = EccentricBinaryParams(**item['source'])
                point = self.coordinates_of(source,index)
                if abs(point[0]) <= basin.half_width_cycles:
                    trial(point)
            mass_center = max(candidates,key=lambda pair:pair[0])[1]
            mass_scores = []
            for mass_point in mass_seeds:
                point = mass_center.copy(); point[4:] = mass_point
                item = trial(point)
                mass_scores.append({'masses_solar':[item['source']['m1_solar'],item['source']['m2_solar']],
                    'overlap':item['profile']['overlap']})
            ranked = sorted(candidates,key=lambda pair:pair[0],reverse=True)
            # Keep starts with different sky directions, not multiple nearly
            # identical starts caused only by flat mass/amplitude directions.
            starts = []
            for _, point in ranked:
                direction = np.array([np.sqrt(1-point[2]**2)*np.cos(point[3]),
                    np.sqrt(1-point[2]**2)*np.sin(point[3]),point[2]])
                if not starts or all(np.linalg.norm(direction-other[1]) > .25 for other in starts):
                    starts.append((point,direction))
                if len(starts) == starts_per_basin:
                    break
            runs = []
            for point, _ in starts:
                fit,coordinates,conditioning = self._local_fit(point,index,ftol=ftol,
                    maxiter=maxiter,finite_difference_step=finite_difference_step)
                item = trial(coordinates)
                runs.append({'start':point.tolist(),'coordinates':coordinates.tolist(),
                    'overlap':item['profile']['overlap'],'success':bool(fit.success),
                    'message':str(fit.message),'iterations':int(fit.nit),'function_calls':int(fit.nfev),
                    'conditioning':conditioning})
            winner = max(candidates,key=lambda pair:pair[0])
            best = self.evaluate(winner[1],index)
            stages.append({'basin_index':index,'basin':asdict(basin),'best':best,
                'best_coordinates':winner[1].tolist(),'independent_start_results':runs,
                'mass_seed_scores':mass_scores,
                'frequency_chirp_boundary_distance_scaled':float(min(basin.half_width_cycles-abs(winner[1][0]),
                    winner[1][1]-self.chirp_scaled_bounds[0],self.chirp_scaled_bounds[1]-winner[1][1]))})
        self.stage_results = stages
        return dict(self.result(), basins_results=stages,
            settings={'sky_points':sky_points,'coarse_shape':list(coarse_shape),
                'starts_per_basin':starts_per_basin,'method':'curvature_preconditioned_SLSQP','ftol':ftol,
                'maxiter':maxiter,'finite_difference_step_scaled':finite_difference_step})

    def refine_best(self, *, ftol=1e-13, finite_difference_step=2e-6, maxiter=120):
        best = self.result()['best']
        source = EccentricBinaryParams(**best['source'])
        index = min(range(len(self.basins)),key=lambda i:abs(self.basins[i].center_hz-best['frequency_hz']))
        point = self.coordinates_of(source,index)
        fit,coordinates,conditioning = self._local_fit(point,index,ftol=ftol,
            maxiter=maxiter,finite_difference_step=finite_difference_step)
        return dict(self.result(),refinement={'success':bool(fit.success),'message':str(fit.message),
            'ftol':ftol,'finite_difference_step_scaled':finite_difference_step,
            'coordinates':coordinates.tolist(),'function_calls':int(fit.nfev),'conditioning':conditioning})

    def result(self):
        if not self.cache:
            raise ValueError('No circular-family candidates evaluated')
        best = max(self.cache.values(),key=lambda entry:entry['profile']['overlap'])
        return {'quantity':'bounded_multibasin_circular_family_maximum', 'best':best,
            'unique_intrinsic_evaluations':len(self.cache),
            'physical_basis_response_evaluations':6*len(self.cache),
            'mass_bounds_solar':list(self.mass_bounds_solar),
            'sky_domain':'full_sphere_with_periodic_longitude',
            'chirp_scaled_bounds':list(self.chirp_scaled_bounds),
            'basins':[asdict(b) for b in self.basins],
            'reference_epoch_s':self.epoch_s,'duration_s':self.duration_s,
            'fixed_band_hz':list(self.metric.band_hz),'fixed_reference_norm':self.metric.reference_norm,
            'noise_weighting':self.metric.noise_weighting,
            'limitation':'Finite spectrum-informed basins and starts; full optimizer/production certification remain 2.7-2.8'}
