"""Bounded-memory native responses and numerical certification diagnostics.

No new waveform or TDI approximation is introduced: each block uses the local
exact retarded links and native prepared factorized TDI, with a conservative
interpolation halo. Source and detector epochs remain absolute across blocks.
"""

from collections import OrderedDict

import numpy as np

from .comparison import AESeries, AEInnerProduct
from .eccentric_jax import eccentric_links_jax, precompute_jax_link_geometry
from .external import state_from_lisaorbits, link_delays_from_lisaorbits
from .noise import aet_from_xyz
from .tdi import PreparedGWTDIMichelson


class ChunkedNativeResponse:
    """Exact native response on a fixed uniform grid, with padded block edges.

    ``times`` contains the retained analysis samples; padding is generated
    outside that interval. The delay bound is checked in every block and is
    not substituted for the actual delays. ``auto`` evolution is deliberately
    rejected: its decision must use the complete observation, not each block.
    """

    def __init__(self, orbit, times, convention, *, block_samples=65536,
                 cached_blocks=2, delay_bound_s=12.):
        if convention.unit!='fractional_frequency':
            raise ValueError('Native GW response requires fractional_frequency units')
        t = np.asarray(times, dtype=float)
        # Reuse the series grid validation rather than another time convention.
        probe = AESeries(t, {'A':np.zeros(t.size), 'E':np.zeros(t.size)}, convention)
        if block_samples < 16 or cached_blocks < 0 or not np.isfinite(delay_bound_s) or delay_bound_s <= 0:
            raise ValueError('Valid block size, cache size and delay bound required')
        self.orbit, self.times, self.convention = orbit, probe.t, convention
        self.dt, self.block_samples = probe.dt, int(block_samples)
        self.cached_blocks, self.delay_bound_s = int(cached_blocks), float(delay_bound_s)
        self.halo = int(np.ceil(8*delay_bound_s/self.dt))+4*(convention.measurement_order+1)
        self.cache = OrderedDict()

    def _block(self, start, stop):
        key = (start,stop)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        indices = np.arange(start-self.halo,stop+self.halo)
        times = self.times[0]+indices*self.dt
        state = state_from_lisaorbits(self.orbit,times,frame='ecliptic')
        geometry = precompute_jax_link_geometry(state,orbits=self.orbit,frame='ecliptic')
        delays = link_delays_from_lisaorbits(self.orbit,times)
        if any(np.any(~np.isfinite(d)) or np.any(d <= 0) or np.max(d) > self.delay_bound_s for d in delays.values()):
            raise ValueError('Actual link delay violates the declared padding bound')
        operator = PreparedGWTDIMichelson(state,delays,
            generation=self.convention.generation,
            measurement_order=self.convention.measurement_order,
            delay_order=self.convention.delay_order)
        block = (geometry,operator)
        if self.cached_blocks:
            self.cache[key] = block
            while len(self.cache) > self.cached_blocks:
                self.cache.popitem(last=False)
        return block

    def __call__(self, source, evolution='fixed', *, pm_ode_rtol=1e-10, pm_ode_phase_atol=1e-8):
        if evolution=='auto':
            raise ValueError('Resolve auto evolution over the full observation before chunking')
        output = {ch:np.empty(self.times.size) for ch in ('A','E')}
        for start in range(0,self.times.size,self.block_samples):
            stop = min(start+self.block_samples,self.times.size)
            geometry,operator = self._block(start,stop)
            links = eccentric_links_jax(source,geometry,physics_mode='1pn',
                evolution_mode=evolution,batch_size=1,
                pm_ode_rtol=pm_ode_rtol,pm_ode_phase_atol=pm_ode_phase_atol)
            ae = aet_from_xyz(operator(links))
            for ch in output:
                values = np.asarray(ae[ch])[self.halo:-self.halo]
                if np.iscomplexobj(values) and np.any(values.imag != 0):
                    raise ValueError('Native real response has an imaginary component')
                output[ch][start:stop] = values.real
        return AESeries(self.times,output,self.convention)


def weighted_relative_residual(metric: AEInnerProduct, reference, candidate, *, fit_amplitude=False):
    """Weighted norm error on explicitly identical times and conventions."""
    a,b = metric.prepare(reference),metric.prepare(candidate)
    scale = metric.inner(a,b)/metric.inner(b,b) if fit_amplitude else 1.
    residual = a-scale*b
    return float(np.sqrt(metric.inner(residual,residual)/metric.inner(a,a)))


def normalized_overlap_change_bound(relative_reference_error, relative_template_error):
    """Conservative overlap perturbation bound for two waveform norm errors.

    If ||delta x||/||x|| <= eps < 1, normalization changes the unit vector by
    at most 2 eps/(1-eps). The sum bounds an individual overlap change; it
    bounds a family maximum only when the template bound holds uniformly.
    """
    errors = np.asarray([relative_reference_error,relative_template_error],dtype=float)
    if np.any(~np.isfinite(errors)) or np.any(errors < 0) or np.any(errors >= 1):
        raise ValueError('Relative errors must be finite and lie in [0,1)')
    return float(np.sum(2*errors/(1-errors)))
