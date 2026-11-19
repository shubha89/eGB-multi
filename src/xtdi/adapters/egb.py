"""Opt-in eGB integration; importing this module requires egb_multi.

No source physics or existing eGB default is changed. The native chunked
response supplies exact retarded geometry and absolute source epochs.
"""

from collections import OrderedDict

import numpy as np

from xtdi import prepare_michelson
from xtdi.streaming import conservative_halo


class XTDIGWTDIMichelson:
    """Drop-in GW-only callable compatible with PreparedGWTDIMichelson."""

    def __init__(self, state, link_delays, *, generation=2,
                 measurement_order=101, delay_order=7, backend="numba"):
        if backend not in ("numba", "jax"):
            raise ValueError("backend must be numba or jax")
        plan = prepare_michelson(state.t, link_delays, generation=generation,
            measurement_order=measurement_order, delay_order=delay_order)
        self.plan = plan.to_jax() if backend == "jax" else plan

    def __call__(self, links):
        real_links = {}
        for label, value in links.items():
            array = np.asarray(value)
            if np.iscomplexobj(array):
                if np.any(array.imag != 0):
                    raise ValueError("eGB GW TDI requires real links (zero imaginary part)")
                array = array.real
            real_links[label] = array
        return self.plan.apply_mapping(real_links)


# This optional subclass reuses eGB's source generation and output checking.
# Only the detector block preparation replaces pyTDI; there is no monkeypatch.
from egb_multi.native_validation import ChunkedNativeResponse


class XTDIChunkedNativeResponse(ChunkedNativeResponse):
    def __init__(self, *args, backend="numba", **kwargs):
        super().__init__(*args, **kwargs)
        if backend not in ("numba", "jax"):
            raise ValueError("backend must be numba or jax")
        self.backend = backend
        self.halo = conservative_halo(self.dt, self.delay_bound_s,
            self.convention.measurement_order, self.convention.delay_order,
            self.convention.generation)
        self.cache = OrderedDict()

    def _block(self, start, stop):
        from egb_multi.external import state_from_lisaorbits, link_delays_from_lisaorbits
        from egb_multi.eccentric_jax import precompute_jax_link_geometry
        key = start, stop
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        grid = self.times[0]+np.arange(start-self.halo,stop+self.halo)*self.dt
        state = state_from_lisaorbits(self.orbit,grid,frame="ecliptic")
        geometry = precompute_jax_link_geometry(state,orbits=self.orbit,frame="ecliptic")
        delays = link_delays_from_lisaorbits(self.orbit,grid)
        if any(np.any(d <= 0) or np.any(d > self.delay_bound_s) for d in delays.values()):
            raise ValueError("actual delay violates the declared padding bound")
        operator = XTDIGWTDIMichelson(state,delays,generation=self.convention.generation,
            measurement_order=self.convention.measurement_order,
            delay_order=self.convention.delay_order,backend=self.backend)
        block = geometry, operator
        if self.cached_blocks:
            self.cache[key] = block
            while len(self.cache) > self.cached_blocks:
                self.cache.popitem(last=False)
        return block
