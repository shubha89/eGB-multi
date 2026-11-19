"""Generator-independent device-resident source state with dynamic grid leaves."""
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from xtdi.common import check_order,real_array
from xtdi.jax_response import _require_x64,_device

@dataclass(frozen=True,eq=False)
class SampledPolarizations:
    """Plus/cross strain (...,2,M), origin and cadence in seconds.

    Direct construction is traceable; use from_arrays for checked host setup.
    The interpolation order is static. Values and both grid scalars are PyTree
    leaves, so equal-shaped parameter proposals reuse an outer compilation.
    """
    origin_s: object
    dt_s: object
    values: object
    order: int=39

    @classmethod
    def from_arrays(cls,hplus,hcross,*,origin_s,dt_s,order=39,device=None,backend="jax"):
        order=check_order(order)
        hp,hc=real_array(hplus,name='hplus'),real_array(hcross,name='hcross')
        if hp.ndim<1 or hc.shape!=hp.shape or hp.shape[-1]<order+1:
            raise ValueError('aligned plus/cross arrays covering the stencil required')
        if not np.isfinite(origin_s) or not np.isfinite(dt_s) or dt_s<=0:
            raise ValueError('finite origin and positive cadence required')
        if backend not in ('jax','numba'):raise ValueError('backend must be jax or numba')
        if backend=='numba':
            values=np.stack((hp,hc),axis=-2);values.flags.writeable=False
            return cls(np.float64(origin_s),np.float64(dt_s),values,order)
        _require_x64()
        target=_device(device)
        return cls(jax.device_put(np.float64(origin_s),target),
                   jax.device_put(np.float64(dt_s),target),
                   jax.device_put(np.stack((hp,hc),axis=-2),target),order)

    def to_host(self):
        """Explicit one-time transfer for the optional Numba CPU backend."""
        values=np.array(jax.device_get(self.values),copy=True);values.flags.writeable=False
        return type(self)(np.float64(self.origin_s),np.float64(self.dt_s),values,self.order)

    @property
    def nbytes(self):return self.values.nbytes+self.origin_s.nbytes+self.dt_s.nbytes

jax.tree_util.register_dataclass(SampledPolarizations,
    data_fields=['origin_s','dt_s','values'],meta_fields=['order'])
