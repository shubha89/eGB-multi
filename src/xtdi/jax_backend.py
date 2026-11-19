"""Optional array-native JAX execution of a verified detector plan.

No global JAX precision setting is changed by this library. Enable x64 before
constructing the backend. `apply_array` stays in JAX and can be differentiated
with respect to input link samples; the detector plan itself is fixed.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .michelson import link_array


@lru_cache(maxsize=32)
def _compiled(generation, order, frequency):
    import jax
    import jax.numpy as jnp

    half = (order+1)//2
    @jax.jit
    def apply(values, parameters):
        centers, fractions, scales, rates, ratios = parameters
        size = values.shape[-1]
        def delay(data, index):
            center, fraction, scale = centers[index], fractions[index], scales[index]
            def sample(offset):
                indices = center+offset
                return jnp.where((indices >= 0) & (indices < size),
                                 jnp.take(data, jnp.clip(indices, 0, size-1), axis=-1), 0.)
            shifted = ((1-fraction)*sample(0)+fraction*sample(1))*scale
            def add_pair(j, total):
                factor = fraction*(1-fraction)*ratios[j]*scale
                return total + factor*(sample(-j)/(j+fraction)+sample(1+j)/(j+1-fraction))
            shifted = jax.lax.fori_loop(1, half, add_pair, shifted)
            return (1+rates[index])*shifted if frequency else shifted
        channels = []
        # Input order 21,32,13,31,23,12. Plan operator order is documented in
        # prepare_michelson; each channel has a,c,ab,cd[,cdab,abcd].
        for channel, (a,b,c,d) in enumerate(((5,0,2,3),(4,1,0,5),(3,2,1,4))):
            base = channel*(6 if generation == 2 else 4)
            arm_a = values[...,a,:]+delay(values[...,b,:],base)
            arm_b = values[...,c,:]+delay(values[...,d,:],base+1)
            if generation == 1:
                result = -arm_a+delay(arm_a,base+3)
                result = result+arm_b
                result = result+(-delay(arm_b,base+2))
            else:
                round_a = arm_a+delay(arm_b,base+2)
                round_b = delay(arm_a,base+3)+arm_b
                result = -round_a+delay(round_a,base+4)
                result = result+round_b
                result = result+(-delay(round_b,base+5))
            channels.append(result)
        return jnp.stack(channels,axis=-2)
    return apply


@lru_cache(maxsize=32)
def _compiled_cpu_channels(generation, order, frequency):
    """Batch independent XYZ channels through each ordered sampled stage.

    This is an opt-in CPU schedule. The existing reference JAX schedule and
    Numba execution remain available; workload qualification is separate from
    numerical validity. Interpolation uses the existing JAX float64 formula.
    """
    import jax
    import jax.numpy as jnp
    half=(order+1)//2
    channel_links=np.array(((5,0,2,3),(4,1,0,5),(3,2,1,4)))
    def apply(values, parameters):
        centers,fractions,scales,rates,ratios=parameters
        size=values.shape[-1];operators=6 if generation==2 else 4
        centers=centers.reshape((3,operators,size))
        fractions=fractions.reshape((3,operators,size))
        scales=scales.reshape((3,operators,size))
        rates=rates.reshape((3,operators,size))
        channel_values=jnp.moveaxis(values[...,channel_links,:],-3,0)
        def channel(streams,cc,ff,ss,rr):
            def delay(data,index):
                center,fraction,scale=cc[index],ff[index],ss[index]
                def sample(offset):
                    indices=center+offset
                    return jnp.where((indices>=0)&(indices<size),
                        jnp.take(data,jnp.clip(indices,0,size-1),axis=-1),0.)
                shifted=((1.-fraction)*sample(0)+fraction*sample(1))*scale
                def add_pair(j,total):
                    factor=fraction*(1.-fraction)*ratios[j]*scale
                    return total+factor*(sample(-j)/(j+fraction)+sample(1+j)/(j+1.-fraction))
                shifted=jax.lax.fori_loop(1,half,add_pair,shifted,unroll=1)
                return (1.+rr[index])*shifted if frequency else shifted
            arm_a=streams[...,0,:]+delay(streams[...,1,:],0)
            arm_b=streams[...,2,:]+delay(streams[...,3,:],1)
            if generation==1:
                result=-arm_a+delay(arm_a,3)
                result=result+arm_b
                result=result+(-delay(arm_b,2))
            else:
                round_a=arm_a+delay(arm_b,2)
                round_b=delay(arm_a,3)+arm_b
                result=-round_a+delay(round_a,4)
                result=result+round_b
                result=result+(-delay(round_b,5))
            return result
        result=jax.vmap(channel)(channel_values,centers,fractions,scales,rates)
        return jnp.moveaxis(result,0,-2)
    sharding=jax.sharding.SingleDeviceSharding(jax.devices('cpu')[0])
    return jax.jit(apply,in_shardings=(sharding,(sharding,)*5),
        out_shardings=sharding)


def _owned_stencil_matrix(operators, field):
    """Reuse an immutable packed owner; otherwise make an owned frozen copy."""
    arrays=[getattr(op.stencil,field) for op in operators]
    base=arrays[0].base
    if (isinstance(base,np.ndarray) and base.shape==(len(arrays),arrays[0].size)
            and base.flags.c_contiguous and not base.flags.writeable
            and all(a.dtype==base.dtype and a.strides==(a.itemsize,)
                    and a.ctypes.data==base.ctypes.data+i*a.nbytes
                    for i,a in enumerate(arrays))):
        return base
    result=np.stack(arrays);result.flags.writeable=False
    return result


@dataclass(frozen=True, eq=False)
class JAXMichelsonPlan:
    plan: object
    parameters: tuple
    schedule: str = "reference"

    @classmethod
    def from_plan(cls, plan, *, schedule="reference"):
        import jax
        import jax.numpy as jnp
        if not jax.config.x64_enabled:
            raise ValueError("xTDI requires JAX x64; enable jax_enable_x64 before creating a JAX plan")
        if schedule not in ("reference", "cpu_channels"):
            raise ValueError("schedule must be reference or cpu_channels")
        operators = plan.operators
        # Cumulative side ratios give the same Lagrange weights, with a
        # different float64 reduction order checked against both references.
        ratios = np.cumprod(np.r_[1., operators[0].stencil.side_coefficients[1:]])
        if schedule == "cpu_channels":
            arrays=[_owned_stencil_matrix(operators,field)
                    for field in ("centers","fractions","scales")]
            arrays += [np.stack([op.rate for op in operators]),ratios]
            for array in arrays:array.flags.writeable=False
            device=jax.devices("cpu")[0]
            parameters=tuple(jax.device_put(array,device,may_alias=True) for array in arrays)
        else:
            parameters = tuple(jnp.asarray(array) for array in (
                np.stack([op.stencil.centers for op in operators]),
                np.stack([op.stencil.fractions for op in operators]),
                np.stack([op.stencil.scales for op in operators]),
                np.stack([op.rate for op in operators]), ratios))
        return cls(plan, parameters, schedule)

    @property
    def nbytes(self):
        """Host plan plus device parameter storage; excludes compiler workspace."""
        return self.plan.nbytes+sum(array.nbytes for array in self.parameters)

    def apply_array(self, links, *, unit="frequency"):
        """Array-only JAX path; finite-input checking belongs to the caller.

        Accepts NumPy/JAX arrays of shape (...,6,N), returns a JAX array.
        No host conversion or synchronization of link values is performed.
        """
        import jax
        import jax.numpy as jnp
        if not jax.config.x64_enabled:
            raise ValueError("xTDI requires JAX x64 during application")
        if unit not in ("frequency", "phase"):
            raise ValueError("unit must be 'frequency' or 'phase'")
        values = (jax.device_put(links,self.parameters[0].device)
                  if self.schedule == "cpu_channels" else jnp.asarray(links))
        if values.ndim < 2 or values.shape[-2:] != (6, self.plan.times.size):
            raise ValueError("links must have shape (...,6,N) on the prepared grid")
        if values.dtype != jnp.float64:
            raise ValueError("JAX links must be real float64")
        compiler=(_compiled_cpu_channels if self.schedule == "cpu_channels" else _compiled)
        return compiler(self.plan.generation, self.plan.measurement_order,
                        unit == "frequency")(values, self.parameters)

    def apply(self, links, *, unit="frequency"):
        """Checked host-input convenience path; result remains a JAX array."""
        values = link_array(links, self.plan.times.size, name="eta/GW links")
        return self.apply_array(values, unit=unit)

    def __call__(self, links, *, unit="frequency"):
        return self.apply(links, unit=unit)

    def apply_mapping(self, links, *, unit="frequency"):
        values = self.apply(links, unit=unit)
        return {channel: values[...,i,:] for i, channel in enumerate("XYZ")}
