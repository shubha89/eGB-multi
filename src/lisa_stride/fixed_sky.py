"""Reuse one sky projection and source grid for batches of fractional-frequency XYZ."""
from collections import deque
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from xtdi import jax_response as response
from xtdi.common import check_order
from .sampled import SampledPolarizations


@partial(jax.jit, static_argnames=("samples", "order"))
def _prepare_block(geometry, direction, plus, cross, origin, step, samples, order):
    (queries, a, b, denominator), valid = response._project(geometry, direction, plus, cross, 0.)
    centers, fraction, supported = response._source_coordinates(queries, origin, step, samples, order)
    scale = response._scale(fraction, (order + 1) // 2, unroll=True)
    return (centers, fraction, scale, a, b, denominator), valid & supported


@partial(jax.jit, static_argnames=("order", "generation", "measurement_order"))
def _evaluate_block(plan, parameters, values, angles, order, generation, measurement_order):
    centers, fraction, scale, a, b, denominator = plan

    def one(polarizations, psi):
        strain = response._paired_interpolate(polarizations, centers, fraction, scale, order, unroll=True)
        cosine, sine = jnp.cos(2 * psi), jnp.sin(2 * psi)
        plus = response._sub(response._mul(cosine, a), response._mul(sine, b))
        cross = response._add(response._mul(sine, a), response._mul(cosine, b))
        links = response._projected_links(strain, plus, cross, denominator)
        xyz = response._tdi(links, parameters, generation, measurement_order, True, unroll=True)
        valid = jnp.all(jnp.isfinite(links)) & jnp.all(jnp.isfinite(xyz))
        return jnp.where(valid, xyz, jnp.nan), valid

    return jax.vmap(one)(values, angles)


@dataclass(frozen=True, eq=False)
class FixedSkyResponse:
    """Prepared JAX detector and exact sky on a specified, padded source grid.

    Each retained time block stores both the TDI plan and source interpolation
    coordinates. Rebuild when detector, direction, basis or source grid changes.
    Source polarizations and the polarization angle can vary per proposal.
    """
    _blocks: tuple
    origin_s: float
    dt_s: float
    _samples: int
    _order: int
    _halo: int
    _generation: int
    _measurement_order: int

    @classmethod
    def prepare(cls, stream, direction, plus, cross, *, origin_s, dt_s, samples, order=39):
        """Prepare all detector blocks. origin_s must use the detector time coordinate."""
        if not isinstance(stream, response.JAXStreamedDetectorResponse):
            raise TypeError("A JAX StreamedDetectorResponse is required")
        order = check_order(order)
        if (not isinstance(samples, (int, np.integer)) or samples < order + 1
                or not np.isfinite(origin_s) or not np.isfinite(dt_s) or dt_s <= 0):
            raise ValueError("Supply a finite source grid with positive cadence and full stencil support")
        target = response._device(stream.device)
        sky = [jax.device_put(jnp.asarray(x), target) for x in (direction, plus, cross)]
        if [x.shape for x in sky] != [(3,), (3, 3), (3, 3)] or any(x.dtype != jnp.float64 for x in sky):
            raise ValueError("One float64 direction and two float64 polarization tensors are required")
        blocks = []
        for start in range(0, stream.times.size, stream.block_samples):
            count = min(stream.block_samples, stream.times.size - start)
            detector, *_ = stream._entry(start, count)
            plan, valid = _prepare_block(detector.geometry, *sky, origin_s, dt_s, samples, order)
            if not bool(valid):
                raise ValueError("Invalid sky or insufficient padded source grid for this detector")
            blocks.append((count, jax.block_until_ready(plan), detector.parameters))
        return cls(tuple(blocks), float(origin_s), float(dt_s), samples, order,
                   stream.halo, stream.generation, stream.measurement_order)

    def apply(self, sampled, angles):
        """Check the source grid and validity, then return batched XYZ on device."""
        if not isinstance(sampled, SampledPolarizations):
            raise TypeError("A SampledPolarizations batch is required")
        if (float(sampled.origin_s) != self.origin_s or float(sampled.dt_s) != self.dt_s
                or sampled.order != self._order):
            raise ValueError("Sampled source and fixed-sky plan must share origin, cadence and order")
        xyz, valid = self.apply_array(sampled.values, angles)
        if not bool(jnp.all(valid)):
            raise ValueError("Invalid source polarizations or angles in the fixed-sky batch")
        return xyz

    def apply_array(self, values, angles):
        """Use (B,2,M) FP64 samples on the prepared grid and B angles in radians.

        Returns device-resident (B,3,N) XYZ and one validity flag per source.
        Callers must check all flags or reject the affected proposals.
        """
        values, angles = jnp.asarray(values), jnp.asarray(angles)
        if (values.ndim != 3 or values.shape[0] == 0 or values.shape[1:] != (2, self._samples)
                or angles.shape != (values.shape[0],)):
            raise ValueError("Supply one polarization pair on the prepared grid and one angle per source")
        if values.dtype != jnp.float64 or angles.dtype != jnp.float64:
            raise ValueError("Source values and angles must be float64")
        pieces, flags, pending = [], [], deque()
        max_pending = max(1, 16 // values.shape[0])
        for count, plan, parameters in self._blocks:
            if len(pending) >= max_pending:
                jax.block_until_ready(pending.popleft())
            xyz, valid = _evaluate_block(plan, parameters, values, angles, self._order,
                                         self._generation, self._measurement_order)
            piece = xyz[..., self._halo:self._halo + count]
            pieces.append(piece)
            flags.append(valid)
            pending.append((piece, valid))
        return jnp.concatenate(pieces, axis=-1), jnp.all(jnp.stack(flags), axis=0)
