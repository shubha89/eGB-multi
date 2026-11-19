"""Prepared native-factorized Michelson TDI from real eta/GW link streams.

Same discrete stage graph as pyTDI's compute_factorized_michelson.
Nested shift derivatives deliberately retain pyTDI's additive retarded-rate
prescription. This module does not perform SCI/TMI/REF metrology reduction.
See THIRD_PARTY_NOTICES for the reference implementation attribution.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib

import numpy as np

from .delay import FractionalDelay, check_order, real_array, readonly, timeshift

LINK_LABELS = ("21", "32", "13", "31", "23", "12")
_CHANNEL_LINKS = (("12", "21", "13", "31"),
                  ("23", "32", "21", "12"),
                  ("31", "13", "32", "23"))


def validate_times(times):
    times = real_array(times, name="times")
    if times.ndim != 1 or times.size < 4:
        raise ValueError("at least four uniformly spaced times in seconds are required")
    steps = np.diff(times)
    if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-9, atol=1e-10):
        raise ValueError("times must be increasing and uniformly sampled")
    return times, float(steps[0])


def link_array(values, size, *, name):
    if isinstance(values, Mapping):
        if set(values) != set(LINK_LABELS):
            raise ValueError(f"{name} must have exactly these link labels: {LINK_LABELS}")
        values = np.stack([real_array(values[label], name=name) for label in LINK_LABELS], axis=-2)
    array = real_array(values, name=name)
    if array.ndim < 2 or array.shape[-2:] != (6, size):
        raise ValueError(f"{name} shape must be (..., 6, {size}) in LINK_LABELS order")
    return array


@dataclass(frozen=True, eq=False)
class DelayOperator:
    chain: tuple[str, ...]
    shift_s: np.ndarray
    rate: np.ndarray
    stencil: FractionalDelay

    def apply(self, data, unit):
        shifted = self.stencil._apply_array(data)
        return shifted if unit == "phase" else (1.0 + self.rate) * shifted

    @property
    def nbytes(self):
        return self.shift_s.nbytes + self.rate.nbytes + self.stencil.nbytes


@dataclass(frozen=True, eq=False)
class MichelsonPlan:
    """Immutable fixed-grid detector plan, reusable for different waveforms.

    Inputs are real eta_ij streams or GW-only sci_ij with zero local metrology.
    Arrays use (..., 6, N); mappings use receiver/emitter labels ij. Outputs
    have shape (..., 3, N), ordered X,Y,Z. `unit='frequency'` includes the
    native Doppler factors and supports Hz or fractional-frequency inputs
    provided all six streams use the same units. No conversion is performed.
    """

    times: np.ndarray
    dt: float
    generation: int
    measurement_order: int
    delay_order: int
    operators: tuple[DelayOperator, ...]
    fingerprint: str

    @property
    def nbytes(self):
        return self.times.nbytes + sum(op.nbytes for op in self.operators)

    def operator(self, chain):
        chain = tuple(chain)
        for op in self.operators:
            if op.chain == chain:
                return op
        raise KeyError(chain)

    def apply(self, links, *, unit="frequency", return_stages=False):
        if unit not in ("phase", "frequency"):
            raise ValueError("unit must be 'phase' or 'frequency'")
        array = link_array(links, self.times.size, name="eta/GW links")
        values = {label: array[..., i, :] for i, label in enumerate(LINK_LABELS)}
        outputs, stages = [], {}
        for channel, (a, b, c, d) in zip("XYZ", _CHANNEL_LINKS, strict=True):
            delay = lambda data, chain: self.operator(chain).apply(np.ascontiguousarray(data), unit)
            arm_a = values[a] + delay(values[b], (a,))
            arm_b = values[c] + delay(values[d], (c,))
            if return_stages:
                stages[channel + ".arms"] = (arm_a, arm_b)
            if self.generation == 1:
                # Preserve native polynomial accumulation, including signs.
                result = -arm_a + delay(arm_a, (c, d))
                result = result + arm_b
                result = result + (-delay(arm_b, (a, b)))
            else:
                round_a = arm_a + delay(arm_b, (a, b))
                round_b = delay(arm_a, (c, d)) + arm_b
                if return_stages:
                    stages[channel + ".rounds"] = (round_a, round_b)
                result = -round_a + delay(round_a, (c, d, a, b))
                result = result + round_b
                result = result + (-delay(round_b, (a, b, c, d)))
            outputs.append(result)
        xyz = np.stack(outputs, axis=-2)
        return (xyz, stages) if return_stages else xyz

    def __call__(self, links, *, unit="frequency"):
        return self.apply(links, unit=unit)

    def apply_mapping(self, links, *, unit="frequency"):
        xyz = self.apply(links, unit=unit)
        return {channel: xyz[..., index, :] for index, channel in enumerate("XYZ")}

    def to_jax(self, *, schedule="reference"):
        """Optional array-native backend; enable JAX float64 first."""
        from .jax_backend import JAXMichelsonPlan
        return JAXMichelsonPlan.from_plan(self, schedule=schedule)


def prepare_michelson(times, delays_s, *, delay_derivatives=None, generation=2,
                       measurement_order=101, delay_order=7):
    """Build without pyTDI; delays in seconds on the reception-time grid.

    `delays_s` and optional d(delay)/dt arrays use shape (6,N) or link mappings.
    Absent derivatives use numpy.gradient, matching pyTDI Data's prescription.
    Zero delays are allowed for analytic controls; negative delays are rejected.
    """
    times, dt = validate_times(times)
    fs = 1.0 / dt
    if generation not in (1, 2) or isinstance(generation, bool):
        raise ValueError("generation must be 1 or 2")
    measurement_order, delay_order = check_order(measurement_order), check_order(delay_order)
    delays = link_array(delays_s, times.size, name="delays")
    if delays.ndim != 2 or np.any(delays < 0):
        raise ValueError("delays must be nonnegative and unbatched")
    rates = np.gradient(delays, dt, axis=-1) if delay_derivatives is None else link_array(
        delay_derivatives, times.size, name="delay derivatives")
    if rates.ndim != 2:
        raise ValueError("delay derivatives must be unbatched")
    atomic = {label: (-delays[i], -rates[i]) for i, label in enumerate(LINK_LABELS)}
    required = []
    for a, b, c, d in _CHANNEL_LINKS:
        required.extend([(a,), (c,), (a, b), (c, d)])
        if generation == 2:
            required.extend([(c, d, a, b), (a, b, c, d)])
    prepared = []
    for chain in dict.fromkeys(required):
        shift = np.zeros(times.size)
        rate = np.zeros(times.size)
        for atom in chain:
            atom_shift, atom_rate = atomic[atom]
            # Geometry and rates are phase-like functions: no Doppler factor
            # here. Add retarded atomic rates exactly as the reference does.
            rate += timeshift(atom_rate, shift * fs, delay_order)
            shift += timeshift(atom_shift, shift * fs, delay_order)
        prepared.append(DelayOperator(chain, readonly(shift), readonly(rate),
            FractionalDelay.prepare(shift * fs, times.size, measurement_order)))
    digest = hashlib.sha256()
    for value in (times, delays, rates):
        digest.update(value.tobytes())
    digest.update(repr((generation, measurement_order, delay_order,
                        LINK_LABELS, "native-additive-retarded-rates-v1")).encode())
    return MichelsonPlan(readonly(times), dt, generation, measurement_order,
                         delay_order, tuple(prepared), digest.hexdigest())


def xyz_to_aet(xyz):
    """Common orthonormal A/E/T, preserving real time-domain units."""
    xyz = real_array(xyz, name="XYZ")
    if xyz.ndim < 2 or xyz.shape[-2] != 3:
        raise ValueError("XYZ must have shape (..., 3, N)")
    x, y, z = [xyz[..., i, :] for i in range(3)]
    return np.stack(((z-x)/np.sqrt(2.), (x-2*y+z)/np.sqrt(6.),
                     (x+y+z)/np.sqrt(3.)), axis=-2)
