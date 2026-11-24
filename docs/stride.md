# STRIDE

Sampled Time-domain Response through Interpolated Delay Evaluation.
STRIDE is included in the `egb-multi` distribution; the primary Python interface is
`lisa_stride`. It uses JAX for detector-array preparation, sky projection,
sampled source interpolation, six GW links and ordered Michelson TDI. Numba is
an optional CPU backend. The legacy `xtdi` CPU interface remains available.

For an executed source-to-response walkthrough, open the
[STRIDE and fastlisaresponse usage notebook](../examples/notebooks/stride_fastlisaresponse_usage.ipynb).
It includes source/XYZ plots, changed-sky detector reuse and a same-input
native comparison with an explicit residual.


## Install

Python 3.12 or later:

```sh
python -m pip install .           # NumPy + JAX; no Numba dependency
python -m pip install '.[cpu]'    # optional Numba CPU implementation
python -m pip install '.[test,cpu,orbits]'   # full reference suite, including pyTDI + Numba
```

Select a supported JAX installation for the intended hardware using the
[official instructions](https://docs.jax.dev/en/latest/installation.html).
Explicitly enable float64 before preparation. STRIDE does not change global
JAX precision or select a GPU on import. The CPU reference suite and separately archived A100 studies
qualify specific configurations. GPU accuracy, memory use and speed require
qualification for the intended hardware, source sampling and orders; float32
is not a substitute for that check.

## Six real links to XYZ

```python
import jax
jax.config.update('jax_enable_x64', True)
import numpy as np
from lisa_stride import TDIPlan, LINK_LABELS, xyz_to_aet

t = np.arange(4096, dtype=float) * 2.5  # reception time, seconds
L = {label: np.full(t.size, 8.3) for label in LINK_LABELS}
links = np.array([np.sin(2*np.pi*.003*t+i) for i in range(6)])
plan = TDIPlan.prepare(t, L, measurement_order=25, device='cpu')
xyz = plan.apply(links)               # checked convenience API, JAX XYZ
xyz.block_until_ready()               # completion before measuring elapsed time
aet = xyz_to_aet(xyz)
# Low-level plan.apply_array(links) is JIT-compatible and stays on device.
# TDIPlan.prepare(t, L, backend='numba') selects the optional CPU backend.
```

No waveform generator, ephemeris reader or reference package is imported by
the JAX core. Six-link arrays have shape `(...,6,N)` in receiver/emitter order
`21,32,13,31,23,12`. Output shape is `(...,3,N)`, ordered X,Y,Z. Frequency inputs
may be fractional frequency or Hz if all six links use the same unit. There is
no unit conversion. `unit='frequency'` retains the additive retarded-rate
prescription; `unit='phase'` omits those factors. A/E/T are orthonormal:
A=(Z-X)/sqrt(2), E=(X-2Y+Z)/sqrt(6), T=(X+Y+Z)/sqrt(3).

## Sampled polarizations and supplied detector geometry

`DetectorResponse.prepare(geometry, backend='jax', device='cpu')` is the primary
factory. Supply a dictionary with:

- `times`: uniform increasing reception seconds, shape `(N,)`;
- `n`: unit propagation directions along each directed arm, `(6,N,3)`;
- `length`: nonnegative directed light times in seconds, `(6,N)`;
- `p_sender`, `p_receiver`: Cartesian positions in light-seconds, `(6,N,3)`.

The caller chooses instantaneous or retarded emitter positions and provides a
consistent frame and time origin. No equal-arm, rigid-orbit or harmonic-source
assumption is imposed by these array interfaces. Actual geometry accuracy and
waveform sampling remain the caller's responsibility.

```python
from lisa_stride import DetectorResponse, SampledPolarizations
# geometry, hp, hc, k, plus, cross are supplied by the analysis.
detector = DetectorResponse.prepare(geometry, device='cpu')
source = SampledPolarizations.from_arrays(hp, hc, origin_s=origin, dt_s=dt)
xyz = detector.sampled_response(source, k, plus, cross, order=39)
# k is the GW propagation direction; plus/cross are transverse (3,3) tensors.
```

This checked API synchronizes validity and raises for invalid sky or uncovered
source support. `sampled_response_array(values, origin_s, dt_s, k, plus, cross)`
returns `(xyz, valid)` on device, without a host copy or wait. Invalid traced
inputs produce all-NaN output and `valid=False`; a likelihood must reject that
status. Finite real float64 input is required. Source coverage includes the
whole centered interpolation stencil, with no source extrapolation or zero
padding. TDI's reference out-of-record link boundary is separately zero.

For Numba, prepare with `backend='numba'` and transfer the source once with
`source.to_host()`, or create it using `from_arrays(..., backend='numba')`.
The CPU path expects immutable contiguous NumPy samples. Transfers are explicit
so repeated-call timing does not hide host/device adaptation.

### Evolving eGB sources

For one Julian year at 2.5 s cadence, including 5,000 s of source padding at
each end, the eGB adapter can construct the source grid directly on the device.

```python
from egb_multi.stride import sampled_pm_uniform

sampled = sampled_pm_uniform(source, origin_s=0., dt_s=2.5, size=12_627_040,
                             method="interpolate", device="cpu")
```

The default `method="direct"` evaluates both polarizations at every requested
time. Both methods integrate the same orbit-averaged Peters–Mathews equations
and retain the selected conservative dynamics. The integrator solves for
changes in radial mean motion, time eccentricity and the two secular phases.
It compares solutions at two step sizes throughout each polynomial segment.
A prepared orbit can be queried in blocks using `prepare_pm` and `pm_strain`
in `egb_multi.pm_jax`. Rebuild it when the source parameters or required
time interval change. The checked preparation and refinement run from Python.
The integration steps, dense evaluations and waveform kernels run in JAX.

On a uniform grid, `method="interpolate"` looks for an integer sample spacing
close to a whole number of orbital periods. At a fixed offset within this
spacing the polarizations vary slowly across successive orbits. The code
interpolates this variation separately at every offset, and compares the
interpolant with direct waveform evaluations between every pair of interpolation
nodes. The largest probe discrepancy must be below `1e-11` times the largest
absolute anchor polarization. Smaller blocks or direct evaluation handle cases
that fail the check. The method preserves the requested output cadence. For
the selected method, block length and measured probe error use
`egb_multi.pm_sampling.pm_uniform_polarizations(..., return_info=True)`.
Interpolation is opt-in and validated on CPU. Different interpolation block
shapes can incur new JAX compilation costs during inference.

For supplied uniform timestamps, `sampled_polarizations(..., source_backend="jax")`
defaults to full Peters–Mathews evolution. Set `evolution_mode="fixed"` to hold the
orbital elements constant. The default PM sampler uses direct JAX source evaluation on the
supplied grid. Use `pm_sampling="interpolate"` to request the fast sampler
when the timestamps exactly match their implicit uniform grid. The executable
examples expose the same option as `--pm-sampling interpolate`. The independent SciPy source remains available through
`source_backend="numpy"`. The partial-PM diagnostic prescriptions are
available with that backend. The [annual source benchmark](../tests/benchmarks/pm_annual.py)
measures synchronized fresh waveform calls and compares against SciPy and
direct JAX. Its clocks end at the two source polarizations.

## Reuse, batches and streaming

Detector plans and source states are PyTrees with dynamic array leaves. Pass
them as arguments to an outer `jax.jit`; changing equal-shaped geometry or a
source grid does not bake a new year of arrays into the compiled program.
Only the interpolation orders, TDI generation and execution schedule are
static. Leading source batches share a sky; `jax.vmap` handles different skies.
Every new sky proposal recomputes its actual projection and wavefront times.
Detector reuse does not imply a retained sky cache.

`StreamedDetectorResponse(times, geometry_provider, ...)` uses JAX by default.
The provider receives each padded absolute time block and must return the exact
requested grid. A bounded detector-only LRU cache defaults to 256 MiB and 32,768
retained samples per block. Cache misses include the geometry and TDI plan.
Changing the provider's physical configuration requires a new stream instance.
One mutable stream/cache belongs to one inference worker.

- `iter_sampled` yields checked `(slice, device_xyz)` blocks and synchronizes
  each block. `sampled_response` collects NumPy XYZ under an output byte limit.
- `iter_arrays(..., max_inflight=2)` yields `(slice, device_xyz, valid)` and limits
  pending response work. The consumer must handle validity and any output it
  retains. This bound and the detector cache exclude source tables, compiler
  workspaces and consumer storage.

`execution_schedule='staged'` is the measured CPU default. `'fused'` exposes the
same array calculation in one compiled response. For GPU workloads,
`'fused_unrolled'` expands the fixed interpolation loops within that response,
reducing repeated GPU launches while retaining float64, the same tap
recurrences and source-query arithmetic. Compilation takes longer and is
recorded separately from warm execution. Select and validate this schedule for
the intended source sampling, orders and hardware. Streamed calls upload each
fresh sky/polarization proposal once per call; they do not retain a sky cache.
Device scheduling is JAX's
responsibility; the Numba `parallel=True` setting is not a JAX worker count.
A downstream CUDA consumer that needs a full NumPy result can optionally use
`cupy.asnumpy(cupy.from_dlpack(xyz))`, after checking the response validity.
DLPack shares the existing JAX device buffer; the blocking readback preserves
its dtype and bytes. See `examples/stride/gpu_readback.py`. CuPy is optional and must
match the installed CUDA runtime. It is not imported by STRIDE's core and is
unnecessary when the likelihood remains on the GPU. Measure this output-copy
choice inside the complete host-to-host timing boundary.

For independent sources with distinct skies, the optional streamed batch
helper groups source evaluations while reusing each detector block:

```python
from lisa_stride.batching import sampled_response_batch_array

xyz, valid = sampled_response_batch_array(
    stream, values, origin_s, dt_s, ks, plus, cross, angles,
    microbatch=1, max_inflight=16, loop_order="blocks",
)
# Reject every source with valid == False before using its output.
```

Here the device arrays have shapes `(B,2,M)`, `(B,3)`, `(B,3,3)`, `(B,3,3)`
and `(B,)`; the returned arrays have shapes `(B,3,N)` and `(B,)`. Sources share
one original sampling grid. Place inputs on the selected device before calling;
include those transfers in complete-call timings. The helper recomputes the
response for each sky and polarization proposal. It collects complete output
on device, which is additional to the detector-cache and workspace memory.

The default time-block-first loop reuses detector preparation across sources,
including with a bounded cache. `loop_order="sources"` can repeat detector
preparation when the complete observation does not fit in the cache. Group size
one avoids the large intermediates seen when an outer `vmap` combines many
annual distinct-sky responses. It improved the tested eight-source annual A100
workload with `fused_unrolled`, 32,768-sample blocks and a warm resident detector
cache, while preserving full-record output bytes. This is a measured setting,
not an automatic choice for all devices or batch shapes. The helper coordinates
compiled kernels on the host; the existing array API remains the interface for
composing a whole calculation under `jax.jit`.

The older `xtdi.jax_backend` CPU-pinned channel schedule is a compatibility
component, not the general accelerator path.

The array response supports JIT, vmap and tested derivatives with respect to
source values/amplitude. Host ephemeris readers and external source generators
are separate interfaces. Their presence does not imply an end-to-end JAX
waveform or parameter-gradient calculation.

## Numerical method and validation

Ordered retarded delays, additive retarded rates, factorized sampled stages
and centered Lagrange interpolation retain the reference conventions. Finite
interpolation stages must not be replaced by a collapsed continuous delay.
Numba's ascending tap sum and JAX's paired tap sum can round differently; the
cross-backend contract is quantitative error, not general byte equality.
The explicit float64 rounding boundaries in the JAX sky calculation preserve
late-epoch source times on the qualified CPU domain. They discard no mantissa
bits. Other devices must be checked independently.

```sh
python -m pytest -q tests/stride/test_jax_public.py   # no Numba/pyTDI needed; CPU test skips
python -m pytest -q tests/stride            # install .[test,cpu,orbits] for full references
python examples/stride/sampled_response.py --backend jax
python examples/stride/sampled_response.py --backend numba
```

The reference suite tests analytic equal-arm transfer, generation 1/2, explicit
rates, geometry ownership, invalid support, late epochs, batch skies, derivatives
and bounded streaming. Physical studies separately compare analytic/numerical
orbits and eGB/native Phenomxpy sources with same-input pyTDI and higher-order
refinement. Passing unit tests is separate from those physical comparisons.
This package supplies a GW response and six-link TDI interface; calibration of
SCI/TMI/REF measurements, clock/ranging errors and data gaps belongs to the
surrounding detector analysis.

GPL-2.0-only. See LICENSE and THIRD_PARTY_NOTICES for attribution.

See the [cadence and batching guide](annual_cadence.md#batching-source-parameters-and-sky-directions)
for fixed-sky reuse and independent-sky PM proposals.
