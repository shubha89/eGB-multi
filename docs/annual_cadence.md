# Optional cadence for an annual PM waveform

The default annual grid samples every 2.5 seconds. Choose `"waveform"` to use the empirical cadence for the initial frequency and eccentricity of the source. Both the padded source and the detector must use the selected native spacing.

```python
import jax
import numpy as np
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.stride import annual_pm_grid, recommend_cadence

jax.config.update("jax_enable_x64", True)
source = EccentricBinaryParams(np.pi * .003, .19, m1_solar=1.1, m2_solar=.4,
                               beta=.3, lambda_=1.2, inclination=.9, phi0=.7)
grid = annual_pm_grid(source, cadence_s="waveform")  # 10 s, 3,155,760 samples
# Example with detector and source times expressed relative to the same epoch.
start_s = 5000.
detector_times = start_s + np.arange(grid.samples) * grid.cadence_s
polarizations = grid.sampled_source(source, start_s=start_s, padding_s=4500.)
```

Use the existing STRIDE response with these detector times and polarizations, choosing padding that covers the detector halo, sky delays and source interpolation stencil. If the orbit uses an absolute clock, add the same epoch offset to both detector times and the origin of the sampled polarizations. Prepare a new detector and fixed-sky cache when the grid changes.

For a sampler, choose one cadence for its frequency and eccentricity bounds. For example, `recommend_cadence((3, 6), (.10, .19)).cadence_s` gives 5 s. Supply that value to `annual_pm_grid` for each proposal. Compare templates with the data on common physical frequency bins with the same covariance and normalization. Coarser templates have fewer native samples. Include power beyond their Nyquist frequency when evaluating loss.

Frequency is the package parameter `source.mean_motion / π`, in mHz. With periastron advance enabled, the package interprets the angular frequency as the averaged azimuthal frequency. The calibration uses 19 one-year PM-on sources, 1PN conservative dynamics, component masses 1.1 and 0.4 solar masses, one fixed sky and orientation, numerical directed arm delays with simultaneous emitter positions, and second-generation XYZ with source, measurement and delay orders 39, 25 and 7. The grid covers 1–10 mHz and initial eccentricities 0–0.30.

The waveform rule requires the largest of the three annual, windowed relative XYZ errors to be below 1e-8. The reference cadence is 1.25 s, with additional 0.625 s comparisons at selected points. The likelihood rule (`cadence_s="likelihood"`) requires the fixed-parameter A/E residual SNR to fall below `residual_limit` at signal `snr` (defaults 1 and 100). It uses the calibration's complex A/E covariance over 0.01–80 mHz. Between measured points the lookup intersects passing cadences at the surrounding nodes and available interior checkpoints. An unresolved point returns `None` from the lookup and raises when an automatic grid is requested. An unmeasured prior, sky, mass pair, orbit or noise model needs its own accuracy check.

At signal SNR 100, six measured sources reached the original 80 s ceiling of the A/E test. Extending those sources in 5 s increments gave limits of 155, 110 and 80 s at 1 mHz for initial eccentricities 0.10, 0.19 and 0.30, respectively. The limits were 100 s at 2 mHz and eccentricity 0.05, and 105 s for the circular 3 mHz source. The circular 1 mHz source passed through 240 s, the largest cadence tested. Its residual SNR at that spacing was 0.00449 and its largest channel-relative XYZ error was 4.49e-5. These extended values are available via the likelihood criterion.

The annual comparison excludes half a day at each end and tapers over the following day. For an output grid that includes the observation endpoints, supply orbit and polarization support for the physical span of every delayed stencil. For example, at 240 s cadence the benchmark’s conservative TDI halo alone spans 139 samples (33,360 s) on each side. The source also needs padding for the sky delay and polarization interpolation.

## Batching source parameters and sky directions

`egb_multi.stride.sampled_pm_batch(sources, origin_s, dt_s, size)` generates
double-precision plus and cross polarizations on one padded grid, in input
order. Each source has its own PM evolution. For the 1PN model and default PM
tolerances the experimental solver compares two resolutions for each orbit.
If a source needs more integration segments or a different phase treatment,
the group uses the standard adaptive solver.

- **One sky.** Prepare `FixedSkyResponse` once for the stream, direction,
  polarization basis and padded source grid. Call `apply(sampled, angles)`
  with a `SampledPolarizations` batch in the detector time coordinate for
  checked grid alignment and validity. The lower-level `apply_array(values, angles)`
  accepts samples on that exact grid and returns explicit validity flags. The retained arrays
  include source interpolation coordinates at the link emission and reception
  times. Use a fresh plan when the sky or source grid changes.
- **Different skies.** Use `lisa_stride.batching.sampled_response_batch_array`
  with the same detector stream and one direction, basis and polarization angle
  per source. All sources share the detector orbit and ordered TDI delays. Each
  direction gets its own projection. Both array interfaces return XYZ with shape
  `(sources, 3, detector_samples)` and one validity flag per source. Check every
  flag before using the output.

Group proposals on a common source and detector grid. If the polarizations
use time relative to a reference epoch, add that epoch to their grid origin
when passing it to a detector expressed in absolute time.
[Runnable example for both sky modes](../tests/benchmarks/annual_pm_batch.py)

## Annual A100 measurements

On one A100-SXM4 80 GB, six fixed-sky sources at 1–3 mHz took 6.27–8.60 ms
each with the standard PM solver and the likelihood cadence, compared with
93.74–96.50 ms at 2.5 s. The timer started from binary parameters and stopped
when the full year's XYZ was ready on the GPU. Detector and sky preparation
took 2.96–4.02 s per adaptive grid. Compilation, transfers back to the host
and likelihood evaluation were outside the warm timer.

The vectorized PM prototype took 2.87 ms for one circular 1 mHz waveform at
240 s cadence, or 10.09 ms for four together. At 1 mHz and initial eccentricity
0.30 it took 6.18 ms for one waveform and 18.43 ms for four at 80 s cadence.
Batching improved throughput in four of the six source families. The largest
per-channel relative difference between a batch and individual calls on the
same grid was 4.71e-14. Against a 1.25 s reference the six nominal sources had
A/E residual SNRs of 0.00449–0.961 at signal SNR 100, with unweighted XYZ
relative errors reaching 0.0335. The [six-source timing and accuracy table](../tests/benchmarks/annual_pm_a100.csv)
contains the measured medians from the prototype used for the package implementation.

A separate annual A100 test started with the polarizations already on the GPU
at 2.5 s cadence. Eight distinct skies took 2.734 s through the six links and
TDI, and eight proposals sharing a prepared sky took 0.890 s. These numbers
measure the response stage. End-to-end A100 timings at adaptive cadence and
noise-weighted cadence validation across different skies remain open.

The installed release wheel was also checked on a CPU with the same numerical
orbit. Four one-year sources at 1 mHz and eccentricity 0.19 shared one sky and
a 110 s grid. Two circular 1 mHz sources used different skies and a 240 s grid.
The largest channel-relative XYZ differences from separate calls at the same
cadence were 1.12e-14 and 7.52e-15, respectively.
