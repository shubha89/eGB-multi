# JAXGB / eGB-multi CPU wall-time benchmark

For differences from arXiv:2608.24546 Figure 8, see the
[paper comparison and measured diagnostic](results/PAPER_COMPARISON.md).

This measures **native XYZ-generation workloads**, not equal-accuracy or
identical-output implementations. JAXGB is a circular baseline; eccentricity
is varied only in eGB-multi. No speed ratio should be interpreted as the cost
of eccentricity alone.

## Configuration

- One source at a time; central GW reference frequencies 1, 3, and 10 mHz.
- eGB eccentricity: 0, 0.1, ..., 0.6; `mean_motion = pi * frequency`.
  At nonzero eccentricity this is a reference carrier frequency, not necessarily
  the frequency of the strongest harmonic.
- Duration: 365.25 days (31,557,600 s).
- CPU backend, JAX float64; no concurrent numerical benchmark processes.
- Masses 0.6 and 0.4 solar masses; distance 1 kpc; inclination 0.8 rad;
  sky longitude/latitude 0.4/0.2 rad; polarization 0.3 rad.
- eGB: `1pn_periastron`, fixed eccentricity, `fdot = 0`; **no Peters–Mathews
  evolution**. This is a controlled runtime experiment, not an assertion that
  radiation reaction is negligible for every one-year source.
- Equal-arm `lisaorbits` model in both implementations. Each package retains
  its own response implementation and approximations.
- TDI generation 2, XYZ channels. eGB uses pyTDI interpolation order 3.
- eGB: 5 s cadence, 6,311,520 retained time samples per channel; batch size 1;
  65,536 retained samples per block with 128 halo samples at both ends.
  The last block is padded to the same size to avoid recompilation. There are
  97 blocks and 6,381,824 calculated samples, including halos/padding.
- JAXGB: 2,048 slow-response samples / Fourier bins per source, rectangular
  window. Returned XYZ arrays remain narrow-band frequency-domain arrays;
  reconstruction to dense time series is **not included**.

## Timing boundaries

`perf_counter()` measures wall time, not CPU process time. eGB's reported
`total_s` is the sum of the measured geometry, link, and pyTDI wall-time
intervals over the full observation. Output finiteness checks, RMS checksums,
JSON writing, package imports, and benchmark orchestration are outside those
intervals. Full waveforms are generated block by block, but not written to disk.

Geometry is recomputed for each eGB block and timed separately, permitting
the waveform/response cost to be inspected without that stage. JAXGB caches
geometry in its constructor; its constructor time is reported separately.
These choices represent memory-bounded eGB use and cached JAXGB use, rather
than an assertion that geometry caching is impossible for eGB.

JAXGB's `get_tdi` is explicitly JIT-compiled. `block_until_ready` waits for each
completed output before stopping the timer. eGB uses its existing JIT kernels;
its public wrapper converts results to NumPy, synchronizing execution.

Cold costs are recorded separately: JAXGB setup, explicit compilation, first
execution; eGB first full-sized block, whose link time includes compilation
and first execution. The eGB first-block number is **not** a standalone compiler
measurement or the time of a cold full-year run.

Three full-year eGB repeats per case are run in seeded randomized order;
JAXGB has 20 warm repeats per frequency. The figure uses medians and the
observed minimum/maximum of eGB repeats, not confidence intervals.

## Reproduce

The run used eGB commit `26fa9b9` and JAXGB commit `ae31825`.
Exact installed numerical-library versions and platform/device information
are in `results/wall_time.json`; `requirements-lock.txt` preserves the installed
dependency snapshot. For an exact dependency replay, install that snapshot
before installing this checkout with `--no-deps`.

```sh
git clone https://gitlab.com/lisamission/jaxgb.git /tmp/jaxgb-benchmark
git -C /tmp/jaxgb-benchmark checkout ae31825
uv venv --python 3.12 .venv-benchmark
uv pip install --python .venv-benchmark/bin/python -e '.[test,orbits]' matplotlib /tmp/jaxgb-benchmark
MPLCONFIGDIR=/tmp/egb-benchmark-mpl .venv-benchmark/bin/python benchmarks/wall_time.py
MPLCONFIGDIR=/tmp/egb-benchmark-mpl .venv-benchmark/bin/python benchmarks/check_benchmark.py
.venv-benchmark/bin/python benchmarks/summarize.py
```

## Interpretation and checks

**Measured resolution limitation:** on the 10 mHz, e=0.6 check segment,
5 s versus 2.5 s cadence changes XYZ by 11.17% in relative L2 norm, with
pyTDI's default order-3 interpolation. Consequently the 5 s timings are a
fixed-configuration workload benchmark, **not a converged-accuracy result**
for the hardest source. No equivalent accuracy between packages is claimed.
JAXGB's 2,048 versus 4,096 sample check gives common-band relative L2 changes
of 1.406%, 0.871%, and 0.00132% at 1, 3, and 10 mHz respectively.
These checks compare resolutions; neither finer calculation is declared exact.

Padded blocks and a contiguous calculation agreed exactly on the retained
test samples. The tested direct-link imaginary component was exactly zero.
All 11 existing repository tests passed; one pyTDI dtype warning was emitted.

The source model and required number of samples differ substantially. A nearly
flat runtime curve versus eccentricity is plausible: eGB's direct kernel uses
the same array sizes and a fixed Kepler-solver operation sequence for all
eccentricities. This does not imply that eccentric waveforms have the same
bandwidth as circular ones.

`check_benchmark.py` checks padded blocking against one contiguous calculation,
5 s versus 2.5 s eGB cadence at the most demanding source, and JAXGB's 2,048
versus 4,096 sample resolution with physical Fourier bins aligned. These checks
do not establish cross-model waveform agreement, equivalent accuracy, or
convergence over every parameter choice.

The installed pyTDI emits a complex-to-real cast warning. The eGB direct link
response is real-valued but stored in complex arrays; the validation script
checks that its imaginary component is zero for the high-eccentricity case.
The package also emits a lisaconstants/Astropy vacuum-permeability compatibility
warning; the benchmark does not use that electromagnetic constant.
