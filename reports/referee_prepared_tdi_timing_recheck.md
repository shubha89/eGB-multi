# Prepared-TDI branch reconciliation and timing recheck

14 October 2026. The fix branch incorporates main commits `5b91c7c` and `6b83fe7`. The reusable-TDI API and waveform notebooks are preserved alongside the convention/noise fixes. The prepared operator accepts explicit native retarded link delays; the general instrument-data path retains SCI/TMI/REF.

The original one-year benchmark was rerun with its three cases, three warm repeats, 5 s cadence, generation 2, orders 3/3 and bounded block-first cache. The benchmark environment matches the saved JAX, NumPy, SciPy, PyTDI and lisaorbits versions. These timings describe fixed-source waveform generation, not a maximized fitting-factor search.

| Case | Previous prepared median | Current prepared median | Current full-reference median | Current speedup |
|---|---:|---:|---:|---:|
| 1 mHz, e=0 | 3.746 s | 4.155 s | 22.020 s | 5.30x |
| 3 mHz, e=0.3 | 3.861 s | 4.387 s | 22.325 s | 5.09x |
| 10 mHz, e=0.6 | 3.926 s | 4.347 s | 22.639 s | 5.21x |

Current shared preparation: geometry 5.743 s plus TDI 7.716 s, total 13.459 s. The earlier total was 8.288 s. Both were measured after the first-block JAX warm-up. Prepared-repeat medians exclude this setup and initial JIT compilation.

The earlier strict sub-four-second observation is not reproduced: current medians are about 11–14% slower. Comparing saved runs does not by itself identify whether the difference is code overhead or machine/runtime conditions. The large speed benefit remains. Every current optimized/prepared block matched the current full PyTDI reference exactly, including boundary/halo samples.

Geometry/operators are cached for one block across cases and repetitions, then released. The benchmark does not assume that a complete annual TDI cache fits in worker memory. A fresh one-source traversal pays preparation plus source evaluation; an adaptive full-year likelihood/search needs a reuse strategy suited to its evaluation order.

The 5 s/order-3 benchmark preserves a specified output, not the numerical accuracy of the referee production configuration (2.5 s, order 101/7, PM injection, full-band optimized circular-family search). The original hard-source cadence limitation remains documented in the main benchmark report.

Validation: 248 tests passed, no skips. These include both TDI generations, varying unequal arms, the zero-delay analytic limit, two-source native-delay cache reuse, explicit geometry reuse, and instrument local-metrology retention. Canonical `src/egb_multi` remains byte-identical to the frozen scientific workspace, and Task 2.8 input hashes remain accepted. No Euler source or job was changed.

[Raw recheck and validation record](referee_prepared_tdi_merge_validation.json).

Reproduction command:

```sh
MPLCONFIGDIR=/private/tmp/egb-merge-mpl XDG_CACHE_HOME=/private/tmp/egb-merge-cache \
/Users/shubhanshutiwari/Desktop/eGB-multi/.venv-benchmark/bin/python \
  benchmarks/compare_pipeline_optimizations.py --days 365.25 --repeats 3 \
  --output outputs/referee/task_2_9_20261014_prepared_pipeline_recheck
```
