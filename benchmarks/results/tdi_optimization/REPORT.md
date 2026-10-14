# Measured pyTDI acceleration

For timings including source-waveform and link generation, see the
[complete-pipeline comparison](../pipeline_optimization/REPORT.md).

The GW-only bridge previously expanded all reference/metrology beatnotes even
though they were zero, and rebuilt all nested delay operators on every XYZ
evaluation. Its main cost was Lagrange fractional-delay interpolation, including
interpolating delay arrays and their derivatives when composing operators.

Two changes preserve the existing factorized pyTDI calculation:

1. `xyz_from_links` now passes eta_ij = sci_ij directly. This equality holds for
   this bridge because all reference and metrology channels are zero.
2. `prepare_xyz_from_links` prepares pyTDI's nested delays and Doppler factors
   once, returning an evaluator for repeated source links on that same grid.
   Interpolation of each new waveform still occurs. No equal-arm or Fourier
   transfer approximation is introduced, and this remains a NumPy/pyTDI path.

## Warm TDI-only timings

CPU, float64, generation 2, measurement/delay interpolation order 3. Seven
repeats per method with randomized order, cached setup excluded from warm
execution. Orbit/link generation is outside all timings; the original and
ordinary optimized calls include their data-object/delay preparation.

| Samples | Cadence | Original | Direct GW intermediates | Prepared operators | Prepared speedup |
|---|---|---:|---:|---:|---:|
| 4,096 | 147.65625 s | 12.73 ms | 7.68 ms | 1.91 ms | 6.67x |
| 65,536 | 5 s | 126.97 ms | 85.09 ms | 19.47 ms | 6.52x |
| 631,152 (one year) | 50 s | 1.299 s | 0.879 s | 0.205 s | 6.32x |

Prepared setup costs were respectively 5.87 ms, 69.27 ms, and 725.03 ms.
The cache is useful when repeatedly changing sources on a fixed orbit/time
grid. It consumes memory proportional to sample count and stored delay chains;
for large data, process a shared block across multiple sources rather than
assuming the entire year's operator cache fits in memory.

These are measured TDI-stage speedups, not measured full-pipeline speedups.
The one-year test here uses 50 s cadence, not the earlier benchmark's 5 s.
Neither changing the cadence nor lowering interpolation order was used to
obtain any speedup within a row. The short 4,096-sample row uses generation 2
and order 3 and is therefore not a reproduction of Figure 8.

## Profile

For 65,536 samples, the original XYZ construction makes 171 `timeshift` calls,
including 105 Lagrange-coefficient evaluations. The prepared path makes 36
`timeshift` calls, including 18 coefficient evaluations. The reduction comes
from removing zero-channel work and moving nested-delay calculations out of
the repeated evaluation, not from reducing the physical signal bandwidth.

## Validation

- 22 tests passed, including all 11 previous repository tests.
- Compare both optimized paths against the full beatnote pyTDI reference for
  generations 1 and 2, interpolation orders 3 and 5, nominal and deliberately
  varying unequal arms, two independent random link sets, and boundary samples.
- Check the analytic zero-delay cancellation limit and input validation.
- All timed physical-waveform cases returned zero measured relative L2
  difference from the original XYZ outputs in every channel.
- pyTDI's existing complex-to-real warning remains; direct GW links are real
  values stored in complex arrays. No new handling of complex analytic signals
  or full instrument/noise beatnotes is claimed.

Further work could cache interpolation coefficients and implement the remaining
gather/multiply/reduce operations in JAX. Those changes are not implemented or
benchmarked here. They must retain the ordered, time-varying delay operators,
Doppler factors, interpolation order, and boundary convention.

Reproduce with:

```sh
PYTHONPATH=src .venv-benchmark/bin/python -m pytest -q
PYTHONPATH=src .venv-benchmark/bin/python benchmarks/profile_tdi.py
```

[Raw timings and versions](timings.json) · [Original profile](original_profile.txt)
· [Prepared profile](prepared_profile.txt)
