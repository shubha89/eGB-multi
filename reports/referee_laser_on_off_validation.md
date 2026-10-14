# Paired laser-on/off validation

7 October 2026. Local checkout: `/Users/shubhanshutiwari/IdeaProjects/eGB-multi`.

## Conclusion

**Laser noise can remain enabled in the tested ideal-clock/ideal-ranging reference. It is not yet negligible with every raw instrument default enabled.** The interpolation fix resolves the large numerical cancellation error found earlier. A separate residual appears when raw noisy ranging measurements are used as the TDI delays. Disabling ranging noise removes this effect in the controlled experiment, even with default clock imperfections retained.

This qualifies the earlier [laser-only audit](referee_laser_noise_and_conventions_audit.md): its negligible residual was established with perfect clocks/ranges. Its low all-default median A/E coherence did not establish negligible laser dependence in every frequency bin. The new paired comparison explicitly tests that dependence.

## Experiment and pairing

The [driver](../benchmarks/referee_laser_on_off.py) generates laser OFF and ON pairs with otherwise unchanged parameters. Native LISA Instrument noise generators use a separate deterministic sub-seed for each named component. Hashes of **93 exposed nonlaser noise arrays** match exactly within every pair. Timestamps and all six measured pseudorange (MPRS) arrays also pass bit-identical array assertions. Consequently, `ON - OFF` isolates the laser-dependent increment, including its coupling to the unchanged delay errors and numerical roundoff; it is not an ideal-clock laser-only simulation.

- Three seeds: 20261007, 20261008, 20261009; two backgrounds per seed, hence 12 primary simulations.
- 262144 samples, 5 s cadence, 1310720 s = 15.1704 days before endpoint cropping.
- Native analytic `lisaorbits` retarded TCB/LTT orbit model; generation-2 frequency TDI; SCI/TMI/REF carriers; N1-12 laser locking.
- Instrument and TDI measurement interpolation: Lagrange order 45; delay-derivative interpolation order 5.
- Endpoint halo: `ceil(8*max_delay/dt) + 4*46`, 200 samples per side for raw defaults, 198 for the ideal references.
- Welch: 65536 samples = 327680 s per segment, 50% overlap, periodic Hann window, constant detrending, mean one-sided density estimates. Six overlapping segments after cropping; frequency resolution 3.05176 microHz.
- Analysis band: 0.1–30 mHz. Noise PSD units: fractional-frequency squared/Hz, converted using the recorded laser carrier frequency.
- Synthetic sampling: physics upsampling 1, no antialias filter, no low-frequency noise cutoff. These settings and the finite duration limit the physical interpretation of the raw residual.

The predeclared acceptance limits were: maximum increment PSD / laser-off PSD below 1e-3, 95th-percentile absolute total PSD change below 1%, maximum coherence change below 0.01, relative SNR change below 1e-3, and absolute physical-extrinsic maximum change below 1e-4. Failed limits were retained; they were not relaxed after seeing the results.

## Results

Here `increment PSD` is the PSD of the paired `ON - OFF` time series. It differs from `PSD_ON - PSD_OFF`, which includes finite-realization cross terms. Each table entry is the larger A/E value across the stated band.

| Background | Seed | Maximum increment PSD / off PSD | 95th-percentile absolute total PSD change | Maximum coherence change |
|---|---:|---:|---:|---:|
| Raw all defaults | 20261007 | 1.059 | 1.379% | 0.1782 |
| Raw all defaults | 20261008 | 0.4350 | 1.020% | 0.2483 |
| Raw all defaults | 20261009 | 0.7599 | 1.373% | 0.2085 |
| Test-mass + OMS; ideal clocks/ranges | 20261007 | 2.948e-6 | 0.04367% | 0.000765 |
| Test-mass + OMS; ideal clocks/ranges | 20261008 | 6.497e-7 | 0.01794% | 0.000362 |
| Test-mass + OMS; ideal clocks/ranges | 20261009 | 2.048e-6 | 0.02982% | 0.000700 |

All three ideal-reference pairs pass the noise criteria. All three raw-default pairs fail. The [validation JSON](../outputs/referee/laser_on_off_20261007/validation_summary.json) therefore reports `requires_investigation`, and the driver exits with status 1 deliberately.

In seed 20261007, the largest all-default increment ratio occurs at **0.112915 mHz in E**. The 0.1–0.3 mHz median increment ratios are 0.03060 in A and 0.03351 in E. Another enhancement occurs at 29.9988 mHz near the band edge; it was retained in the statistics, not removed. Its association with a transfer minimum has not been independently established here. Median coherence changes only from 0.139405 to 0.139115 for this pair, despite these localized effects: median coherence alone is an inadequate laser-cancellation criterion.

## Clock/ranging isolation

The [isolation driver](../benchmarks/referee_laser_clock_ranging.py) repeats matched seed-20261007 pairs, removing selected contributions in both members of each pair while retaining all other defaults. Each pair independently verifies the same 93 nonlaser-array hashes, timestamps and MPRS equality.

| Additional terms disabled in both runs | Maximum increment PSD / off PSD | 95th-percentile absolute total PSD change | Maximum coherence change |
|---|---:|---:|---:|
| Clock imperfections only; ranging remains noisy | 9.532 | 13.90% | 0.5276 |
| Ranging noise only; default clocks retained | 2.910e-7 | 0.004993% | 0.0000910 |
| Clock imperfections and ranging noise | 1.168e-6 | 0.02541% | 0.000479 |

The installed defaults have zero ranging bias and white ranging ASD **3e-9 s/sqrt(Hz)**. Removing clock imperfections alone does not resolve the residual. Removing ranging noise does. This identifies the noisy-delay path as the remaining cause in this setup; it does not imply that actual LISA ranging performance will produce these raw unfiltered residuals. The changed laser-off background also affects the ratios between different rows, so their relative magnitudes should not be interpreted as a decomposition of one common PSD.

Native instrument source adds ranging noise to the ideal pseudorange. The local carrier adapter forwards measured MPRS to PyTDI as delays and computes their derivatives. Improved interpolation cannot make an inaccurate delay estimate exact. This mechanism agrees with the separate ranging-error and interpolation-error terms discussed by [Staab et al., Physical Review D 109, 043040](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.043040). No new sky/frame or link-indexing inconsistency was identified by this experiment.

## Effect on the waveform objective

The local eGB model supplies two fixed eccentric signals and their circular-template physical amplitude bases. Each signal is held fixed while replacing the laser-off PSD with the laser-on PSD; the fitter maximizes physical amplitude/orientation parameters at fixed intrinsic parameters.

| PSD background | Carrier frequency, eccentricity | Relative SNR change | Fixed-overlap change | Physical-extrinsic maximum change |
|---|---|---:|---:|---:|
| Raw all defaults | 1 mHz, 0.1 | -0.2361% | -5.224e-4 | -5.224e-4 |
| Raw all defaults | 3 mHz, 0.3 | +0.05396% | +6.065e-4 | +6.136e-4 |
| Test-mass + OMS | 1 mHz, 0.1 | +0.003962% | -1.985e-6 | -1.985e-6 |
| Test-mass + OMS | 3 mHz, 0.3 | +0.01100% | +9.700e-5 | +9.705e-5 |

Both ideal-reference examples pass the declared waveform tolerances; both raw-default examples fail at least one. The 3 mHz ideal-reference overlap change is close to the 1e-4 tolerance and remains a finite-realization result, not a tighter convergence bound.

These are **diagonal PSD reweighting diagnostics**, using the first noise seed and one common fixed signal/TDI response constructed with that seed's all-default MPRS. The secondary PSDs also weight that same fixed response. This isolates the change of weighting; it is not an SCET-aware signal/noise injection recovery, a certified secondary-reference absolute SNR, a full covariance likelihood, or a fitting factor maximized over frequency, chirp and sky. Absolute SNRs in the archived JSON are not used as mission sensitivity claims.

## Implementation, evidence and next step

The [Task 2 driver](../benchmarks/referee_task_2_objective_orientation.py) accepts `--noise-model secondary_plus_laser`. It retains native laser, test-mass and OMS noise amplitudes and explicitly disables other noise contributions and clock/ranging imperfections. Native component selection replaces redundant manual amplitude overrides for `secondary_only` as well. The user adopted this reference on 7 October 2026; see the [separate change note](referee_noise_reference_change_note.md).

The new option completed a full-size [PSD smoke run](../outputs/referee/task_2_3_20261007_secondary_plus_laser/psd_metadata.json). That run verified configuration and execution, not the six-case orientation-validation milestone. The full existing suite passed **117 tests**, with one upstream Astropy compatibility warning; see the [test log](../outputs/referee/laser_on_off_full_tests.log).

Task 2 now proceeds with the adopted idealized test-mass/OMS reference and laser enabled. Finish its integration with deterministic clean responses, then the full circular-family search and production convergence checks. A complete instrument reference additionally requires a justified ranging estimator/calibration, clock/time-coordinate treatment and filtering model, followed by the same paired test; that extension is outside the adopted reference's scope. Turning laser noise off in raw-default runs would conceal the ranging dependence. These scientific test results remain unchanged by the subsequent reference decision.

Evidence:

- [Primary noise comparisons](../outputs/referee/laser_on_off_20261007/noise_comparison.json), [waveform comparisons](../outputs/referee/laser_on_off_20261007/waveform_comparison.json), and per-pair PSD CSV/time-series NPZ files in that output directory.
- [Verified clock/ranging isolation](../outputs/referee/laser_clock_ranging_20261007_verified/noise_comparison.json). An earlier isolation run is preserved separately; its diagnostic model labels were shadowed by a link-loop variable. The labels were fixed and all three isolation pairs rerun, reproducing the numerical results exactly.
- [Combined summary](../outputs/referee/laser_on_off_20261007_summary/summary.json) verifies every archived source hash and all on/off nonlaser-array hashes. The original primary manifest used a shape/dtype-prefixed uint8 hashing convention for files; the summary verifies that recorded convention explicitly. New isolation manifests use plain file-byte SHA256. Source snapshots, package versions and settings are preserved.
- [PNG figure](../outputs/referee/laser_on_off_20261007_summary/laser_on_off_and_ranging.png) and [PDF figure](../outputs/referee/laser_on_off_20261007_summary/laser_on_off_and_ranging.pdf): unsmoothed, unclipped estimates, visually inspected. Versions: NumPy 2.4.6, SciPy 1.18.0, JAX 0.10.2, PyTDI 2.1.0, LISA Instrument 2.3.0, LISA Orbits 3.0.3.

Reproduce using fresh output directories, from this checkout:

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python benchmarks/referee_laser_on_off.py --output outputs/referee/repeat_laser_on_off
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python benchmarks/referee_laser_clock_ranging.py --output outputs/referee/repeat_laser_clock_ranging
python benchmarks/referee_task_2_objective_orientation.py --noise-model secondary_plus_laser --psd-only --output outputs/referee/repeat_secondary_plus_laser
python benchmarks/referee_laser_on_off_summary.py --paired outputs/referee/repeat_laser_on_off --isolation outputs/referee/repeat_laser_clock_ranging --smoke outputs/referee/repeat_secondary_plus_laser --output outputs/referee/repeat_laser_summary
python -m pytest -q
```

The first command is expected to exit 1 while raw defaults violate the declared limits. Continue the independent isolation/summary commands separately to inspect that scientific failure.
