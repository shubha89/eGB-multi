# Circular sky mass and harmonic search validation

**Task 2.6 implemented and validated within the finite search described below.** The exact circular 1PN template family now varies sky position and both component masses, in addition to frequency, chirp and coherent physical orientation. Four native cases pass the declared waveform, candidate-retention, bound-expansion and refinement gates. The full optimizer and production fitting-factor certification remain Tasks 2.7–2.8.

The [verified summary](../outputs/referee/task_2_6_20261007_summary/summary.json) records current-source and archived-snapshot checks for 29 files per run. All eGB imports, scripts and outputs use this checkout. The adopted laser/test-mass/OMS reference with ideal clocks and ranging is unchanged.

## Physical family and numerical method

The [family search](../src/egb_multi/circular_family_search.py) profiles the same six-response physical orientation basis as Tasks 2.4–2.5. Inclination, polarization and orbital phase share coefficients across A and E; one positive common amplitude is fitted analytically. The chirping phase-independent 1PN terms remain in the objective. Each intrinsic candidate varies frequency, frequency derivative, sine of ecliptic latitude, periodic longitude and two mass coordinates.

The sky domain is the full sphere, including both poles. Equal-area Fibonacci points seed local fits. Each local longitude chart covers a full period and evaluation is canonical modulo 2π. Two coordinates on the unit square map onto the entire ordered mass triangle, 0.1 ≤ m2 ≤ m1 ≤ 1.4 solar masses. This is the computational domain in the parameter contract, not a population prior. Masses remain independent of the prescribed circular chirp.

Frequency/chirp coordinates are frequency offset times the retained duration and frequency derivative times duration squared, referred to the observation midpoint. All-sky retarded-time bounds use the norms of the actual sender and receiver positions, including emission light-travel times. The entire domain is checked for positive instantaneous frequency inside the fixed band before waveform generation.

Local SLSQP fits use measured diagonal curvature to scale their numerical coordinates. A curvature floor of 1e-6 limits scaling along flat directions; it changes neither the physical bounds nor the overlap. Physical finite-difference steps remain explicit. The first unscaled native circular search stopped at overlap 0.999999720603, despite small subsequent refinement changes, and failed the predeclared recovery gate. That [failed record](../outputs/referee/task_2_6_20261007_circular_unknown_sky_mass/validation_summary.json) remains archived. Curvature scaling resolves the failure. Three other incomplete unscaled runs have interruption records and remain separate from the passing evidence.

## Frequency basins and fixed comparison settings

Frequency seeds come from peaks of the actual noise-weighted A/E response spectrum. A Hann window suppresses leakage for seed selection only; the fit objective remains untapered. Every peak above the relative threshold is retained, and weaker peaks are added to reach the stated seed-spectrum power coverage. The nominal carrier is also retained. This accommodates the package's precessing radial/azimuthal sidebands without imposing exact half-integer frequency ratios.

The initial pass uses basin half-width 4/T, chirp bounds ±4/T², 16 equal-area sky points plus the poles, and two distinct sky starts per basin. Expansion uses half-width 6/T, chirp bounds ±8/T², 32 sky points plus the poles and three starts. Peak thresholds decrease from 1% to 0.5% of the strongest weighted peak, and target tapered-spectrum coverage increases from 99% to 99.9%. All initial basin winners are retained and reevaluated with the same objective. Seed-spectrum coverage is a diagnostic, not an upper bound on fitting-factor error or an operational detection procedure.

The native runs use 8192 samples at 10 s cadence, generation-2 TDI, measurement interpolation order 45 and delay order 5. Cropping 191 samples at either end gives T = 78100 s. One finite diagonal PSD from the adopted [2.3 reference](../outputs/referee/task_2_3_reference_integrated_20261007/psd_metadata.json) weights all candidates. Bands are fixed at 0.2–10 mHz for the 1 mHz cases and 0.6–30 mHz for the 3 mHz cases. Injection norm, timestamps, PSD and band never vary with a candidate or harmonic.

## Native results

The fixed-sky/mass baseline is rerun at these exact settings, with continuous frequency/chirp and the same physical orientation profile. It is a controlled restriction comparison; its numbers should not be compared directly with archived runs at different durations, cadences or weighting.

| Injection | Fixed sky and masses | Expanded family | Initial / expanded basins | Winning frequency in mHz |
|---|---:|---:|---:|---:|
| Circular with unknown sky and masses | 0.999996220 | 1.000000000 | 1 / 1 | 3.004007 |
| Eccentric with fixed evolution, e = 0.1 | 0.887435262 | 0.887436419 | 3 / 4 | 1.000030 |
| Eccentric with Peters–Mathews evolution, e = 0.3 | 0.533856774 | 0.705805134 | 5 / 7 | 4.499751 |
| Eccentric with fixed evolution, e = 0.6 | 0.035332762 | 0.465603503 | 17 / 18 | 3.499972 |

The deliberately chirping circular injection has unknown sky and masses relative to the initial prototype. Its final overlap deficit is 3.67e-11, and its sky-direction chord error is 0.00127. The imposed short-observation chirp tests optimization rather than an astrophysical population. Returned native mass values are not mass measurements: the largest amplitude-fitted endpoint mass-shape residual at the four best candidates is 5.54e-09. No mass-fixed reduction is certified by these checks.

![Best evaluated circular fits across frequency basins](../outputs/referee/task_2_6_20261007_summary/sky_mass_harmonic_validation.png)

The plot compares staged basin winners under the same injection normalization; the black star includes all evaluated candidates and final refinement. It has no interpolated contours. A [PDF copy](../outputs/referee/task_2_6_20261007_summary/sky_mass_harmonic_validation.pdf) is available.

| Injection | Initial intrinsic evaluations through refinement | Expanded intrinsic evaluations through refinement | Expansion change in overlap |
|---|---:|---:|---:|
| Circular with unknown sky and masses | 648 | 479 | 2.39e-14 |
| Eccentric with fixed evolution, e = 0.1 | 1486 | 1868 | 1.44e-12 |
| Eccentric with Peters–Mathews evolution, e = 0.3 | 2983 | 8728 | 1.14e-11 |
| Eccentric with fixed evolution, e = 0.6 | 8995 | 9848 | 6.5e-13 |

Each intrinsic evaluation generates six basis responses. Expanded counts include re-evaluated retained candidates; independent explicit-response and mass-audit calls are additional.

## Verification and remaining certification

The largest change from wider bounds, additional sky starts and increased peak coverage is 1.14e-11; final refinement changes the overlap by at most 1.14e-11. The maximum winning-basin starting-point spread is 3.57e-06, with 0 unsuccessful local starts in the winning basins. These records remain part of the comprehensive optimizer audit in 2.7; agreement of refinement alone is not proof of the global maximum.

Injections and explicit recovered templates use the original native factorized TDI path. Cached template responses agree in weighted shape within 1.32e-15. Explicit native overlaps agree with the profile within 6.66e-16; independent global orientation checks agree within 2.11e-15. All predeclared gates in the passing run manifests hold.

The [full test log](../outputs/referee/task_2_6_conditioned_full_tests.log) records **156 passing tests** and one upstream Astropy compatibility warning. Independent analytic responses verify joint recovery of frequency, chirp, sky and both masses with diagonal and full Hermitian covariance weighting. A separate two-line test selects the stronger frequency basin while preserving the full signal normalization. Ordered mass boundaries, longitude periodicity, poles, candidate retention and invalid-domain failures are also tested.

The [native driver](../benchmarks/referee_task_2_sky_mass_harmonics.py) runs all cases by default and requires a fresh local output directory:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/miniconda3/bin/python \
  benchmarks/referee_task_2_sky_mass_harmonics.py \
  --output outputs/referee/task_2_6_reproduction --samples 8192 --dt 10
```

The passing archive uses four processes selected with `--cases`. The [summary driver](../benchmarks/referee_task_2_6_summary.py) verifies inputs and creates the plots; the [report driver](../benchmarks/referee_task_2_6_report.py) writes this report from its numerical record.

Task 2.7 now checks the complete optimizer, including additional independent starts, the small-eccentricity limit and search-domain sufficiency. Task 2.8 certifies cadence, bandwidth, source evolution, PSD/covariance and the production error budget. The short native runs and tapered seed coverage do not replace those checks. See the [Task 2 tracker](referee_task_2_subtasks.md).
