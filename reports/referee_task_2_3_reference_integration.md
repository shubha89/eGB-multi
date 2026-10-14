# Task 2.3: clean responses and adopted noise-reference integration

Date: 7 October 2026. Manuscript: DV14615.

**Task 2.3's objective implementation and reference integration pass their validation checks.** The six-case physical-orientation audit also passes under the adopted `secondary_plus_laser` reference. Production PSD/covariance accuracy is not certified: the new covariance sensitivity test makes this a substantive Task 2.8 requirement. The results below are fixed-intrinsic comparisons, not full fitting factors.

All waveform code, scripts and outputs use this checkout, `/Users/shubhanshutiwari/IdeaProjects/eGB-multi`. The [run manifest](../outputs/referee/task_2_3_reference_integrated_20261007/manifest.json) records package versions, configuration and source snapshots. The [integration verification record](../outputs/referee/task_2_3_reference_integrated_20261007/integration_validation_record.json) verifies 27 snapshot/current-source hashes and the completed tests.

## What “clean waveform” means

A clean waveform contains the predicted GW response without an additive random detector-noise realization. It retains the selected source dynamics, the moving LISA response and TDI projection. It is conditional on the waveform model and numerical settings; “clean” does not mean exact physics or certified convergence.

The injection and every template use predetermined TCB timestamps, the same deterministic orbit-derived light-travel times, and the same endpoint crop. Noise is simulated separately to estimate the PSD used in their inner product. A noise seed can therefore change a finite estimated PSD and the weighted comparison, but it cannot change the clean waveform itself.

The [driver](../benchmarks/referee_task_2_objective_orientation.py) now constructs clean responses through `make_clean_response`, whose inputs contain no simulated instrument measurements. The [orbit-delay adapter](../src/egb_multi/external.py) supplies reception-time TCB delays, with native ij labels meaning receiver i/emitter j. The retarded geometry and TDI delay calculation use this common adapter. Native ICRS positions are transformed to barycentric mean ecliptic J2000; emitter positions are evaluated at emission.

## Reference and delay checks

The adopted noise components remain laser, test-mass and OMS noise at native amplitudes, with ideal clocks/ranging and other noise contributions disabled. See the [separate change note](referee_noise_reference_change_note.md) for differences from the submitted manuscript.

The integration run uses 262144 samples, 5 s cadence and 15.1704 days before cropping; TDI generation 2, instrument/TDI measurement interpolation order 45, and delay-derivative interpolation order 5. A deterministic 198-sample halo at each end allows eight nested light-travel delays plus four interpolation-kernel margins. Orbit-file samples are spaced by 600 s. These are validation settings rather than frozen production settings.

The full-run measured ideal-reference pseudoranges agree with deterministic analytic orbit delays to **5.610e-12 s** at worst. The agreement gate is 1e-10 s, a conservative 3 cm allowance for orbit-file interpolation, not a stochastic ranging-error model. Timestamps agree exactly. Both sets of arrays and the [clean-response contract](../outputs/referee/task_2_3_reference_integrated_20261007/clean_response_contract.json) are saved.

The [causal seed-isolation test](../tests/test_referee_clean_response.py) generates two native reference realizations with seeds 20261007/08, 2048 samples and 5 s cadence. Their noise differs. Their clean A/E responses remain bit-identical even after deliberately perturbing the noise object's timestamps by 0.123 s and its measured ranges by 1 ms. This tests isolation of the response from simulated measurements; it does not validate real clock/ranging calibration.

The earlier paired laser audit covered 0.1–30 mHz. The integration verifier also assesses its saved three-seed spectra over **0.02–30 mHz**, covering the low-frequency case's requested band. The largest paired laser-dependent increment PSD / laser-off PSD is **1.386e-5**; the largest channel/seed 95th percentile of absolute total-PSD change is **4.413e-4**. These remain below the earlier declared 1e-3 and 1e-2 diagnostic tolerances. This extends the existing finite-run evidence, without certifying production convergence.

## Objective and orientation validation

The response unit is fractional frequency. Native instrument noise PSDs in Hz squared/Hz are divided by the recorded optical carrier frequency squared, with carrier frequency 2.816e14 Hz. The estimator uses periodic Hann Welch windows, constant detrending, 65536 samples per segment and 50% overlap: six complete overlapping segments after cropping.

The real objective uses `dt*rfft`, a one-sided density PSD and `4*df` weighting. DC/Nyquist endpoints are excluded. No waveform taper or implicit time shift is applied. Each case fixes the injection norm, timestamps, PSD and physical band `[0.2*f_inj,10*f_inj]` for every candidate. Candidates share physical phase/orientation coefficients across A/E.

| Case | Carrier [mHz] | Eccentricity | Initial overlap | Physical orientation maximum |
|---|---:|---:|---:|---:|
| Circular, general orientation | 0.1 | 0 | 0.334050157 | 1 within numerical precision |
| Circular, face-on | 1 | 0 | 0.567618026 | 1 within numerical precision |
| Circular, edge-on, negative chirp | 3 | 0 | 0.529789803 | 1 within numerical precision |
| Circular, positive chirp | 3 | 0 | 0.739114584 | 1 within numerical precision |
| Eccentric, fixed evolution | 1 | 0.1 | 0.886558210 | 0.908836522 |
| Eccentric, Peters–Mathews evolution | 3 | 0.3 | 0.574199526 | 0.582458229 |

All six cases pass the existing declared gates. The maximum noise-weighted basis/explicit-response relative error is **1.573e-12**; the maximum overlap disagreement is **9.992e-16**. The largest circular amplitude-fitted relative residual is **7.310e-11**. The four circular injections include a common amplitude change of 2.3. Explicit local angle variations are checked around the positive-chirp solution. Full results are in [orientation_validation.json](../outputs/referee/task_2_3_reference_integrated_20261007/orientation_validation.json) and the [summary](../outputs/referee/task_2_3_reference_integrated_20261007/validation_summary.json).

## A/E covariance: implemented and assessed, production certification open

Exact A/E noise orthogonality requires idealized conditions; unequal arms or noise levels can break it. See [Hartwig et al., PRD 107, 123531](https://arxiv.org/abs/2303.15929). The native OMS amplitude tuple labels six measurement types, not six unequal link amplitudes. Our constellation nevertheless has unequal, flexing arms, so exact orthogonality is not assumed.

The [comparison module](../src/egb_multi/comparison.py) now supports either explicit diagonal weighting or a full Hermitian covariance,

$$
C(f)=\begin{pmatrix}S_A(f)&S_{AE}(f)\\S_{AE}^*(f)&S_E(f)\end{pmatrix},
\qquad S_{AE}=\mathbb E[\tilde A\tilde E^*],
\qquad (a|b)=4\Delta f\,\operatorname{Re}\sum_{f\in B}\tilde a^\dagger C^{-1}\tilde b.
$$

Cholesky whitening implements the matrix objective after dimensional PSD scaling. Nonfinite or non-positive-definite covariance is rejected; there is no eigenvalue floor or covariance shrinkage. Independent direct-DFT/inverse-matrix tests include an imaginary cross spectrum to check conjugation. Zero cross covariance recovers the diagonal objective exactly, and unit conversion preserves the full-matrix norm.

The [covariance diagnostic](../benchmarks/referee_ae_covariance.py) reuses three verified laser-enabled reference realizations, using 16384-sample segments. It pools 90 overlapping segments, approximately 85.4 effective averages under a stationary zero-covariance null. Over 0.1–30 mHz the coherence median is **0.00839**, near the approximate null median **0.00818**; 5.39% of bins exceed the approximate null 95th percentile. This does not measure exact population orthogonality: the null is approximate, individual estimates remain noisy, and colored leakage can affect the lowest frequencies.

The sensitivity test holds the clean signals and pooled diagonal PSDs fixed and adds only the pooled estimated cross spectrum. It does not replace the integration run's single-realization PSD.

| Case | Pooled diagonal maximum | Full estimated-covariance maximum | Change | Relative weighted SNR change |
|---|---:|---:|---:|---:|
| Four circular cases | 1 within numerical precision | 1 within numerical precision | Below 1.2e-14 | -5.02% to +2.34% |
| Eccentric, 1 mHz, e=0.1 | 0.875860526 | 0.878690703 | +0.00283018 | +0.723% |
| Eccentric, 3 mHz, e=0.3 | 0.584643223 | 0.558274898 | -0.02636832 | -0.806% |

Circular recovery passes with both weightings; the largest full-covariance circular fitted residual is 3.178e-9. The eccentric maximum changes are material diagnostic sensitivities. **They do not establish a true physical covariance of that size or justify using this noisy covariance as the production PSD.** Segment-length/realization changes also change the diagonal weighting, as the two eccentric tables show.

Task 2.8 must obtain a stable covariance/PSD estimate or a validated model-based covariance, then quantify the effect on optimized overlaps and loss contours. It must justify diagonal weighting or retain the supported full-matrix objective. Increasing averages, resolving signal harmonics adequately, and controlling colored spectral leakage must be considered together. This covariance uncertainty is distinct from the already tested laser-cancellation residual.

## Verification and next work

The [full suite](../outputs/referee/task_2_3_integration_full_tests.log) passes **130 tests** in 9.72 s. The short causal test and all six integration cases pass. Source snapshots and input hashes preserve reproducibility. The initial singular-covariance error-message test and an archived-metadata schema lookup were corrected; their initial logs remain saved rather than replacing them with passing output.

Reproduce inside the repository, with `/opt/miniconda3/bin/python`, one BLAS thread, and local Matplotlib/cache directories. Use fresh output directories:

```bash
python benchmarks/referee_ae_covariance.py --output outputs/referee/NEW_COVARIANCE_RUN
python benchmarks/referee_task_2_objective_orientation.py --noise-model secondary_plus_laser --output outputs/referee/NEW_REFERENCE_RUN --covariance-diagnostic outputs/referee/NEW_COVARIANCE_RUN/covariance.npz
python -m pytest -q
```

The subsequent [2.5 validation](referee_task_2_5_frequency_chirp.md) passes continuous frequency/chirp fitting at fixed sky and masses, with the coherent physical orientation/amplitude maximum at each candidate. The [2.6 validation](referee_task_2_6_sky_mass_harmonics.md) now passes staged sky/mass/finite harmonic-basin checks in four native cases. Complete optimizer validation in 2.7 and Task 2.8 certification precede revised production results and manuscript claims. The [tracker](referee_task_2_subtasks.md) records those dependencies.
