# Adopted reference noise and changes from the submitted manuscript

**Decision adopted on 7 October 2026:** use the `secondary_plus_laser` reference for the revised circular-template comparisons. It retains laser frequency noise, test-mass acceleration noise and optical-metrology noise at their native instrument amplitudes, with ideal clocks and ranging. This is an idealized instrumental-noise reference for the waveform study. It changes the submitted paper's noise assumptions and requires revised PSDs and comparisons.

All code, runs and evidence remain in `/Users/shubhanshutiwari/IdeaProjects/eGB-multi`, using this checkout's eGB waveform implementation. The original production files and the [submission archive](../outputs/referee/task_2_1_2_2_20261007/submission_archive/manifest.json) remain the historical baseline. This decision does not imply that the revised production calculations or manuscript edits are complete.

## Final adopted weighting: 12 October 2026

**Task 2.8 is complete at representative annual numerical-validation scope; the settings are frozen.** The noise components and ideal clocks/ranges chosen on 7 October are retained. The production weighting now uses a **deterministic full Hermitian A/E covariance**, computed as an annual stationary average of native nested moving-secondary transfer symbols plus the ideal N1-12 primary-laser transfer. Native component amplitudes, discrete spectral filters and TDI conventions are recorded. Short simulated realizations support diagnostics; their noisy Welch cross spectra are not the production weighting.

The common analysis band is **10 microHz–80 mHz**, with waveform cadence **2.5 s**, measurement interpolation order **101**, fine covariance spacing **1/(2097152 × 5 s)** and fixed infrared cutoff **1/year**. The cutoff is an explicit regularization of the adopted model, consistently tied to the one-year observation. It is not a claim that physical laser noise vanishes below that frequency. No 30-day cutoff was adopted to make the failed sampled-instrument laser control pass.

The extended reference includes the moving-secondary correction near 60 mHz and the computed primary laser residual. This avoids treating them as negligible throughout the enlarged band. Nested covariance-grid comparisons bound overlap changes between the specified implemented metrics by **5.04e-5** over all annual analysis bins; selected filter/geometry/quadrature and finite-stationary controls were also reviewed. The final six-start blind high-PM native overlap **0.5243898267233** agrees with its matching reference within **5.41e-9**. Same-cadence band and finer-cadence follow-ups pass the working allowances.

These results validate a declared stationary/adiabatic, idealized instrumental-noise reference for the waveform comparison. They do not repair the full sampled-instrument locking arithmetic, establish a complete time-dependent calibrated LISA likelihood, or prove a global maximum over a continuum. Both masses remain free. The revised production grid and manuscript/referee edits remain **2.9 and 2.10**.

The [frozen configuration](../outputs/referee/task_2_8_20261012_completion/frozen_production_configuration.json), [review decision and input hashes](../outputs/referee/task_2_8_20261012_completion/completion_decision.json), and [detailed evidence](referee_task_2_8_window_precision_validation.md) preserve the settings and scientific scope. Dated progress entries below describe their original stages and are superseded by this final adoption.

## Changes to the noise assumptions

The submitted manuscript's Section III B, page 8, explicitly lists all default LISA Instrument noise sources. Figure 4 shows the associated Welch PSD. The local [manuscript source](Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex) contains the same description.

| Item | Submitted paper | Adopted reference |
|---|---|---|
| Laser frequency noise | Enabled | Enabled, retaining the native amplitudes and locking configuration |
| Test-mass acceleration noise | Enabled | Enabled at native levels |
| Optical-metrology noise | Enabled in science, test-mass and reference interferometers | Enabled in SCI/TMI/REF carriers at native levels |
| Clock imperfections | Default noise, offsets and drifts | Disabled; use ideal time coordinates |
| Ranging imperfections | Default measured ranging noise | Disabled; delays are assumed accurately known |
| Other instrumental contributions | Modulation, backlink and reciprocal-backlink noise, longitudinal/angular jitter, differential wavefront-sensing noise, tilt-to-length coupling and MOC time-correlation noise | Their noise contributions are omitted from this reference |
| Galactic confusion and other astrophysical foregrounds | Omitted | Omitted |
| Noise weighting method | Simulated all-default noise and a Welch PSD | Deterministic annual moving-secondary-plus-primary full A/E covariance; simulations used as diagnostic controls |
| Role of noise | Weight clean waveform comparisons | Same role; no additive noise realization in the signal/template comparison |

The new model explicitly selects `('laser', 'test-mass', 'oms')` through LISA Instrument's component-selection API. The [Task 2 driver](../benchmarks/referee_task_2_objective_orientation.py) accepts `--noise-model secondary_plus_laser`. The [integrated reference metadata](../outputs/referee/task_2_3_reference_integrated_20261007/psd_metadata.json) and [instrument defaults](../outputs/referee/task_2_3_reference_integrated_20261007/instrument_defaults.json) record the implementation and amplitudes. The earlier PSD-only smoke run remains archived.

Ideal timing and ranging represent negligible residual errors after correction. They do not establish that a real instrument has zero errors or that omitted backlink/jitter contributions are negligible. Clock-noise suppression and ranging sensor fusion have literature support, but finite residuals remain: [Hartwig and Bayle](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.123027), [Reinhardt et al.](https://arxiv.org/html/2307.05204v3). The revised paper must describe the selected noise components and idealization explicitly, rather than claiming an all-default or complete mission-noise PSD.

## Evidence supporting the decision

The [paired laser-on/off report](referee_laser_on_off_validation.md) compares three matched seeds over 0.1–30 mHz. Each pair has 93 bit-identical nonlaser noise arrays, identical timestamps and identical measured pseudoranges.

| Test | Observation | Interpretation |
|---|---|---|
| Laser enabled with test-mass/OMS noise and ideal clocks/ranges | Largest paired laser-dependent increment PSD / laser-off PSD is 2.948e-6 across A/E and three seeds | Laser noise can remain enabled in this tested reference |
| All raw defaults enabled | The corresponding largest ratio is 1.059; all three seeds fail the declared noise criteria | The raw-default PSD contains a material laser-dependent residual |
| Disable ranging noise alone, retaining default clocks | Largest ratio is 2.910e-7 in the controlled seed | The noisy ranging-delay path accounts for the remaining effect in this setup |
| Two fixed-waveform PSD reweighting examples | Ideal-reference SNR and physical-extrinsic maximum changes pass the declared tolerances | Supports the reference choice at the tested scope; full search and production convergence remain pending |

These runs use 262144 samples at 5 s cadence, 15.1704 days before cropping, generation-2 frequency TDI, instrument/TDI interpolation order 45, delay-derivative interpolation order 5 and native analytic retarded orbits. They use synthetic unfiltered sampling, without a low-frequency noise cutoff. These are diagnostic settings, not a frozen production cadence, duration or analysis band.

The [summary](../outputs/referee/laser_on_off_20261007_summary/summary.json) and [validation record](../outputs/referee/laser_on_off_20261007_summary/validation_record.json) preserve provenance checks and the failed raw-default outcome. The waveform tests reweight one common fixed response and are not full fitting-factor searches or a realistic clock-calibrated injection recovery.

## Numerical and convention corrections

The following corrections are separate from the adopted change of noise components. Their effects must not all be attributed to changing the PSD.

| Correction | Revised treatment |
|---|---|
| Instrument noise projection | Include SCI/TMI/REF carriers, rather than projecting science carriers alone |
| Numerical laser cancellation | Validate TDI precision/cadence and diagnose the first instrument locking arithmetic loss; deterministic-symbol weighting does not repair the full sampled instrument |
| Signal and PSD conventions | Record generation-2 TDI and matching A/E definitions; convert Hz-noise PSDs to fractional-frequency squared/Hz using the recorded carrier frequency |
| Coordinate frame | Transform native ICRS orbit positions to the source's barycentric mean ecliptic J2000 frame |
| Moving-link geometry | Evaluate the emitter at emission using native reception-time light-travel times |

See the [convention audit](referee_laser_noise_and_conventions_audit.md). The original cached PSD lacks complete generation provenance; it can serve as a historical weighting curve, not a certified compatible physical reference.

For the adopted model, the integrated driver now supplies predetermined TCB timestamps and deterministic orbit-derived delays directly to the clean GW response. It does not use simulated instrument measurements to construct the injection or templates. Ideal-reference MPRS agree with these delays within 5.61e-12 s, and a causal two-seed/adversarial-measurement test preserves bit-identical clean A/E responses. See the [2.3 integration report](referee_task_2_3_reference_integration.md).

The objective now supports full Hermitian A/E covariance as well as explicit diagonal weighting. A three-seed, 90-segment covariance diagnostic has coherence near the finite-estimator null level, but using its estimated cross spectrum shifts the eccentric maxima by +0.00283 and -0.02637. This does not establish population covariance of that size. This sensitivity motivated the deterministic annual covariance adopted after the 2.8 controls above; laser cancellation alone did not settle the weighting.

## Remaining work for Task 2

The reference-model decision is settled. Task 2.3's implementation/reference integration passes, and all six orientation/objective cases pass with laser enabled. Task 2.5 validates continuous frequency/chirp fitting at fixed sky and masses using the adopted PSD. Task 2.6 now extends the exact circular family over sky, both masses and finite frequency/harmonic basins, with four passing native cases. These optimizer extensions do not change the adopted noise assumptions. Tasks 2.1–2.2 and 2.4 retain their validated scope; the short native optimizer audit in 2.7 now passes eight cases and 78 basin/seed checks, and 2.8 now closes representative annual reference convergence and freezes settings. The full revised production results remain 2.9.

| Order | Task | Required result |
|---:|---|---|
| Done at implementation/validation scope | 2.3 integration and fixed-intrinsic 2.4 confirmation | Deterministic clean responses, recorded reference provenance, six passing laser-enabled cases, diagonal/full-matrix objective checks and covariance sensitivity assessment. See the [integration report](referee_task_2_3_reference_integration.md). Annual reference convergence is documented in completed 2.8. |
| Done at bounded validation scope | 2.5 continuous frequency and chirp search | Both offset circular injections recover unit overlap at fixed sky/masses; all four native cases pass refinement, explicit-response and global-orientation checks; 145 tests pass. See the [2.5 report](referee_task_2_5_frequency_chirp.md). |
| Done at finite-basin validation scope | 2.6 sky, mass and harmonic search | Full sky and ordered masses retained; four native cases pass bound/start/coverage expansion and explicit-response checks; 156 tests pass. Higher harmonics improve the e = 0.3 and e = 0.6 examples. See the [2.6 report](referee_task_2_6_sky_mass_harmonics.md). No mass-fixed reduction or production fitting factor is certified. |
| Done at short native scope | 2.7 optimizer validation | Eight passing cases and 78 basin/seed checks, with circular recovery, small-e limit, independent starts and retained original candidates. |
| Done at representative annual reference scope | 2.8 convergence jointly with Comment 6 | Numerical/domain/band and independent native controls complete; full Hermitian moving-secondary-plus-primary weighting and reproducible settings frozen. Working absolute FF allowance 0.001; no mass reduction, full mission-noise certification or rigorous global-search bound claimed. |
| 6 | 2.9 production comparisons | Rerun the science grid, refine eccentricity near loss-contour crossings, regenerate Figure 7/Table I and audit Figures 5, 6 and 8. Replace Figure 4 with the adopted PSD. |
| 7 | 2.10 manuscript and referee response | State the template family/domain, maximization method, noise assumptions, validation and changed numerical results; update terminology, equations, captions and conclusions. Coordinate the loss-contour interpretation with Comment 3. |

Within every search, fix the PSD, physical analysis band, timestamps and injection norm. All candidate harmonics must use that same normalization. The 0.1–30 mHz diagnostic band is not automatically adequate for the complete production source grid; relevant harmonics and precession/chirp broadening must fit the certified band and sampling.

On representative cases, compare old and adopted PSD weighting while holding the corrected waveforms, objective and candidate set fixed. Then compare restricted and expanded searches using the same adopted PSD. Keep the archived submitted results separately. This distinguishes the effect of changing noise assumptions from response corrections and improved maximization.

## Manuscript changes and conditions for revisiting the decision

Update Section III B and Figure 4 to specify the enabled noise sources, ideal clock/ranging assumptions, units, TDI generation and PSD estimator. Recompute affected waveform comparisons before updating Figure 7, Table I and numerical conclusions. State the reference's scope wherever the paper currently implies complete realistic instrument-noise weighting.

Revisit the reference if a complete instrument-sensitivity prediction becomes an objective, if finite calibration residuals or omitted instrumental terms materially change the results, or if A/E covariance/PSD convergence changes the loss contours beyond the adopted error budget. A complete calibration pipeline is an extension of this chosen reference, rather than a prerequisite for reporting results explicitly conditional on the idealized model.

The [Task 2 tracker](referee_task_2_subtasks.md) contains the implementation dependencies. Task 2 finishes when the defined circular-family search, its numerical error budget, the revised production results and the corresponding manuscript/referee changes are all complete.

## Historical 2026-10-08 validation update

Task 2.7–2.8 work is recorded in the [validation and error-budget note](referee_task_2_7_2_8_validation.md). The physical noise-component choice is unchanged. Eight new native realizations plus the original reference do not yet provide a stable narrow-line PSD/covariance: segment-length changes move common-candidate overlaps by 0.0023–0.0089, exceeding the provisional 0.0004 weighting allowance. No production PSD is certified.

The native default `noises_f_min_hz=None` resolves to the inverse simulation duration, so unequal-duration noise runs have different low-frequency regularization. This must be controlled explicitly in subsequent production validation. A deterministic frozen-arm covariance from the same native discrete component spectra and pyTDI combinations is being assessed as an estimator alternative; it has not been adopted. Moving-arm laser residuals, finite-window bias and annual weighting remain validation requirements.

The 8 October controls show why this regularization must be explicit. Using a
fixed cutoff 1/year in paired 15-day simulations at four annual epochs fails
the tighter 1e-4 laser-increment diagnostic gate. The native infrared laser
model becomes much larger at low frequency when its cutoff is lowered. A
laser-only/superposition diagnostic is nonzero, so numerical precision and
instrument couplings still require an audit. A fixed 1/(30 days) cutoff passes
all four epoch controls (maximum increment ratio 2.29e-5 over 20 microHz–40 mHz).
It remains a proposed diagnostic setting, not an adopted physical revision.
The earlier short-duration laser-on/off result remains valid at its recorded
settings; it cannot certify arbitrary annual noise regularization.

A subsequent matched-input arithmetic decomposition places almost all of the
superposition discrepancy in the exported instrument measurements: its
projected PSD ratio is up to 2.01e-4, compared with less than 5.33e-11 from the
remaining TDI-evaluation contribution. The true laser-only residual is still
separate. In the deterministic covariance pool, changing the cutoff from 15
to 60 days shifts the profiled maxima by at most 8.18e-7. Neither check replaces
the required annual optimized-fitting-factor validation. All results remain
archived in the [2.7–2.8 evidence note](referee_task_2_7_2_8_validation.md).

Within the diagnostic deterministic covariance, grid interpolation near the
30 mHz TDI2 null and noise-filter discretization also need refinement. The
finest tested grid gives maximum relative covariance error 1.56e-5, and
0.03125-to-0.015625 s noise-filter sampling changes covariance by at most
1.55e-5 over the tested band. These numerical controls do not settle the
production estimator, moving-arm laser residual, or finite-window approximation.
See the [2.7–2.8 evidence note](referee_task_2_7_2_8_validation.md).

A separate waveform correction integrates Peters–Mathews azimuthal phase with the adaptive ODE instead of a trapezoid over requested output samples. This removes output-partition dependence without changing the 1PN source model. The old discrepancy, old source module and every pre-fix run snapshot are preserved. Revised runs must carry the new source hash; this numerical correction must be documented separately from noise-reference changes.


## Historical follow-up window and precision validation

The [new evidence note](referee_task_2_8_window_precision_validation.md)
records the approved 0.001 FF uncertainty target, exact Welch window
expectations, a verified precision loss at the first native laser-locking
addition, and annual finite-covariance/physical-primary-laser diagnostics.
The physical enabled-noise choice is unchanged. The first-operation TwoSum
control does not claim that the full native instrument is repaired. No
shorter cutoff has been adopted to make a failed residual control pass.

On 9 October, expanding the noise-transfer check through 80 mHz exposed a
limitation of the frozen-delay secondary covariance near the second TDI
null. Its maximum relative covariance difference from the native nested
moving-delay symbol is 0.000998765 near 60.088 mHz, agreeing between
4096-geometry/16-epoch and 8192-geometry/32-epoch calculations. This is a
physical transfer approximation difference, distinct from frequency-grid
interpolation error. The final closure-grid calculation therefore uses the
native moving secondary symbol and checks its frequency, geometry, filter
and annual-epoch resolution. Both are stationary annual averages of local
symbols, not an exact time-dependent likelihood. The enabled components,
native amplitudes, ideal clocks/ranges and absence of foregrounds are unchanged.
This validation reference is not yet frozen for production. The
[extended controls](../outputs/referee/task_2_8_20261009_moving_secondary_refined/symbol_comparison.json)
and [closure review](../outputs/referee/task_2_8_20261009_closure_review/closure_review.json)
retain the distinction between numerical convergence and model scope.

For the retained near-contour, high-e and high-PM annual candidates, full
finite stationary covariance differs from pointwise weighting by about
-1.86e-6, +2.41e-7 and -8.00e-6, respectively. Those are selected-candidate
checks, not production fitting-factor bounds. Annual optimization and the
remaining numerical/band/contour gates must pass before estimator adoption
and revised paper numbers are finalized.

The completed near-contour annual baseline has native overlap 0.87632287. Full finite stationary covariance, including the computed primary laser, shifts its winning candidate by -1.866e-6. Across every annual analysis bin, a separate covariance perturbation bound limits the effect of including that primary laser to 1.84e-5 for any overlap or maximum over the same family. This bounds the specified pointwise metrics; time-dependent covariance and native interpolation/locking approximations remain separate. Full A/E covariance remains the reference, although its difference from diagonal weighting is bounded by 2.70e-5 for those implemented metrics. See the [all-bin bound](../outputs/referee/task_2_8_20261008_uniform_metric_bounds/metric_bounds.json) and [updated tracker](referee_task_2_subtasks.md).
