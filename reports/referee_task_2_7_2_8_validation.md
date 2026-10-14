# Task 2.7 and 2.8: optimizer and production validation

Historical status from 2026-10-08 (superseded by the current status below). **Task 2.7 passes its declared short native optimizer audit;
Task 2.8 remains open.** Neither the annual production fitting factor nor a
production configuration is certified by the results below. All scripts and
eGB imports are in this checkout. Archived failures and their source snapshots
are retained; subsequent fixes do not overwrite them.

## Current status: 12 October 2026

**Tasks 2.7 and 2.8 are complete at their stated validation scopes.** Task 2.8 closes representative annual numerical convergence under the adopted idealized stationary reference, and its [production configuration is frozen](../outputs/referee/task_2_8_20261012_completion/frozen_production_configuration.json). The final blind native verification and band/cadence follow-ups pass; see the [final evidence and limitations](referee_task_2_8_window_precision_validation.md) and [updated tracker](referee_task_2_subtasks.md). Production-grid results and manuscript/referee changes remain Tasks 2.9–2.10.

The opening 8 October status and the material below are historical. The user subsequently approved the 0.001 working target; the final band/evolution allocations are 0.00021/0.00009, retaining total 0.001. Statements below describing open annual convergence gates are superseded by the linked final review.

## Historical accuracy contract

The provisional target is an absolute fitting-factor uncertainty of 0.001,
chosen relative to the paper's 0.1 loss criterion. The user has been asked
whether a tighter target is wanted; this is an initial working assumption,
not an agreed final publication tolerance. Initial allocations are:

| Contribution | Absolute overlap allowance | Required evidence |
|---|---:|---|
| Intrinsic and extrinsic optimization | 0.0001 | Independent searches, circular recovery, small-e limit, retained restricted candidates, basin/domain expansion |
| Waveform cadence and TDI interpolation | 0.0002 | Identical absolute times and observation interval, native refinement, source and template weighted errors |
| Analysis band and colored leakage | 0.0002 | Common band within a search; stability when the complete search is repeated with an expanded band |
| Source evolution | 0.0001 | Controlled phase/evolution accuracy over one year; no inference from an unverified auto-switch tolerance |
| PSD and A/E covariance | 0.0004 | Estimation bias/variance, seed and segment-length stability, annual response, diagonal versus full covariance |

These allowances sum to 0.001. Representative pointwise checks do not establish
a uniform error bound over the template family. Contour stability still needs
tests near crossings before these allocations can be frozen.

## Implemented changes and resolved discrepancies

1. **Peters–Mathews phase was dependent on the requested output samples.** The
   old trapezoidal integration could skip the interval between the source epoch
   and the first requested sample. A one-year stress case gave a 0.0025549 rad
   phase difference between a complete evolution and a separately requested
   tail. The phase is now integrated with the adaptive evolution ODE. All three
   PM evolution modes have partition and independent tight-ODE checks. See
   [original discrepancy](../outputs/referee/task_2_7_2_8_20261008_phase_partition_original/discrepancy.json).
   The physical model and frequency convention are unchanged. Old runs retain
   the old source snapshot; they must not be relabeled as results of this fix.

2. **A coarse frequency seed grid could miss the central maximum.** Expanding
   a seven-point grid to a width of 24/T left 4/T spacing. A new offset circular
   injection recovered only 0.6633. That failure is preserved in
   [the original small-e run](../outputs/referee/task_2_7_20261008_small_e/optimizer_validation.json).
   The replacement uses 25 frequency points across the same interval, giving
   1/T spacing, and nine chirp points.

3. **Local refinement could stay in a competing sky maximum.** The denser
   search reached 0.999999779633 for the new circular stress case, just outside
   the predetermined 2e-7 circular-recovery deficit gate. Waveform-Jacobian
   refinement alone did not resolve it. Additional sky starts, with approximate
   local Doppler transport used only for seeding, recovered the true branch.
   Exact native evaluation then gave unit overlap to roundoff and a fitted
   relative residual of 1.54e-9. Injection coordinates were not used as starts.
   Both component masses remained free; this does not certify a mass reduction.
   See [successful sky-start probe](../outputs/referee/task_2_7_20261008_sky_partner_probe/validation_summary.json).

4. **Bounded-memory native response is available for annual checks.** Each
   block uses absolute source and detector times, actual retarded geometry,
   nested delays, and interpolation halos. Chunked and original native TDI
   agree in short circular and evolving-source tests. Auto evolution must first
   be resolved over the full observation; it cannot be independently selected
   in each block.

## Task 2.7 evidence

The complete optimizer audit includes the four archived 2.6 native cases and
new eccentricities 0, 0.001, 0.003 and 0.01 with deliberately incorrect initial
sky, masses, frequency and chirp. Original restricted chirp candidates and all
archived basin winners are retained under the same current objective.

The nine legacy chirp factors are evaluated at the recorded prototype.
For the three representative eccentric cases its carrier, sky and masses equal
the injection values. The new offset circular stress test deliberately uses
an incorrect prototype; its restricted-grid score is an initialization
diagnostic, not a stored submitted-paper result. Historical stored rows remain
separately archived in the 2.1 audit.

Independent DE seeds and derivative-free polishing are a separate check from
the staged SLSQP search. Finite restarts provide empirical agreement, not a
mathematical proof of a global maximum. Runs still in progress must not be
counted as passes.

| Run | Current scope/status | Evidence |
|---|---|---|
| Offset circular sky-start probe | Passed native/profile and circular recovery checks | `outputs/referee/task_2_7_20261008_sky_partner_probe` |
| Small-e sequence with added sky refinement | All four cases passed; initial dense-grid circular discrepancy preserved | `outputs/referee/task_2_7_20261008_small_e_refined_prefix` and `...small_e_refined_tail` |
| Independent archived circular search | Two constrained seeds passed; deficits 1.44e-7 and 8.81e-8 | `outputs/referee/task_2_7_20261008_independent_circular_guarded` |
| Independent e = 0.1 search | Passed both seeds; largest seed deficit 2.06e-7 | `outputs/referee/task_2_7_20261008_independent_e01` |
| Independent e = 0.3 search | Passed both seeds; largest seed deficit 3.18e-6 | `outputs/referee/task_2_7_20261008_independent_e03_union` |
| Independent e = 0.6 search | Passed both seeds across 25 basins; largest seed deficit 8.42e-5 | `outputs/referee/task_2_7_20261008_independent_e06_expanded` |

All 78 declared basin/seed pairs are present exactly once across the four
archived cases. The new small-e sequence adds four cases. These satisfy the
provisional optimizer gates and close 2.7 at this finite native-audit scope.
DE iteration limits and local optimizer statuses remain in the records;
finite restart agreement is not a proof of a global maximum.

| Native case | Common band (mHz) | Prototype-restricted grid | Retained/refined maximum | Largest independent seed deficit |
|---|---:|---:|---:|---:|
| Offset circular, unknown sky/masses | 0.6–30 | 0.856436682 | 0.999999999963 | 1.44e-7 |
| e = 0.1, fixed | 0.2–10 | 0.887427208 | 0.887436477 | 2.06e-7 |
| e = 0.3, evolving | 0.6–30 | 0.533810613 | 0.705805148 | 3.18e-6 |
| e = 0.6, fixed | 0.1–20 | 0.035303788 | 0.465266047 | 8.42e-5 |

The high-e band differs from the original 2.6 metric as explained below.
The circular restricted score uses deliberately incorrect prototype parameters.
See [combined numerical records](../outputs/referee/task_2_7_2_8_20261008_summary/optimizer_checks.json).

The refined native small-e overlaps are 1 to roundoff, 0.999992732943,
0.999934593102 and 0.999272960769 at e = 0, 0.001, 0.003 and 0.01.
The circular fitted residual is 1.54e-9. See the
[small-e recovery figure](../outputs/referee/task_2_7_2_8_20261008_summary/small_e_recovery.pdf).

The stricter 0.9999 seed-spectrum target for e = 0.6 requested a basin touching
the old 0.2f–10f band edge. That search stopped explicitly, with its traceback
and configuration retained in `task_2_7_20261008_independent_e06_union`. The new
run uses a common 0.1f–20f band for every candidate and injection normalization.
Its score cannot be directly called an improvement over the old metric; old
candidates are reevaluated in the expanded band. This is a band-sufficiency
diagnostic, not a silently discarded harmonic.

A 44-candidate band diagnostic gives maxima 0.465603503, 0.465265601,
0.465197475 and 0.465166732 in bands 0.2–10 mHz, 0.1–20 mHz,
0.05–40 mHz and 0.02–40 mHz respectively. Each band is common to the injection
and all candidates; physical orientation is refitted. The last expansion
changes the pool maximum by 9.89e-5 relative to the 0.1–20 mHz band.
This is under the provisional band allowance, but does not certify a complete
band-dependent intrinsic search or annual leakage. See
`task_2_8_20261008_band_pool`.

Independent eccentric checks include a union-of-basins DE implementation:
every declared basin receives two initial population points, neither injection
nor archived winner coordinates are used, and every basin's best is locally
refined. Two random seeds are audited. Discrete basin indexing is an optimizer
coordinate only; the physical six-coordinate domains are unchanged. Such
finite stochastic searches do not provide a mathematical global-error bound.

## Task 2.8 evidence

The submitted comparison lasts 365.25 days (31,557,600 s). The new cadence tests
use the same retained interval `[0,T)` for every resolution and source epoch
0; padding lies outside the analysis interval. A 2.5 s/order 61 reference is
compared with coarser settings. A diagnostic score with a truncated Nyquist
band cannot pass the production-band gate.

The one-day native pilot passed its proposed 10 s/order 45 **pointwise**
waveform-error gate. The largest conservative overlap-change bound was
1.84e-6. This is not a one-year or optimized-family certification.
All three one-year cases (low frequency, high frequency/eccentricity, and high
frequency with asymmetric masses) have completed and passed the declared
10 s/order 45 waveform gate. The annual high-frequency cases fail coverage at 20 s: Nyquist is 25 mHz while the prescribed
band extends to 30 mHz. Their response residual is about 0.0047–0.0049.

The low-frequency annual case has completed and passed its declared waveform
gate. At 10 s/order 45 its conservative pointwise overlap bound is 8.54e-7.
The high-frequency annual cases give source residuals of
1.34e-6 and 1.45e-6 at order 45, versus about 0.00122–0.00131 at order 15.
The corresponding overlap-change bounds are 2.68e-6 and 2.90e-6. These settings
comparisons retain the same provisional PSD and do not certify the stochastic
weighting. A separate annual native check passed for all three cases: refining
PM ODE tolerances 1e-10/1e-8 to 2e-13/1e-10 gives overlap-change bounds no larger
than 3.11e-9. The high-frequency fixed-versus-PM source residuals are 1.18 and
0.86, so fixed source evolution cannot be substituted for PM in those cases.
See [annual convergence figure](../outputs/referee/task_2_7_2_8_20261008_summary/annual_waveform_convergence.pdf).

A carrier-envelope circular response is available for further annual fits.
It retains native retarded geometry, raw nested pyTDI delay polynomials,
both masses, the prescribed JAX chirp, all six basis responses and their exact
physical coefficient constraints. It interpolates only slowly varying
demodulated responses, and is an explicitly recorded numerical approximation.
Short nonzero-phase/chirp tests verify both direct and sampled basis paths.
Annual selected-template checks pass at the low and 3 mHz carriers.
At a 30 mHz template near a TDI2 null, comparison with native order 45 gives
an overlap-change bound 0.00218; refining native interpolation to order 81
reduces it to 8.08e-7. This establishes that **order 45 is not adequate for
every high-harmonic circular template**, even though it passes the three
eccentric injections above. No optimized annual FF is certified by these checks.
See `task_2_8_20261008_year_envelope_validation` (preserved failure) and
`task_2_8_20261008_year_envelope_order81` (refined check).

### PSD/covariance gate currently fails

Eight new approximately 30-day native realizations and the original
approximately 15-day realization were analyzed at three Welch segment lengths.
No smoothing, covariance flooring, or A/E orthogonality assumption was used.
Physical orientation is refitted over a common pool of retained circular
candidates; intrinsic reoptimization has not yet been done.

| Case | Diagonal segment-length score range | Full covariance range |
|---|---:|---:|
| e = 0.1 fixed | 0.002314 | 0.003450 |
| e = 0.3 evolving | 0.006298 | 0.008879 |
| e = 0.6 fixed | 0.005947 | 0.003594 |

All exceed the provisional 0.0004 PSD allowance. See
[the preserved discrepancy](../outputs/referee/task_2_8_20261008_noise_ensemble/validation_summary.json).
Large original-to-ensemble or diagonal-to-covariance shifts are estimator
diagnostics, not established physical coherence or revised paper results.

An additional control is needed: native `noises_f_min_hz=None` selects the
inverse simulation duration. Thus the 15-day and 30-day simulations use
different low-frequency regularization. Their weighting is explicitly equal
per realization despite different segment counts. Neither their population
PSD nor their annual covariance is certified. The cutoff, discrete noise
filters, finite-window bias and statistical uncertainty must be quantified
before choosing a production estimator. The native package supplies exact
discrete component PSDs; see [LISA Instrument noise documentation](https://lisa-simulation.pages.in2p3.fr/instrument/noises.html).

### Deterministic covariance alternative under validation

`src/egb_multi/noise_covariance.py` projects the exact native discrete component
spectra through the raw SCI/TMI/REF pyTDI polynomials with frozen delays. Native
monochromatic-input checks verify the transfer sign and local metrology mapping
for unequal arms. The covariance is Hermitian and positive; no covariance
floor or statistical smoothing is introduced.

This model agrees with broad simulated spectral averages. For equal realization
weights, 50%-overlapping Hann segments give an effective count of about 104.48,
predicting normalized CSD RMS 0.09783 for uncorrelated Gaussian channels. The
measured RMS is about 0.10. This supports an estimator-scatter explanation;
it does not prove that the moving instrument's population covariance vanishes.
The diagnostic frozen-arm annual covariance gives coherence around 1e-15 over
20 microHz–20 mHz, while initial-epoch full/diagonal common-candidate score
differences are at most a few 1e-6. Annual 16-to-32 epoch integration changes
those candidate scores by less than 1e-12. See
[diagnostic results](../outputs/referee/task_2_8_20261008_frozen_noise_diagnostic/diagnostic_summary.json).

Noise-filter discretization is separate from waveform cadence. Changing native
noise physics sampling from 5 s to 0.25 s shifts the e = 0.6 common-candidate
score by about 0.000444, slightly exceeding the provisional PSD allowance.
Refining 0.25 s to 0.125 s changes it by 8.32e-7; the other cases are smaller.
See `outputs/referee/task_2_8_20261008_noise_discretization`. A final estimator
choice has been requested from the user. This alternative has not been adopted:
moving-arm laser residuals, finite-window response and PSD-grid resolution
still need quantified validation before freezing production weighting.

Covariance interpolation is now checked on 115,227 selected annual Fourier
bins, including every bin from 29 to 31 mHz. The original 65536/5 s frequency
grid gives maximum relative covariance error 0.00398; refining the grid to
1048576/5 s gives 1.56e-5. This is numerical interpolation, not PSD smoothing.
See `task_2_8_20261008_covariance_grid`.

A stronger covariance-wide noise-filter control found that 0.25 to 0.125 s
changes covariance by up to 0.000988 near 40 mHz, despite the smaller common-
candidate overlap shifts quoted above. That failure is preserved. With 0.03125 s
noise physics sampling, refinement to 0.015625 s changes covariance by at most
1.55e-5; 16-to-32 annual epochs changes it by 3.45e-7. Halving or doubling the
one-year cutoff changes it by at most 1.50e-5 over 20 microHz–40 mHz.
See `task_2_8_20261008_covariance_physics_controls` and
`task_2_8_20261008_covariance_physics_refined`.

**The one-year cutoff remains a substantive unresolved laser control.** Four
paired 15-day simulations at annual epochs, all using cutoff 1/year, exceed
the tighter 1e-4 laser-increment diagnostic allowance. They preserve 93
bit-identical nonlaser arrays, identical times and identical MPRS within each
pair. A laser-only comparison also shows a nonzero superposition difference;
this does not yet distinguish floating-point sensitivity from instrument
couplings. These results do not revoke the earlier short-run reference check,
but prevent extending that check to the proposed annual cutoff without further
evidence. A fixed 30-day cutoff passes paired controls at all four annual
epochs, with maximum increment ratio below 2.29e-5 over 20 microHz–40 mHz;
it is not an adopted production regularization. See
`task_2_8_20261008_laser_fixed_30day_cutoff` and
`task_2_8_20261008_laser_fixed_30day_other_epochs` for those controls, and
`task_2_8_20261008_annual_laser_residual` and
`task_2_8_20261008_laser_superposition`.

The instrument/TDI split has now been tested explicitly. Six raw laser inputs
and 93 nonlaser inputs are matched by hash, with identical delay arrays.
Projecting the raw-stream superposition error through TDI accounts for nearly
all of the measured discrepancy: maximum PSD ratio 2.01e-4 versus the secondary
reference, while the remaining TDI-evaluation contribution is below 5.33e-11.
Thus this discrepancy is already present in exported instrument measurements;
it is not evidence of additional loss of linearity in the TDI evaluation.
The separate true laser-only residual is not resolved by this decomposition.
On this arm64 environment, NumPy longdouble has the same precision as float64,
so a cast alone would not be a higher-precision check. See
`task_2_8_20261008_laser_arithmetic`.

The matched-seed input-scale control finds a primary laser-noise mean of
2.81e6 Hz and maximum absolute value 3.21e6 Hz for cutoff 1/year, versus
1.07e5 Hz and 4.27e5 Hz for cutoff 1/(30 days). The six-laser instrument uses
one primary noise stream under its native locking configuration. The exported
beatnote-fluctuation maximum is only 84.7 Hz in the arithmetic control. These
observations are consistent with precision sensitivity upstream of the
exported differences; they do not by themselves prove a particular internal
operation is at fault. See `task_2_8_20261008_laser_input_scale`.

Cutoff-weighting controls use the refined native noise definitions and the
fine frequency grid. Across cutoffs 15, 30 and 60 days, the largest full-
covariance common-candidate overlap range is 8.18e-7 (the evolving case).
Thirty-day versus one-year cutoff shifts are below 2.17e-7; annual-model
diagonal/full shifts are below 3e-10 in this pool. The common band is
20 microHz–40 mHz throughout. These are physical-orientation reprofilings,
not intrinsic or annual full-search certification. See
`task_2_8_20261008_cutoff_weighting`.

## Follow-up window and precision controls

The user approved the 0.001 absolute FF uncertainty target and the staged
convergence path. The [follow-up evidence note](referee_task_2_8_window_precision_validation.md)
records exact finite-window expectations, the first native arithmetic loss,
selected annual finite-covariance controls, physical primary-laser covariance
and the annual searches. These extend the evidence below without overwriting
any earlier failed estimator or residual control. The full suite now passes
197 tests after the 9 October optimizer-chart repair. The exact high-PM
boundary failure and repaired annual continuation are documented in the
follow-up note. Production remains uncertified.

## Outstanding completion gates

The [production decision note](referee_task_2_8_production_decisions.md)
sets out the estimator/cutoff choices, candidate settings and the dependent
annual checks. These settings are proposals, not a frozen configuration.

- Resolve the PSD/covariance discrepancy with a fixed, documented noise
  regularization and a validated estimator or deterministic covariance model.
- Reoptimize representative annual
  injections under cadence, interpolation, band and evolution refinements.
- Check optimizer domain boundaries and restart convergence on those annual
  cases. Distinguish physical source evolution from numerical ODE tolerance.
- Validate contour-crossing stability and freeze an accuracy-supported
  production configuration. Only then mark 2.8 complete and start 2.9.

## Reproducibility

Drivers: `benchmarks/referee_task_2_optimizer_certification.py`,
`benchmarks/referee_task_2_optimizer_refinement.py`,
`benchmarks/referee_task_2_numerical_certification.py`,
`benchmarks/referee_task_2_psd_certification.py`.
Additional diagnostics: `benchmarks/referee_task_2_frozen_noise_audit.py` and
`benchmarks/referee_task_2_evolution_precision.py`.
Library helpers: `src/egb_multi/circular_refinement.py` and
`src/egb_multi/native_validation.py`. Every native run saves source snapshots,
configuration, SHA256 provenance and numerical records beneath
`outputs/referee/`. The full suite after the source, response, covariance,
restart-coverage, direct-basis and fractional-frequency unit guards passed
176 tests in the final full-suite check. These software checks are
separate from scientific optimizer and production certification.

The [provenance audit](../outputs/referee/task_2_7_2_8_20261008_summary/provenance_audit.json)
passes listed snapshot and immutable-input hashes for 32 archived runs. This
verifies the listed bytes, not unlisted dependency closure or scientific validity.

## Preserved benchmark failures and provenance controls

The first DE/Powell circular restart aborted when Powell proposed an
out-of-domain point. The physical evaluator rejected it. The corrected driver
rejects such proposals before waveform evaluation and a DE/constrained-SLSQP
restart passes. A transient benchmark indentation error skipped those restart
calls; its apparent pass is explicitly invalidated in
`task_2_7_20261008_independent_circular_constrained/validation_summary.json`.
The corrected driver requires every basin/seed pair exactly once. The guarded
rerun contains the two independent records quoted above.

The refinement driver now copies its exact input JSON bytes before processing.
Earlier refinements read a growing completed-prefix result file; their exact
original prefixes have been reconstructed and verified against the original
SHA256 manifests, saved as `input_optimizer_validation.json`, with separate
archive records. No changed input file is presented as immutable provenance.

A native nested moving-delay secondary covariance symbol was also compared
with the frozen-arm annual model. The largest relative covariance discrepancy
over the tested 0.3–36 mHz band is 2.56e-6. This tests delay noncommutation and
frequency factors, while remaining an adiabatic symbol calculation: it does
not include finite-observation frequency-bin covariance or laser-locking
residuals. See
[the symbol comparison](../outputs/referee/task_2_8_20261008_moving_covariance_symbol/symbol_comparison.json).
