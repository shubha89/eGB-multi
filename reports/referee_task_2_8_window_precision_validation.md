# Task 2.8: window covariance and instrument precision

Updated **12 October 2026, after the final independent verification**. **Task 2.8 is complete at representative annual numerical-validation scope, and the production configuration is frozen.** The reference contains laser, test-mass and OMS noise with ideal clocks/ranges and no Galactic foreground. Clean injections and templates remain deterministic and noise-free. The revised production grid and manuscript/referee response remain Tasks 2.9 and 2.10.

## Final verification and closure

The six blind starts under the final 2.5 s, 10 microHz–80 mHz, full A/E moving-secondary-plus-primary metric all terminated successfully. All six intrinsic coordinates were free; no retained candidate or known injection-sky start was used. Two starts recovered the winning branch. The verified native overlap is **0.5243898267232836**, just **5.41105616e-9** above the matching archived result. Native minus spectral overlap is **-1.48325796e-13**; the frequency/chirp boundary distance is **46.6359** in scaled coordinates. The lower maxima reached by other starts are preserved as local-search evidence.

The optional controls also completed:

| Comparison | Signed change in native optimized overlap | Interpretation |
|---|---:|---|
| Original 20 microHz–40 mHz to 10 microHz–80 mHz, same 2.5 s reference | -1.70285232e-4 | Within the original 2e-4 band allowance |
| 80 to 100 mHz upper limit, same 2.5 s reference | -6.47144560e-10 | Negligible selected-case added-band effect |
| 2.5 to 1.25 s at 100 mHz | +1.08287441e-5 | Within the 2e-4 waveform allowance; envelope intervals also double |

All eight combined-reference band cases, all nine repaired high-PM refinement levels, twelve repaired high-PM basins, covariance/laser controls and selected finite-stationary controls are complete. The provenance refresh found **46 completed summaries and zero hash errors**. The latest full software suite remains **205 passed in 35.62 s**; it is separate evidence from the physics validation. A live task-process check found no active validation jobs.

The representative FF=0.9 bracket is **[0.089,0.089125]**. Final-reference endpoint overlaps **0.9000878015644455** and **0.8998282949490907** preserve the crossing. The local slope approximately -2.07605 maps the working 0.001 FF allowance to an eccentricity displacement of approximately **0.000482**; the narrow sampling bracket alone does not justify more precise manuscript thresholds.

The user-accepted 5 s band shift of 2.08388396e-4 remains recorded as history. Working allocations are noise **0.0004**, waveform **0.0002**, band/leakage **0.00021**, evolution **0.00009**, optimizer **0.0001**, summing to **0.001**. The final same-cadence band test now also satisfies the original 0.0002 allowance. Selected PM tightening shifts are below 1.7e-11.

## Frozen configuration and scientific scope

| Item | Adopted setting |
|---|---|
| Observation and waveform cadence | 31557600 s; 2.5 s waveform samples |
| TDI and conventions | Generation 2, fractional frequency; measurement order 101, delay order 7; native analytic retarded delays in BME J2000; A=(Z-X)/sqrt(2), E=(X-2Y+Z)/sqrt(6) |
| Evolution and circular response | Peters–Mathews injection; circular eccentricity zero with independently optimized linear fdot; physical six-response basis; PM rtol 1e-12 and phase atol 1e-10; 32768 envelope intervals |
| Noise weighting | Full Hermitian A/E annual stationary covariance from native nested moving-secondary symbols plus ideal N1-12 primary-laser symbol |
| Common band | 10 microHz–80 mHz for all candidates and the injection norm |
| Covariance discretization | Frequency spacing 1/(2097152 × 5 s); noise-filter dt 0.03125 s; 16 annual epochs, 8192 secondary geometry intervals; primary geometry dt 150 s |
| Infrared regularization | Fixed 1/year, 3.1688087814e-8 Hz; a declared noise-model regularization, not an instrument prediction |
| Intrinsic search | Frequency/chirp, full sky and both ordered masses free: 0.1 ≤ m2 ≤ m1 ≤ 1.4 solar masses; scaled chirp [-48,48] |
| Restarts and production safeguards | Reserve independent proposals in expanded basins, retain old candidates under the same metric, verify the native winner and boundaries; refine any production point failing the representative controls |

The nested covariance-grid comparison covers **2,524,293 annual bins**, with a same-family overlap-change bound **5.04035339e-5** between those specified implemented metrics. Filter, annual-quadrature, geometry and finite-window checks are selected empirical controls. The repaired high-PM finite-stationary fixed-intrinsic/orientation-profile shift is approximately **2.625e-5**. These checks support the working numerical allocations at the declared scope; they do not create a uniform bound on an untested template continuum or physical noise model.

The annual stationary/adiabatic PSD is the adopted reference for this waveform study. Moving-secondary and physical primary-laser terms are included rather than inferred negligible from earlier short runs. The 1/year cutoff is retained consistently; no shorter cutoff was adopted to make the native laser test pass. The sampled instrument's first locking arithmetic loss was diagnosed, but its full propagation chain has not been repaired by the deterministic-symbol calculation. A complete time-dependent, clock/ranging-calibrated mission likelihood remains outside this reference. Finite searches give empirical maximization evidence, not a global-maximum theorem.

Reproducible records: [closure decision](../outputs/referee/task_2_8_20261012_completion/completion_decision.json), [frozen configuration](../outputs/referee/task_2_8_20261012_completion/frozen_production_configuration.json), [final blind native verification](../outputs/referee/task_2_8_20261012_high_pm_final_metric_independent/annual_refinement.json), [optional band/cadence controls](../outputs/referee/task_2_8_20261012_high_pm_band_cadence/annual_refinement.json), [current inventory](../outputs/referee/task_2_8_20261008_summary/work_state.json), and [selected cadence convergence figure](../outputs/referee/task_2_8_20261012_cadence_figure/annual_cadence_convergence.pdf). The closure decision records hashes of the inputs, source code and reviewed evidence. Historical failed/superseded runs remain preserved.

## Historical evidence record

The material below preserves earlier decisions, diagnostic settings and progress checkpoints. Statements that a calculation is running or adoption is pending describe those earlier stages; the closure and frozen settings above supersede them.

## What the new controls establish

| Control | Verified result | Scope |
|---|---|---|
| Finite Welch expectation | Exact window/detrend lag calculation agrees with an independent dense Toeplitz calculation. Integral-grid refinements change the tested expectation by at most 4.26e-9 relatively. | Stationary covariance and the declared Hann windows; not an annual likelihood. |
| Nine native realizations | Around 30 mHz, the predicted window bias reaches 8.63%, 2.16% and 0.54% for segments of 16384, 32768 and 65536 samples. Observed A/E cross-spectrum scatter follows the finite-estimator scale. | Matched native sampling, cutoff and epochs; statistical scatter is retained. Broad agreement does not bound an optimized narrow-line FF. |
| First instrument arithmetic loss | Native N1-12 locking adds a small REF OMS fluctuation to the primary laser fluctuation. Maximum secondary recovery error is 2.328e-10 Hz. Error-free TwoSum recovers that term exactly, and 128 independent 80-digit Decimal checks confirm the represented sums. | The first locking addition only. This does not certify a corrected full instrument or remove physical laser residuals. |
| Annual rectangular periodogram diagonal | Reweighting a retained e=0.1 candidate changes its orientation-profiled overlap by -2.449e-4. | A sensitivity diagnostic that omits cross-frequency covariance; it is not an adopted noise metric or an intrinsic FF. |
| Full finite stationary covariance | The same candidate changes by -1.864e-6. Tightening CG from 1e-7 to 1e-9 changes the score by 2.75e-13; doubling the spectral integral grid from 8388608 to 16777216 changes it by 5.97e-13. Reprofiling physical orientation changes the finite score by about 1.7e-12. | Exact Toeplitz covariance restricted to the same real Fourier band, including Fourier pseudo-covariance. Annual-averaged stationary model; selected intrinsic candidate. |
| Physical primary-laser covariance | On 164138 frequencies and 16 annual epochs, the in-band generalized covariance ratio is at most 9.312e-6; around the TDI null it is at most 7.09e-8. Geometry refinements change the in-band covariance by at most 1.185e-7 relative to secondary noise. | Ideal N1-12 primary laser, exact chain-rule delay rates and a local adiabatic transfer. Sampled instrument interpolation and moving finite covariance remain separate. |
| Cancellation-safe laser evaluation | Grouping commuting-limit paths and using expm1 agrees with direct arbitrary-precision summation to 1.14e-27 in absolute transfer on the independent selected-frequency control. | Controls polynomial summation, not precision of native orbit/delay inputs. |
| Below-band laser leakage | Including the resolved physical primary-laser spectrum down to 1/(100 years) changes the retained annual finite-covariance score by about -7.64e-11. | Selected candidate; the below-band laser/secondary ratio itself becomes large because secondary TDI noise vanishes toward zero frequency. This is not a reason to ignore leakage without calculating it. |
| Annual circular controls | Two independently initialized searches yield native physical overlaps within 4.05e-11 of unity. | Cauchy-Schwarz supplies the upper bound of one, so the native candidates bracket the circular FF. This does not establish unique recovered masses or certify eccentric searches. |
| Selected annual numerical screen | At a retained e=0.1 candidate, 10 to 2.5 s refinement changes the native overlap by 6.12e-8; orders 61/81/101, envelope refinement and tighter PM integration also agree. Selected template error bounds are below 2.79e-8. | Physical orientation reprofiled; intrinsic parameters were held fixed. Optimized-FF convergence remains open. |
| Faster spectral preparation | Across annual 0.1, 3 and 30 mHz controls, the six weighted responses agree with actual sampled FFTs within 1.64e-10 relative to the oscillatory scale; profile differences are below 2.28e-15. | Full band and all six physical responses retained. Internal spline-jump sampling terms remain approximated; each search winner still requires native verification. |
| All-bin covariance perturbation | Across all 1261673 annual bins from 20 microHz to 40 mHz, including the computed primary laser changes normalized overlaps and same-family maxima by at most 1.840e-5. Replacing full A/E weighting by diagonal weighting is bounded by 2.692e-5. | Candidate-independent mathematical bounds between the two specified interpolated pointwise metrics. They do not bound missing physical effects, exact time-dependent covariance or finite-window covariance. Full A/E remains the reference. |

The last full software run (9 October) passed **205 tests**. These tests validate software and the
small independent mathematical controls; they do not close annual intrinsic
optimization, numerical convergence or contour convergence.

## Meaning for the noise decision

The instrument superposition discrepancy has a verified numerical origin at
its first affected addition. Native exported float64 measurements have
already lost information there; casting those measurements afterward cannot
recover it. The TwoSum result is a local stable reference, not a claim that
the entire native simulation has been repaired.

The physical moving-delay primary-laser residual is small in the tested
analysis band. It remains explicitly represented in the covariance
diagnostic. No laser amplitude has been reduced, no laser noise has been
disabled, and no 30-day cutoff has been adopted to hide a failed control.
The retained 1/year cutoff is used throughout these annual diagnostics.

A finite-window periodogram diagonal and the inverse full finite covariance
are different calculations. Their different overlap shifts are expected to
be tested, not combined as independent estimates of the same error. The
full-covariance result supports using the pointwise spectrum for this
candidate, but still needs stress cases and optimized-winner checks.

The earlier selected-epoch laser-symbol checks exceeded their 1e-5 diagnostic
ratio allowance and remain failed in their original archives. The dense
annual calculation reports a different quantity: the annual-averaged
covariance, with physical product rates, infrared-stable secondary transfer
evaluation and a resolved TDI-null grid. It does not overwrite those failures
or reinterpret a PSD ratio as a fitting-factor error bound.

## Annual fitting work

The four cached annual native injections are available for reuse. The
candidate-pool maxima for the eccentric cases are **not fitting factors**:
frequency/chirp, sky and both masses had not yet been optimized in that pool.
The annual search driver now performs those searches with physical
orientation profiled, retaining prior pool candidates. The independent
circular control excludes injection-pool seeds from its first search.

The annual baseline uses an immutable common band of 20 microHz–40 mHz,
full Hermitian A/E weighting, TDI2, 10 s samples, order 81, an 8192-interval
circular envelope, and both ordered masses in [0.1,1.4] solar masses. These
are validation settings, not frozen production settings. The secondary-only
baseline remains labeled diagnostic until the combined noise-model gate
closes; the primary-laser covariance is separately included in the finite
noise controls.

The short-run frequency widths do not transfer automatically to annual
signals. The high-e injection needs wider seed basins to cover its annual
sidebands; a 48-cycle control covers 0.999286 of its tapered seed power in 21
basins, whereas a 12-cycle control fails coverage. For the high-PM case, 96- and 192-cycle controls cover 0.999088 and
0.999125 of the seed power in 12 basins; narrower controls fail. This
establishes seed coverage only, and does not close the optimizer gate. These coverage failures do not establish that
the physical signal reaches the analysis-band edge. The selector first has
to cover the broad annual sidebands; its tapered seed coverage is not an
FF-error bound.

The sampled and spectral evaluators have matching retained-candidate scores
within 1e-8. The slower original search prefixes were frozen and hashed into
the replacement runs, then their candidates were re-evaluated. The original
logs, snapshots and termination records remain available. No stopped original
optimizer procedure is relabeled as completed. The current high-PM baseline
uses a scaled chirp bound of 24, exceeding the parameter-contract value of
18.12 for this source; the initial bound of 8 would not cover that contract.

The near-contour annual search completed its four selected basins and final
refinement. Its native physical winner has overlap 0.8763228675138;
native minus spectral overlap is 3.81e-13. This is a completed bounded
baseline, not a production convergence certificate. Its 648 intrinsic
evaluations include retained candidates, independent proposals, local fits
and final refinement. The first tighter two-start, local fit with both masses free
reoptimization gives native overlap 0.8763228675158. All six planned
numerical/evolution levels and the three noise/domain levels have now
completed. The order-101 reoptimization gives native overlap
0.8763228675173, within 4e-12 of the baseline. Refining cadence from 10 to
2.5 s changes the reoptimized native score by -6.12e-8. The expanded-domain
level includes a fresh full multibasin search; the other numerical levels
are local reoptimizations of retained basins.

Doubling the circular envelope to 16384 intervals and reoptimizing gives
0.8763228675167, within 3e-12 of the baseline native score. Applying full
finite stationary covariance to the completed optimized baseline winner,
with the computed physical primary laser included, gives 0.8763210015585,
a change of -1.866e-6 from its pointwise score. This closes the selected
winner check only; it does not establish an optimized maximum under the
finite covariance or a bound on time-dependent noise.

The new annual refinement driver requires a completed baseline. It preserves
the union of discovered basin winners, re-evaluates them in each immutable
metric, frees all six intrinsic coordinates, and records local versus full
multibasin reoptimization explicitly. It supports separate envelope,
interpolation, cadence, PM accuracy, primary-laser, A/E covariance and domain
controls. A local control is not automatically treated as a new global search.

## Overnight stress-case results and high-PM repair

The high-e annual baseline completed all 21 requested basins with native
winning overlap 0.3823214508015. All nine planned reoptimization levels
completed, including a full search under the expanded frequency/chirp
basins and added sky/mass starts. The 10-to-2.5 s cadence change is
-2.894e-7. Full finite stationary covariance, including the computed primary
laser, changes the baseline winning candidate by +3.015e-6. These are
selected-case controls; they do not close the production error budget.

The high-PM annual search **failed** after completing 5 of 12 basins. The
captured execution returned code 1 with `ValueError: Candidate outside
declared circular-family domain` inside the SLSQP local fit. Its 1272
saved evaluations have interim best overlap 0.47083614, which is not a
completed fitting factor and has no final native winner verification.
The original snapshot, logs and failure summary remain unchanged.

The failure was reproduced on 9 October from the final saved curvature
probes. The starting chirp coordinate was the legal lower endpoint -24,
and its measured curvature scale was 0.0866825705915513. The old
multiply/divide chart conversion returned -24.000000000000004, one ulp
outside the declared domain, before the optimizer took a step. This is a
floating-point coordinate error, not evidence that the physical search
domain or noise model needs changing.

The repaired inverse chart maps transformed box endpoints to their exact
physical endpoints. Interior division is unchanged; any residual roundoff
overshoot must be within four ulps of the bound magnitude. Genuine
scaled-coordinate violations and public out-of-domain evaluations still
raise. All six intrinsic coordinates and both masses remain free, and the
physical bounds, objective, finite-difference steps and convergence settings
are unchanged. Per-fit metadata records correction events, axes and ulps.

An actual annual local replay from the failed start preserves its original
overlap exactly. The short repaired fit executes successfully through the
previous failure; its native winner agrees with the spectral evaluator
within 2.48e-13. That five-iteration control deliberately reaches its
iteration limit and is not a completed annual optimization. Independent
constrained-quadratic and strict-domain regression controls pass. A separate
coordinate audit checks 9728 corners under 152 archived curvature-scale
sets: all decode to exact endpoints, with maximum correction one ulp, and
2432 representative interior points are unchanged. This is a chart audit,
not additional waveform or fitting-factor convergence evidence.

A separate repaired annual run retains the original configuration and
selected basins. It verifies input and unchanged-source hashes, freezes the
original checkpoint/logs, and re-evaluates all five completed basin winners
and eight retained log winners with exactly zero score difference. It
completed **all 12 basins and final refinement**. All **14 new local fits**
and the final tighter fit terminate successfully, including the exact
previously failing start. Across these 15 fits there are 85 endpoint
correction events, confined to the chirp coordinate, with maximum
correction one ulp. The native physical winner has overlap
**0.47083613744963976**; native minus spectral overlap is
**-3.643640944517301e-12**. Its spectral score improves on the retained
original best by about 2.77e-11.

The completed repair review verifies all twelve basin records, original
resume-file bytes, zero-score resume controls, recorded endpoint corrections,
native verification, test log and declared snapshot/input hashes. This
completes the search repair and the bounded high-PM baseline, not production
fitting-factor certification. The unchanged baseline metric remains the
secondary-only diagnostic; laser inclusion is tested separately. A new
dependency queue completed the saved-winner finite-covariance check with
the computed physical primary laser. Its overlap is 0.47091940028858736,
versus pointwise overlap 0.4708361374771412 in that comparison, a change of
+8.326281144616088e-5. Both CG solves converge at rtol=1e-9, with measured
relative residuals below 6.27e-10. This is the saved physical candidate with
amplitude fitted; it does not reoptimize intrinsic parameters or all physical
orientation parameters under finite covariance. Integral-grid and spectrum
refinements remain separate controls. The nine numerical/noise/domain
reoptimization levels are now running. The original failed baseline and
dependency records remain failed archives.

The pending work is to finish high-PM refinements, review representative
stress-case coverage, finish noise-model and expanded-band checks,
optimize/refine the FF=0.9 crossings, then review the combined error budget
before freezing the production configuration.

## Closure campaign started 9 October

The original high-PM numerical sequence completed all nine levels, at the older local maximum. Those controls are superseded for fitting-factor convergence by the independent search below.
Native overlaps at cadences 10, 5 and 2.5 s are 0.47083613747910,
0.47085020691066 and 0.47085674752127. The successive shifts are
1.407e-5 and 6.541e-6. This is an empirical convergence trend under the
same physical observation and band, within the waveform allowance; it is
not a uniform continuum-error theorem. Envelope, higher-order interpolation,
tighter PM integration, primary-laser inclusion and diagonal/full A/E controls
change this selected optimized result by less than 1.3e-10 from its reoptimized
baseline. The expanded-domain search also completed near 0.4708361376, showing that local stability alone did not detect the missed branch.

Three further annual native stress searches completed: the 0.1 mHz,
e=0.1, (1.2,0.2) solar-mass case gives 0.78287058154203;
the 1 mHz, e=0.3 case gives 0.57906976032255; and the small-e=1e-4
case gives 0.99999986347713. Each result retains full sky, ordered masses,
frequency/chirp and physical orientation, with a native winner check.
The e=0.088 optimized endpoint gives 0.90215455136602; the e=0.090
endpoint is running. These are representative bounded annual results,
not revised production thresholds.

The blind high-e control gives native overlap 0.38232145087232, within
7.1e-11 of its archived baseline. The blind high-PM control gives
**0.52454725927779**, improving the older 0.47083613744964 by
**0.05371112182815**. Its native/spectral difference is -1.01e-14.
The original independent validation correctly reports `requires_investigation`
because this improvement exceeds the 0.0001 optimizer allowance. Its failure,
all starts, and the old nine refinement levels remain preserved.

The archived expanded-domain search ran 24 local fits: ten from retained
winners and fourteen from mass variants, with **zero independent proposal
restarts**. Ranking the retained local maximum above fresh proposals consumed
all starts. A controlled 24-sky/5-chirp blind start also reaches 0.5245472593,
so sky-grid resolution alone is not the established cause. The restart policy
now reserves a fresh blind proposal in every expanded-search basin. The
[restart diagnosis](../outputs/referee/task_2_8_20261009_high_pm_seed_resolution/restart_policy_diagnosis.json)
freezes the completed control and records the archived start origins.

The [controlled seed-resolution diagnosis](../outputs/referee/task_2_8_20261009_high_pm_seed_resolution)
compares 24/96 sky points and 5/25 chirp values at an identical epoch,
physical band, covariance, and expanded chart. The batched proposal calculation
matches the frozen scalar calculation within 2.8e-16 on identical inputs,
with identical selected frequencies; this control does not establish a speedup.
The [replacement queue](../outputs/referee/task_2_8_20261009_high_pm_branch_queue/queue_state.json)
then runs all twelve annual basins with denser sky/chirp seeds, retaining the
new native-verified winner and the earlier basin winners. Known injection
sky coordinates and their antipode are extra starts only; all six intrinsic
coordinates remain free. Blind searches exclude those starts.
A separate review compares the archived blind results with the replacement
baseline without relabelling the original failed comparison. Nine numerical,
noise and expanded-domain levels plus finite-covariance checks are repeated
at the new branch.

Waiting contour, band and final-cadence plans that referenced the old branch
were stopped before any child started. Their snapshots, plans, logs and
supersession notes are preserved. The replacement
[contour queue](../outputs/referee/task_2_8_20261009_contour_overlap_queue/queue_state.json),
[band campaign](../outputs/referee/task_2_8_20261009_band_campaign_branch_repaired/work_state.json),
and [final-cadence queue](../outputs/referee/task_2_8_20261009_final_cadence_branch_queue/queue_state.json)
preserve their scientific dependencies. The contour now uses the completed closure queue as a scheduling dependency and can run alongside the unrelated high-PM replacement. Its output remains uncertified until the joint independent-optimizer and combined-reference review passes. The original
[closure queue](../outputs/referee/task_2_8_20261009_closure_queue/queue_state.json)
continues the crossing endpoints, finite controls and extended covariance/laser
grids. No physical noise component, mass domain or waveform model changed.

The [adaptive crossing driver](../benchmarks/referee_task_2_annual_contour.py)
will run a full annual search at every additional eccentricity node. It
retains the union of completed basin winners under each new injection,
alongside independent seeds. The nominal bisection width and the
eccentricity displacement corresponding to the absolute FF allowance are
reported separately. Final combined-reference controls are still required.

The extended native moving-delay secondary covariance numerical controls
completed successfully. Comparing the nested implemented metrics over all
**2,524,293 annual bins** from 10 microHz to 80 mHz gives relative covariance
epsilon 2.5201e-5 and a same-family overlap-change bound **5.0404e-5**.
Direct interpolation, filter, epoch and geometry refinements were also
checked at 203,587 physical frequencies. Their maximum relative covariance
changes are 6.1640e-6, 6.1686e-5, 7.6283e-7 and 3.2193e-11,
respectively. These direct physical controls are sampled; their maxima are
not uniform mathematical bounds. The old frozen-delay approximation differs
from the moving reference by about 0.001 near the second TDI null, so the
expanded-band campaign uses the moving secondary reference.
The finite low-frequency/asymmetric-mass candidate control and physical
orientation reprofile shift overlap by -1.1522e-6. This is a fixed-intrinsic
stationary covariance comparison. The old-branch high-PM refined finite
check agrees with its earlier value; its replacement-branch check remains
queued. The extended primary-laser grid also completed with its separate high-precision transfer-function control.

The [band campaign](../benchmarks/referee_task_2_band_campaign.py) waits
for immutable completed dependencies, then reoptimizes with full A/E
secondary-plus-primary covariance, tighter waveform controls and the
expanded 10 microHz–80 mHz band. It retains old harmonic basins and selects
new weighted response peaks in the expanded band. It records injection
and template power outside the original band using the same broad metric.
A candidate overlap must not be substituted for a justified family upper
bound in the orthogonal-band Cauchy inequality. A separate inequality bounds
only the increase between the two true maxima over an unchanged family.
Writing the added signal-power fraction as eta, a verified candidate gives
L <= FF_in. Cauchy implies FF_full <= g(FF_in), where
g(u)=sqrt((1-eta)u^2+eta); since g(u)-u decreases with u,
FF_full-FF_in <= g(L)-L. This does not bound a decrease, either absolute
FF, or changes from expanding the template family. The
[implementation](../src/egb_multi/band_bounds.py) avoids cancellation at
small eta and passes independent random-family, saturation and high-precision
controls. Finite-basin maxima remain empirical. These runs and their combined uncertainty review must finish
before Task 2.8 or a production configuration is marked complete.

All **205 tests pass** with the repository-local precision dependency on
the Python path; the [complete log](../outputs/referee/task_2_8_20261009_closure_plan/tests_with_batched_seeds.log)
is preserved. This software check is separate from the scientific gates.

## Initial loss-crossing screen

At the selected 1 mHz, (0.8,0.6) solar-mass, inclination 1.21 rad source,
the retained harmonic-pool overlaps are 0.918079 at e=0.080,
0.902153 at e=0.088 and 0.898003 at e=0.090. This supplies an initial
candidate crossing bracket. These are **not optimized FFs** and are not a
replacement for the paper's table. The local screened slope is about -2.08,
so an absolute FF uncertainty of 0.001 corresponds to roughly 0.00048 in
eccentricity. Direct optimized bracketing, changes of winning harmonic and
numerical refinement must determine the final reportable precision.

## Reproducible artifacts

- [Welch window audit](../outputs/referee/task_2_8_20261008_window_audit/validation_summary.json)
- [First-operation precision](../outputs/referee/task_2_8_20261008_stage_precision_year/first_operation_precision.json)
- [Cached annual sources and candidate pool](../outputs/referee/task_2_8_20261008_annual_weighting_pilot/annual_weighting_comparison.json)
- [Finite covariance and orientation](../outputs/referee/task_2_8_20261008_finite_annual_near_contour_profile/finite_orientation_profile.json)
- [Refined finite covariance](../outputs/referee/task_2_8_20261008_finite_annual_near_contour_refined/fixed_candidate_comparison.json)
- [Dense physical laser covariance](../outputs/referee/task_2_8_20261008_laser_covariance_grid/covariance_controls.json)
- [Finite covariance with physical laser](../outputs/referee/task_2_8_20261008_finite_annual_physical_laser_endpoint/fixed_candidate_comparison.json)
- [Refined integral with physical laser](../outputs/referee/task_2_8_20261008_finite_annual_physical_laser_refined/fixed_candidate_comparison.json)
- [High-e finite covariance](../outputs/referee/task_2_8_20261008_finite_annual_high_e/fixed_candidate_comparison.json)
- [High-PM finite covariance](../outputs/referee/task_2_8_20261008_finite_annual_high_pm/fixed_candidate_comparison.json)
- [High-PM annual seed widths](../outputs/referee/task_2_8_20261008_high_pm_seed_widths/seed_width_controls.json)
- [Native annual circular bracket](../outputs/referee/task_2_8_20261008_circular_annual_native_bound/circular_annual_controls.json)
- [Selected annual candidate refinements](../outputs/referee/task_2_8_20261008_annual_candidate_screen/candidate_convergence.json)
- [Spectral evaluator with input provenance](../outputs/referee/task_2_8_20261008_spectral_objective_provenance_control/spectral_validation.json)
- [Initial candidate crossing bracket](../outputs/referee/task_2_8_20261008_contour_bracket_pilot/annual_weighting_comparison.json)
- [Annual reoptimization driver](../benchmarks/referee_task_2_annual_refinement.py)
- [All-bin metric bounds](../outputs/referee/task_2_8_20261008_uniform_metric_bounds/metric_bounds.json)
- [Completed annual near-contour baseline](../outputs/referee/task_2_8_20261008_annual_search_near_contour_spectral/near_contour_annual_search.json)
- [Finite covariance of optimized near-contour winner](../outputs/referee/task_2_8_20261008_finite_annual_optimized_near_contour/fixed_candidate_comparison.json)
- [High-e completed annual baseline](../outputs/referee/task_2_8_20261008_annual_search_high_e_spectral/high_e_annual_search.json)
- [High-e refinement results](../outputs/referee/task_2_8_20261008_annual_refinement_high_e/annual_refinement.json)
- [High-PM failure record](../outputs/referee/task_2_8_20261008_annual_search_high_pm_spectral/validation_summary.json)
- [Exact boundary replay and actual annual repair](../outputs/referee/task_2_8_20261009_boundary_repair/validation_summary.json)
- [Boundary replay driver](../benchmarks/referee_task_2_boundary_repair.py)
- [Archived optimizer-chart audit](../outputs/referee/task_2_8_20261009_repair_summary/archived_chart_controls.json)
- [Repaired annual checkpoint](../outputs/referee/task_2_8_20261009_annual_search_high_pm_repaired/high_pm_search_baseline.json)
- [Completed repaired annual baseline and native winner](../outputs/referee/task_2_8_20261009_annual_search_high_pm_repaired/high_pm_annual_search.json)
- [Completed search repair review](../outputs/referee/task_2_8_20261009_search_repair_review/repair_review.json)
- [Repair review driver](../benchmarks/referee_task_2_search_repair_review.py)
- [Finite covariance of the completed high-PM winner](../outputs/referee/task_2_8_20261009_finite_annual_optimized_high_pm_repaired/fixed_candidate_comparison.json)
- [Full test log](../outputs/referee/task_2_8_20261009_closure_plan/tests_with_batched_seeds.log)
- [Immediate dependency-ordered batch](../outputs/referee/task_2_8_20261008_summary/convergence_queue_plan.json)
- [Original batch execution state](../outputs/referee/task_2_8_20261008_convergence_queue/queue_state.json)
- [Repair dependency queue](../outputs/referee/task_2_8_20261009_high_pm_repair_queue/queue_state.json)
- [Closure queue](../outputs/referee/task_2_8_20261009_closure_queue/queue_state.json)
- [Optimized contour queue](../outputs/referee/task_2_8_20261009_contour_queue/queue_state.json)
- [Combined-reference band campaign](../outputs/referee/task_2_8_20261009_band_campaign/work_state.json)
- [Moving-secondary extended controls](../outputs/referee/task_2_8_20261009_moving_queue/queue_state.json)

Each queue executes one dependent calculation at a time. The original
high-PM dependency failure is preserved; the new queue references the
separate repaired baseline. The queued work includes
finite-covariance checks of completed optimized winners, annual stress-case
reoptimization, and noise/domain controls after the near-contour numerical
run. Dependency failures are preserved and prevent the dependent job from
proceeding. Execution completion is recorded separately from scientific
convergence; the queue cannot mark Task 2.8 complete. Expanded physical-band
and optimized contour-crossing checks remain outside this initial batch.

Drivers and their exact source snapshots are stored with each run. The
optional precision dependency, mpmath 1.3.0, is installed within this checkout
at `outputs/referee/.precision_deps`; the package's precision extra records
it. All calculations import this checkout's `src/egb_multi`.

The mpmath-missing launch and two input-path/endpoint launch failures are
preserved separately. A later corrected run has its own snapshot and output
directory. Overall 2.8 completion still requires representative annual
optimized fits, refinement and independent-start agreement, band/evolution
checks, stable FF=0.9 crossings, and a justified frozen configuration.
