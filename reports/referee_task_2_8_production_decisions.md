# Task 2.8: adopted production configuration and validation record

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

## Historical validation plan and candidate settings (8–9 October)

1. **PSD estimator.** Prefer an annual deterministic A/E covariance computed
   from the exact native discrete component spectra and pyTDI transfer
   polynomials, with simulations used for validation. The nine-realization
   Welch ensemble fails the provisional overlap-weighting allowance: changing
   segment length moves retained eccentric-case maxima by 0.002–0.009.
   The deterministic alternative has passed selected numerical grid, filter,
   annual-quadrature and moving-symbol controls, but its physical approximations
   and finite-window treatment are still under validation. The current
   frozen-arm implementation supplies secondary-noise covariance only.
   A separate moving-delay, ideal N1-12 primary-laser calculation is now
   available. Its inclusion changes the implemented annual pointwise metric
   by a relative covariance epsilon below 9.20e-6 across all 1261673 analysis
   bins. The resulting conservative absolute overlap bound is 1.84e-5 for
   every template and for a maximum over the same family. This is stronger
   than a retained-candidate comparison, but controls only those two specified
   pointwise metrics. Native interpolation/locking details and finite or moving
   covariance approximations remain separate physical checks. The alternative is
   further convergence work on simulation-derived PSD/covariance estimates.

2. **Fixed infrared regularization.** Use the same explicit cutoff across noise
   durations and seeds. Retain 1/year as an unresolved control and test
   1/(15 days), 1/(30 days) and 1/(60 days) as sensitivity controls. The 30-day
   value is a diagnostic candidate, not a physically established mission cutoff.
   Four paired epoch controls pass at that setting; cutoff 1/year fails the
   tighter laser-increment gate in all four. In a fixed candidate pool, changing
   the deterministic cutoff from 15 to 60 days moves the maxima by at most
   8.18e-7. This is not proof of annual optimized-FF insensitivity. Resolve the
   instrument precision sensitivity and distinguish true residual laser noise
   from arithmetic error. A cutoff must not be selected solely because it makes
   the laser diagnostic pass. Its final justification must include the native
   noise definition, finite-window effects and annual optimized-FF sensitivity.

The deterministic-covariance validation path is approved. Adoption of its final
physical approximations and cutoff still depends on the tests below. The
1/year and 15/30/60-day controls remain distinct; no shorter cutoff has been
adopted to make a failed laser test pass.

The extended moving-secondary numerical grid has now passed. Its nested
implemented-metric comparison gives an absolute same-family overlap-change
bound of 5.04e-5 across all annual 10 microHz–80 mHz bins. Direct physical
filter/epoch/geometry controls are sampled. The extended reference uses native
nested moving delays; the old frozen approximation has a relative covariance
correction near 0.001 around 60 mHz and remains a diagnostic. This progress
still does not certify the full time-dependent or sampled-instrument covariance.

## Candidate numerical settings

| Item | Candidate setting | Current evidence and limitation |
|---|---|---|
| Total absolute overlap uncertainty | 0.001 initially | Working target relative to the paper's 0.1 loss contour; user-approved working target; tighten locally if the contour precision requires it |
| Observation | 365.25 days, retained `[0,T)`, padding outside | Annual native checks use identical absolute intervals; truth epoch 0 and template epoch midpoint |
| Waveform samples | 10 s initially, independently refine to 5 and 2.5 s | Three annual injections pass at order 45 under the provisional PSD; final weighting and optimized fits need rechecking |
| Circular-template TDI interpolation | Order 81 for the validation candidate | The 30 mHz near-null template fails order 45 and passes a pointwise order-81 check; uniform family accuracy is not established |
| Optional circular envelope | 8192 intervals, refine to 16384 | Exact implemented JAX circular physics, both masses, native retarded geometry and all six physical responses retained; selected annual checks only |
| Noise-filter discretization | 0.03125 s for evaluating native discrete spectra | Refinement to 0.015625 s gives maximum relative covariance change 1.55e-5 over the tested annual bins; this is separate from waveform cadence and 5 s diagnostic simulations |
| Covariance frequency spacing | 1/(1048576 × 5 s), approximately 0.191 microHz | Maximum interpolation error 1.56e-5 on 115,227 selected annual bins including every bin from 29–31 mHz; no zero removal or flooring |
| Annual covariance quadrature | 16 midpoint epochs, check 32 | Maximum relative covariance change 3.45e-7 in the refined-noise control |
| A/E weighting | Full Hermitian matrix initially | Deterministic diagonal/full pool shifts are tiny; native moving-symbol and finite-window validation still required |
| Analysis band | 20 microHz–40 mHz for the next diagnostic | Common to all candidates and injection normalization; short pool band changes are quantified, annual full-search stability remains open |
| Sky and masses | Full sky; 0.1 ≤ m2 ≤ m1 ≤ 1.4 solar masses | Both masses stay free; no parameter reduction is adopted |
| Frequency/chirp basins and restarts | Validate specifically on annual signals | Short-observation basin widths and scaled chirp bounds cannot be assumed sufficient over one year |

The annual near-contour baseline has now completed all four selected basins
and final refinement: the native physical winner has overlap
0.8763228675138 and differs from its spectral evaluator by 3.81e-13.
This is a bounded baseline result; representative cases, numerical/domain/band
refinements and contour certification are still required. The high-PM
baseline uses scaled chirp bounds of ±24, covering its contract bound of
±18.12. Both native annual circular controls bracket their FF within
4.05e-11 of one.

The high-PM domain exception has a verified numerical cause: dividing the
curvature-scaled lower chirp endpoint moved -24 one ulp outside its declared
box. The optimizer chart now preserves exact endpoints while the physical
evaluator retains strict bounds. The annual replay and regression controls
pass; the resumed baseline rechecks archived candidates before continuing.
This repair changes no physical choices, search bounds or noise settings.
Its [evidence note](referee_task_2_8_window_precision_validation.md) separates
the short replay from the completed bounded annual baseline: all twelve
basins and fourteen new local fits complete, with native overlap
0.47083613744964 and native/spectral agreement within 3.65e-12.
The high-PM saved-winner finite-covariance check completed with overlap
change +8.33e-5; its numerical/noise/domain refinement sequence is running.
The finite check does not reoptimize the intrinsic family. Those convergence
checks and the combined error-budget review remain separate completion gates.

## Independent high-PM branch discovery: previous convergence superseded

The blind annual search found native overlap **0.52454725927779**, versus
**0.47083613744964** from the arithmetic-repaired bounded search. Its native
and spectral evaluations differ by -1.01e-14. This improvement of 0.05371112
exceeds the optimizer allowance, so all nine old high-PM refinement levels
remain selected-candidate diagnostics and cannot establish fitting-factor
convergence. The arithmetic endpoint repair remains valid.

The archived expanded-domain search ran 24 local fits: ten from retained
winners and fourteen from mass variants, with **zero independent proposal
restarts**. Ranking the retained local maximum above fresh proposals consumed
all starts. A controlled 24-sky/5-chirp blind start also reaches 0.5245472593,
so sky-grid resolution alone is not the established cause. The restart policy
now reserves a fresh blind proposal in every expanded-search basin. The
[restart diagnosis](../outputs/referee/task_2_8_20261009_high_pm_seed_resolution/restart_policy_diagnosis.json)
freezes the completed control and records the archived start origins.

A controlled sky/chirp proposal comparison and a denser full twelve-basin
baseline precede nine replacement refinement levels, a finite-covariance
check, and a separate comparison with the archived blind search. Extra known
injection sky starts are explicitly labelled and do not fix any intrinsic
parameter. Blind controls use no retained template or known sky start.
The pending contour, combined-reference band and final-cadence plans now wait
for the replacement results. Full A/E laser/test-mass/OMS reference components,
ideal clocks/ranges and both free ordered masses remain unchanged.

## Recommended convergence sequence

The working target is total absolute fitting-factor uncertainty of 0.001,
approved for this validation work. Track errors in the optimized fitting
factor and in the eccentricity of the 0.1 loss crossing. A waveform check or
retained-candidate comparison alone does not close an annual search gate.

| Stage | Calculation | Working acceptance requirement |
|---|---|---|
| 2.8a | Validate deterministic native covariance against simulations with matched cutoff, sampling, epochs and enabled components. Account for the Welch spectral window, detrending, effective segment count and statistical scatter. Refine covariance frequency grid, noise filters and annual quadrature; compare full A/E covariance with diagonal weighting. | Total noise-weighting contribution to absolute FF uncertainty at most 0.0004, including relevant model approximations and estimator uncertainty. |
| 2.8b | Isolate laser-only, secondary-only and combined measurements with matched inputs. Locate the first arithmetic discrepancy and use a genuinely higher-precision or algebraically stable reference for those operations. Repeat cutoff controls at annual epochs and quantify the physical residual under moving delays and native locking. | Demonstrate that the adopted laser residual is included or negligible within 2.8a's allowance. Keep the 1e-4 residual/secondary PSD ratio as a diagnostic; it is not itself an FF-error bound. |
| 2.8c | Establish representative one-year optimized fits with full sky, both masses, frequency/chirp and physical orientation free. Cover circular/small-e cases, low frequency, high eccentricity, asymmetric masses, harmonic switches, near-null winning templates and cases near the loss contour. Retain old candidates under each current metric. | Complete baseline records for every declared case, including winning parameters, basin, boundaries, termination status and native winner verification. No target total is inferred from fixed-template recovery alone. |
| 2.8d | Vary waveform cadence through 10, 5 and 2.5 s; vary TDI interpolation separately, including orders 61, 81 and a higher reference near problematic nulls. Refine any circular envelope approximation separately. Keep identical physical observation intervals, epochs, band and noise spectra. | Cadence plus interpolation/envelope contribution at most 0.0002 in FF. Seek a convergence trend across three settings and check both injection and optimized templates. |
| 2.8e | Expand the common analysis band until additional harmonics and colored leakage have negligible effect. Refine cadence when expanding the upper frequency limit. Validate finite-window covariance and any taper consistently for signal, templates and weighting. | Band/leakage contribution at most 0.0002 in FF. The current 20 microHz–40 mHz band is a diagnostic starting point, not a blanket limit for every production source. |
| 2.8f | Tighten PM ODE controls without changing the physical evolution model; expand frequency/chirp and harmonic basins, increase independent sky/mass starts and tighten optimization. Check any winner on a search boundary. | Evolution contribution at most 0.0001 and optimizer contribution at most 0.0001 in FF. Both masses remain free. Finite search agreement remains empirical evidence within the declared domain. |
| 2.8g | Repeat the most sensitive annual cases with numerical refinements combined. Bracket and refine the eccentricity crossings of FF = 0.9. Save a frozen configuration and all provenance only after these checks pass. | Conservative error accounting within 0.001, plus crossing locations stable at the precision reported in the revised figure/table. Tighten locally if required. |

Reevaluate each baseline winner under refined settings as a screening check,
then reoptimize the declared stress cases. A refinement can change the winning
harmonic or sky basin even when the old winner's score barely changes. Carry
the union of discovered candidates into each metric, and add independent starts;
this helps separate metric changes from optimization failures.

The allowances sum conservatively to 0.001. Do not combine them in quadrature
without evidence that the errors are independent, or present sampled differences
as rigorous uniform bounds. Locally, a smooth crossing has eccentricity
uncertainty approximately delta_FF / abs(dFF/de). Flat crossings and changes of
winning basin require direct bracketing. Use that uncertainty to determine
eccentricity sampling and reportable precision.

The native library documents exact discrete noise spectra, their dependence
on sampling, and low-frequency regularization in its
[noise definitions](https://lisa-simulation.pages.in2p3.fr/instrument/noises.html).
For the simulation comparison, the expected Welch PSD includes convolution
with the spectral window; see Nam et al., *Time-delay interferometry noise
transfer functions for LISA*, Appendix A, Eq. (A4),
[arXiv:2211.02539](https://arxiv.org/pdf/2211.02539).
Physical interpolation/filter/locking residuals are treated by Staab et al.,
*Laser noise residuals in LISA from on-board processing and time-delay
interferometry*, especially Secs. IV–VI,
[Phys. Rev. D 109, 043040 (2024)](https://doi.org/10.1103/PhysRevD.109.043040).
Their higher-precision processing control is distinct from this checkout's
observed upstream measurement discrepancy; it does not establish its cause.

The [full evidence note](referee_task_2_7_2_8_validation.md),
[noise change note](referee_noise_reference_change_note.md), and
[machine-readable state](../outputs/referee/task_2_8_20261008_summary/work_state.json)
preserve results, failures, snapshots and open requirements. Software tests
and representative numerical checks do not by themselves certify production
fitting factors or contours.

Refresh the execution/evidence inventory with `python benchmarks/referee_task_2_8_progress.py`. The immediate dependent batch is recorded in [queue state](../outputs/referee/task_2_8_20261008_convergence_queue/queue_state.json). Completing a queued execution does not automatically certify a scientific gate.
