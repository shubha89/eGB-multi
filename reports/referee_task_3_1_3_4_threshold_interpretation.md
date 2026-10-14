# Task 3 Circular Template SNR Loss and Eccentricity Measurability

Research assessment for DV14615, 12 October 2026. Tasks 3.1–3.4 are complete at targeted audit, mathematical-definition and focused literature-comparison scopes. Manuscript edits remain for the final stage. Revised numerical FF contours remain Task 2.9 work.

The chosen `1-FF=0.1` contour specifies a tolerated fractional loss of optimal matched-filter SNR when searching an eccentric signal with a declared circular family. It does not specify the confidence with which eccentricity can be distinguished from zero or measured. A fixed normalized mismatch can have very different statistical consequences at different SNRs.

## 3.1 Wording that requires qualification

The submitted PDF and current local LaTeX are distinct inputs. Their wording must not be conflated. The [wording index](../outputs/referee/task_3_20261012/manuscript_wording_audit.json) preserves source-line and one-based PDF-page locators; search hits are not automatically classified as errors.

| Passage | What is present | Final-stage action |
|---|---|---|
| Submitted abstract, PDF p. 1 | Explicitly equates the representative mismatch to 10% SNR loss, then says circular templates “remain adequate only below” the quoted eccentricities. | Preserve the existing SNR-loss qualification; tie “adequate” solely to that selected allowance, source family, reference and aggregation. Replace legacy numbers through 2.9. Do not imply adequacy for parameter estimation. |
| [Current abstract, lines 105 and 119](Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:105) | Says eccentricity invalidates the usual approximation and refers to a primarily frequency-dependent threshold; the submitted abstract's explicit loss sentence is absent here. | State the chosen SNR-loss criterion explicitly in the final coordinated abstract. |
| Section IV opening, submitted p. 10; [local line 1020](Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:1020) | Calls the result a “measurable loss in match.” | Describe a computed circular-template SNR loss; reserve “measurable” for a stated statistical analysis. |
| Figure 7 and Table I, submitted p. 11; [local caption](Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:1098), [table caption](Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:1114) | Defines the plotted threshold through a median mismatch crossing and interpolation. | Name it a 10% circular-template SNR-loss contour/crossing; retain the specified sky/inclination/mass aggregation. A sample median is not a universal source or population criterion. |
| Conclusions, submitted p. 12; [local line 1259](Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:1259) | Uses a “significant match” loss and frequency-dependent threshold. | Specify significance here as the chosen engineering allowance, not a detection confidence; qualify the numerical claim with the repaired search and reference. |
| Submitted final discussion, p. 12 | Explicitly leaves parameter-estimation precision to future work. | Preserve that distinction and make it consistent with the abstract and threshold discussion. |

Legitimate unrelated uses need no global replacement: the relative-power threshold in the harmonic plot, the automatic evolution tolerance, general astrophysical motivation about measurable eccentricity, detector link “measurements,” and the binary-pulsar discussion refer to different quantities. Source-correction comparisons at fixed parameters remain diagnostics, not full circular-family FF results.

The old restricted search cannot be converted into a true FF merely by changing its wording. Tasks 2.1–2.8 provide the repaired formulation and representative verification; the full-grid values and crossings must come from 2.9. We have not established that the previous quoted eccentricities survive those changes.

## 3.2 Mathematical meaning of the contour

Let `h=(h_A,h_E)` be a deterministic eccentric signal and `g` a circular template, with Fourier transforms in the same fractional-frequency units, detector timestamps, TDI convention and analysis band `B`. The real coherent inner product is

\[
(h|g)=4\,\mathrm{Re}\int_B\tilde h(f)^\dagger
S_{AE}(f)^{-1}\tilde g(f)\,df,
\qquad \rho^2=(h|h).
\]

`S_AE` is the fixed one-sided Hermitian noise-covariance density. The current [comparison module](../src/egb_multi/comparison.py) implements its discrete counterpart, with matching signal/noise units and a positive-definite covariance. The adopted annual reference is idealized and stationary; this norm does not assert a complete calibrated mission likelihood.

For the specified physical circular family `C`,

\[
\mathrm{FF}(h;C,S_{AE},B)
=\max_{\lambda\in C}\frac{(h|g_\lambda)}
{\sqrt{(h|h)(g_\lambda|g_\lambda)}},
\qquad \epsilon_{\mathrm{circ}}=1-\mathrm{FF}.
\]

The family retains frequency, prescribed chirp, full sky, both ordered masses, inclination, polarization and orbital phase, with one common free positive amplitude across A/E. The [frozen configuration](../outputs/referee/task_2_8_20261012_completion/frozen_production_configuration.json) supplies the actual domain and empirical search settings. Detector times stay fixed; a source reference-epoch change is not a freely shifted detector time series. No independent channel amplitudes enlarge the physical family.

The maximum expected matched-filter SNR using that family is

\[
\rho_{\mathrm{circ}}=\mathrm{FF}\,\rho,
\qquad \frac{\rho-\rho_{\mathrm{circ}}}{\rho}=1-\mathrm{FF}.
\]

Consequently `FF=0.9` means retaining 90% of optimal SNR. It retains 81% of squared SNR. The corresponding missing 19% is a noise-weighted response quantity; it is not a statement that 19% of the emitted gravitational-wave power resides in omitted harmonics.

This normalized quantity is invariant under an overall positive rescaling of the injection when the waveform shape, metric and allowed template family remain fixed. Its statistical significance is not invariant, because the signal norm changes. Changes in noise spectral shape, observing time, source evolution or parameter domain can change the FF itself.

For a unit-norm template shape `u`, the common amplitude is profiled as

\[
a_*=(h|u),\qquad
\|h-a_*u\|^2=\rho^2-(h|u)^2.
\]

Choose the physical orientation with nonnegative correlation. This family contains the opposite waveform sign via a polarization rotation, so the free positive amplitude does not obstruct the projection identity. After the shape maximization,

\[
R^2\equiv\min_{g\in C}\|h-g\|^2
=\rho^2(1-\mathrm{FF}^2)
=\rho^2(2\epsilon_{\mathrm{circ}}-\epsilon_{\mathrm{circ}}^2).
\]

This is the exact amplitude-profiled residual identity for the stated metric and family; a numerically estimated FF still carries its search/numerical errors. It agrees with the amplitude minimization in [Baird et al., Eq. (17)](https://arxiv.org/abs/1211.0546). Equal-norm waveform subtraction before amplitude profiling would instead give `2*rho^2*(1-FF)`; that is a different residual definition.

For the reported aggregate contour, specify where the **median** of the per-injection losses reaches 0.1, and state the aggregation separately for Figure 7 and each mass-conditioned Table I entry. It is a property of the sampled configurations. If production curves have multiple crossings, report them or define which crossing is summarized; monotonicity should not be assumed from the word “threshold.”

## 3.3 SNR-dependent criteria and their limits

With a fixed Gaussian covariance and zero-noise data `d=h`, if the eccentric family contains the injection, the difference between its maximized log likelihood and the best circular log likelihood is

\[
\Delta\ln\mathcal L_{\max}=\frac{R^2}{2}
=\frac{\rho^2}{2}(1-\mathrm{FF}^2).
\]

This does not equal a Bayes factor: evidences integrate the likelihood over priors. Real-noise fits also fluctuate and fit some noise. No new evidence, posterior width, detection efficiency or false-alarm probability has been computed here.

For a small loss,

\[
R^2\simeq2\rho^2(1-\mathrm{FF}),\qquad
1-\mathrm{FF}\sim\frac{C}{2\rho^2}.
\]

The coefficient `C` depends on the statistical question. It cannot be chosen solely from the number of fields in a source model.

| Primary framework | What it establishes | Restriction for our interpretation |
|---|---|---|
| [Lindblom, Owen and Brown (2008), II.A, Eq. (5)](https://arxiv.org/abs/0809.3844) | A unit noise-norm bound on model error is a sufficient accuracy standard in its measurement argument. | Useful conservative scale for waveform differences, not a calibrated eccentricity-discovery confidence. Distinguish model-accuracy requirements from allowed SNR loss. |
| [Baird et al. (2013), V.A, Eqs. (12), (16)–(19)](https://arxiv.org/abs/1211.0546) | Regular high-SNR Fisher geometry gives mean squared noise-induced waveform displacement `k`, and confidence-region sizes use a chi-square quantile. | Profiling changes the relevant dimension; confidence level also changes the coefficient. Their mass/spin confidence-region analysis is contextual, not an eccentric LISA calibration. |
| [Vallisneri (2012), II, Eqs. (8), (11)–(14)](https://arxiv.org/abs/1207.4759) | In a regular high-SNR nested-model setting, the unreabsorbed waveform correction governs likelihood/evidence scaling; the evidence contains prior-volume and noise terms. | Supports the SNR-squared scaling and the need to project out nuisance parameters. Its specific one-extra-parameter decision calibration is not transferred to eccentric LISA binaries. |

To make the confidence coefficient explicit, define `Q_k(p)` by `P(chi_k^2 <= Q_k(p))=p`. The regular Fisher approximation gives a residual bound of order `Q_k(p)`, hence a small-mismatch scale `Q_k(p)/(2*rho^2)`. The referee's `D/(2*rho^2)` is a scaling estimate associated with an effective dimension, not a universal p-value prescription.

For this analysis, assigning `D=11` from the eccentric waveform parameter count would be unjustified. The effective rank, profiling/marginalization and statistic have not been specified for an eccentricity test. At `e=0`, eccentricity is on a boundary and its periastron direction is not identifiable; harmonic alternatives can also produce separated likelihood modes. A regular local Gaussian approximation need not calibrate that comparison.

The [scalar illustrations](../outputs/referee/task_3_20261012/scalar_residual_illustrations.csv) use chosen SNRs solely to show the algebra. At `FF=0.9`, `R=sqrt(0.19)*rho`: `R=4.36` for `rho=10` and `8.72` for `rho=20`, with zero-noise log-likelihood gaps 9.5 and 38. For a loss of `10^-4`, `R` rises from approximately 0.141 at SNR 10 to 1.414 at SNR 100. These are not eccentricity forecasts or assigned detection thresholds. Use the exact expression at a 10% loss rather than treating the small-loss approximation as exact.

## 3.4 Comparison with the published Galactic-binary analysis

The direct reference is [Moore et al. (2024), Sections 3–4.1 and Appendices A–B](https://arxiv.org/abs/2310.06568v2). The following comparison keeps the observables distinct.

| Aspect | Moore et al. | Our revised FF calculation |
|---|---|---|
| Quantity | Posterior median(e)/standard deviation(e)>3; Bayes-factor cross-check | Maximum coherent circular-template SNR fraction; chosen loss 0.1 |
| Data/statistics | Zero-noise injections; Bayesian inference with 11 sampled parameters and flat priors | Deterministic eccentric injection; optimized circular family; no posterior/evidence calculation |
| Duration | Four years | One year in frozen reference |
| Noise | SciRD instrumental and duration-scaled Galactic confusion; independent TDI channels | Full annual Hermitian A/E covariance; laser/test-mass/OMS; ideal clocks/ranges; no foreground |
| Source | Fiducial 1.4+1.4 solar masses; fixed-e harmonic expansion, prescribed chirp/precession | Point-mass Newtonian/1PN source with Peters evolution; template chirp independent; masses free |
| Response | Klein (2021) response reference | Native retarded links and generation-2 pyTDI |
| Aggregation | Fiducial sky/inclination plus Appendix A controls | Sampled sky/inclination/mass summaries, with declared search bounds |

The [Task 4 assessment](referee_task_4_literature_novelty.md) verifies the cited response approximation. Frequency labels must be compared using twice the radial/periastron-to-periastron frequency, not an unqualified “orbital frequency”; precession and harmonic phase conventions still differ between source models.

For `x=f_GW/mHz`, the published Eq. (21) is

\[
e_{\min}(f_{\mathrm{GW}},\rho)\approx
(\rho^{-1.54}+\rho^{-1})
\{1.08+0.87\arctan[1.08(x-2.13)]
-0.55\arctan[2.08(x-1.22)]\}.
\]

The [reference CSV](../outputs/referee/task_3_20261012/moore_published_fit_reference.csv) evaluates this fit at directly sampled frequencies/SNRs; it does not reproduce Bayesian runs. Selected approximate values are:

| f_GW (mHz) | e_min at SNR 10 | e_min at SNR 19 |
|---|---|---|
| 0.5 | 0.0906 | 0.0445 |
| 1.0 | 0.0704 | 0.0346 |
| 2.0 | 0.0513 | 0.0253 |
| 3.0 | 0.1311 | 0.0645 |

The fit's frequency domain is 0.5–10 mHz; local evaluations use SNR 8–19. It is approximate. Our 0.1/0.2 mHz grid points lie outside that frequency domain and receive no extrapolated values. Appendix B's evidence thresholds were aligned empirically with its posterior criterion; they are not adopted here as universal odds thresholds.

The resulting comparison supports a distinction of meaning, rather than a numerical agreement claim. SNR-dependent published measurability can coexist with a fixed fractional-loss contour. Revised eGB contour values are pending, and the configurations differ; a matched-configuration reproduction would be additional research. Our FF calculation alone cannot establish eccentricity precision, source-reclassification efficiency or population completeness.

## Evidence and next dependency

The [Task 3 record script](../benchmarks/referee_task_3_record.py) preserves four pinned primary PDFs, submitted/referee documents, wording locators, the mathematical contract, scalar examples, reference-fit values and source hashes. Nine independent two-coordinate projection examples satisfy the residual identity, and Eq. (21)'s transcription is checked against the Fig. 3 fiducial point. These checks concern algebra and provenance, not new waveform or Bayesian validation.

Selected primary pages and the submitted abstract were rendered and inspected. Poppler reported a font-embedding warning for the submitted PDF; the relevant abstract is readable in its rendering and agrees with the extracted wording. No manuscript compilation or final PDF consistency check was performed.

Next: finish 2.9 before finalizing numerical conclusions in 3.5, then apply the manuscript changes and response in 3.6. No effective dimension, universal significance or new astrophysical measurement threshold has been assigned.
