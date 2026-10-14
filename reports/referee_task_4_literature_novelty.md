# Comment 4: Literature and contribution assessment

Reviewed 12 October 2026. Research preparation for DV14615; manuscript and bibliography edits have not been applied. This assessment addresses the four requested references and three closely relevant foundations. It is a focused comparison, not an exhaustive priority search.

The referee's central concern is supported: established work already models eccentric harmonics, secular evolution and precession, and performs eccentric LISA data analysis. Our contribution should be stated through the specific implementation, its stated approximations and its verified calculations. Complexity of an unimplemented alternative does not demonstrate that the alternative is impractical.

## 1. Demands and present citation roles

The [referee report](</Users/shubhanshutiwari/Desktop/Yourmanuscript DV14615 Tiwari.pdf>), PDF page 2, requests acknowledgement of Moore et al. (2310.06568) as waveform/data-analysis work, discussion of established Fourier-domain eccentric models (0906.0313 and 1602.03081), recognition of harmonic confusion including 2605.15265, and clearer positioning of the present work.

| Current location | Verified issue | Revision action at the final manuscript stage |
|---|---|---|
| [Introduction, line 158](Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:158) | Moore's key occurs once, as motivation for identifying neutron-star components. | Discuss its waveform, multi-harmonic inference and source-misidentification work explicitly. |
| [Source attribution, line 192](Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:192), and lines 341/356 | Tessmer–Gopakumar and Tanay et al. are already acknowledged for source dynamics. | Preserve attribution; expand Tanay's role to its FD construction. Do not describe an existing citation as missing. |
| [Response and benchmark discussion, line 1143](Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:1143) | The direct-link route is described, followed by a native-workload circular comparison. | Specify the implementation contract and approximation limits; qualify performance evidence as described below. |
| [End of Section IV, line 1208](Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:1208) | Harmonic/precession bookkeeping is used to suggest erosion of the FD speed advantage without an implemented eccentric FD comparison. | Acknowledge existing solutions, then state the reason for choosing a direct TD reference. Omit an unmeasured crossover or general feasibility claim. |
| [Bibliography](Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.bib) | Moore, Tanay, Tessmer–Gopakumar and Klein et al. (2018) exist. Entries for 0906.0313, 2605.15265 and 2106.10291 were not found. | Add the prepared entries and use existing entries in the relevant technical discussion. |

The paper does not literally state that FD waveforms are impossible. The problem is the implication of its comparative discussion and the absence of the most relevant precedents in that discussion. The reply should acknowledge the concern without attributing stronger wording to the submitted text.

## 2. Capability comparison

Each row describes the inspected construction, not everything that may exist in the authors' software. PDF page numbers are one based. Pinned PDFs/text and detailed marker locators are in [literature evidence](../outputs/referee/task_4_20261012/literature_evidence.json).

| Reference and locator | Source and numerical construction | Detector/data-analysis scope and implication |
|---|---|---|
| [Moore et al. (2024), 2310.06568v2](https://arxiv.org/abs/2310.06568v2), Sections 2–3, 4.3–4.4, Appendix C; model on PDF pp. 3–4 | Quasi-monochromatic harmonic model with linear chirp and advancing periastron; small-e harmonic amplitudes and dominant polarization terms. It is not a fully coupled evolving-eccentricity Peters integration. | Includes a LISA response and A/E/T likelihood; uses multi-harmonic heterodyning for Bayesian inference. Demonstrates harmonic misidentification and reclassification. Direct precedent for eccentric Galactic-binary analysis; neither eccentricity inference nor harmonic confusion is an eGB novelty claim. |
| [Yunes et al. (2009), 0906.0313v3](https://arxiv.org/abs/0906.0313v3), Sections III–IV and VI | Analytic post-circular source waveform: Newtonian higher harmonics with amplitude/phase eccentricity corrections; expansion through eighth order and SPA Fourier construction. | Establishes analytic FD eccentric waveforms. Its source approximant should not be attributed our native unequal-arm nested TDI pipeline. Its approximation domain must be checked before a numerical comparison to a long-lived, nearly monochromatic source. |
| [Tanay, Haney and Gopakumar (2016), 1602.03081v2](https://arxiv.org/abs/1602.03081v2), opening of Section II on PDF p. 3; Sections II.2 and III | FD construction: Newtonian amplitude, 2PN Fourier phase, consistent eccentricity evolution through sixth order. The paper explicitly does not pursue precession's effect on FD harmonic structure. Its separate TD quasi-Keplerian construction includes precession and 2PN dynamics/evolution. | Established eccentric FD evolution and TD source dynamics. Keep its two models separate; credit evolution without assigning the TD precession content to the FD model. |
| [Xuan et al. (2026), 2605.15265v2](https://arxiv.org/abs/2605.15265v2), Section IV, Eqs. (13)–(16), PDF p. 14; Section V; Appendix A.4, p. 23 | Stellar-mass BBH population/harmonic analysis and x-model PN comparison. The apparent circular chirp-mass mapping matches the first frequency derivative; the authors state that this is not a full waveform fit. Their PN comparison fixes intrinsic parameters. | Relevant harmonic-confusion context from another population. The inspected SNR construction uses sky/polarization-averaged sensitivity; implementation in the full global fit is left for future work. Do not describe these results as DWD recovery benchmarks or a measured full-search FF. |
| [Klein et al. (2018), 1801.08542v2](https://arxiv.org/abs/1801.08542v2), Section II, Eqs. (3)–(5), (16)–(18), II.2; periastron construction on PDF pp. 5–6 | FD eccentric waveforms separate orbital and periastron phases, include PN secular evolution and treat spin precession using shifted uniform asymptotics. Amplitudes use a small-e expansion with a stated validation domain. | Directly relevant counterexample to a general claim that FD constructions cannot practically accommodate precession/evolution. Already in our bibliography. Its reported accuracy is not a validation of eGB or an identical detector setup. |
| [Klein (2021), 2106.10291v1](https://arxiv.org/abs/2106.10291v1), Section IV, Eqs. (117)–(131), response paragraph on PDF p. 9 | Efficient FD eccentric/precessing waveform construction. | Explicit rigid-adiabatic XYZ TDI response; this is the response reference cited by Moore. Establishes prior FD eccentric LISA/TDI modelling while giving a concrete response approximation to distinguish from eGB's direct retarded links. No runtime from this paper is transferred to eGB. |
| [Tessmer and Gopakumar (2007), gr-qc/0610139v1](https://arxiv.org/abs/gr-qc/0610139v1), abstract and polarization construction | Established eccentric Galactic compact-binary polarizations on slowly precessing quasi-Keplerian orbits. | Existing source-model foundation already cited by our manuscript. Describe the present work as building on it; a new source-physics derivation or first TD eccentric waveform has not been established. |

Bibliographic identities are preserved in the evidence record. Xuan (2026) is recorded as the checked May 2026 preprint, submitted to PRD, without inventing a journal publication. Yunes is a 2009 paper; its v3 correction is from 2014. The dates generated in some arXiv HTML renderings are not used as publication dates.

## 3. eGB contribution and evidence boundaries

The [Task 5 source audit](referee_task_5_1_5_2_audit.md) supplies the implementation contract. Its inspected-source hashes are rechecked by the Task 4 record script. The following are defensible implementation attributes, rather than claims of being first or universally superior:

- Modular Newtonian/1PN conservative eccentric dynamics with quadrupole polarizations and fixed or isolated Peters evolution choices, with the inherited physics explicitly attributed.
- Evaluation on six retarded one-way links, followed by time-domain pyTDI observables, supporting a direct response reference under specified orbital, interpolation and TDI settings.
- A reproducible package workflow for examining the effect of source-model choices at the response level. Public/versioned reproducibility remains dependent on Task 1.
- The revised circular-family search and covariance/convergence evidence from Task 2. Numerical results across the intended production grid remain dependent on 2.9.

Use **direct retarded-link reference** with declared numerical settings. Calling a calculation “exact” must not imply exact source physics, exact interpolation or a complete calibrated instrument likelihood. The native configuration supplies retarded orbital geometry; the historical JAX geometry call without the `orbits` argument is a different path. Package options do not by themselves establish validation of every option.

The [frozen 2.8 configuration](../outputs/referee/task_2_8_20261012_completion/frozen_production_configuration.json) establishes representative annual numerical checks, full A/E covariance weighting and the stated ideal clocks/ranges reference. It does not establish physical adequacy for all Galactic populations, a rigorous global-maximum proof, a full multi-source search, or the pending production contours.

Claims to avoid: first eccentric LISA model; first inclusion of eccentric harmonics/precession/evolution/TDI; discovery of harmonic confusion; highest physical accuracy; universal eccentricity measurability threshold; demonstrated FD impracticality; fastest eccentric implementation. This focused review establishes relevant prior work, not a comprehensive priority certificate.

## 4. Existing performance and overlap evidence

The [figure driver](../benchmarks/plot_jaxgb_comprehensive_comparison.py:20) reads [the 1PN/periastron CSV](../benchmarks/jaxgbaccurate_calibrated_eccentric_study_1pn_periastron.csv). The [generation script](../benchmarks/jaxgbaccurate_calibrated_eccentric_study.py) and [comparison function](../benchmarks/production_eccentric_circular_mismatch_grid.py:123) were inspected, not executed. This is the local artifact lineage identified for the discussion; matching every manuscript image to that lineage remains a final-stage check.

| Benchmark fact | Consequence |
|---|---|
| Seven days, 4096 samples, cadence 147.65625 s; equal-arm orbit; 128 sparse response bins | These timings describe a short, coarse native workload, not the annual converged production setting. |
| eGB generation 1, JAXGB generation 1.5; historical geometry precomputation without native `orbits` | Different response contracts remain. A nonzero circular residual cannot be assigned uniquely to conventions, response approximations or numerical error without isolating them. |
| Independent fitted complex scales in A, E and T before taking real outputs | This calibration is not the common physically constrained source-orientation/amplitude maximization now used by Task 2. |
| Absolute normalized complex overlap at fixed intrinsic source parameters | CSV labels containing `fitting_factor` do not establish maximization over the full circular family. Do not use this as the revised FF evidence. |
| Raw circular residual is subtracted in the auxiliary “convention removed” columns | Baseline subtraction is a diagnostic definition, not proof that the physical or numerical discrepancy has been repaired. |
| Nyquist frequency is about 3.386 mHz, although the nominal mask extends to 10 mHz | At a 3 mHz carrier, the third source harmonic near 4.5 mHz is already above Nyquist. High-frequency/eccentric rows need convergence checks before serving as accuracy evidence. |
| Sparse circular versus dense eccentric native workloads, without matched target accuracy | The quoted runtime ratio cannot establish an eccentric FD/TD crossover or a general speed/accuracy comparison. Source-power harmonic counts also do not measure noise-weighted response accuracy or total cost. |

The [machine-readable audit](../outputs/referee/task_4_20261012/existing_benchmark_audit.json) records the CSV timing statistics and input hashes. These values describe existing artifacts; no benchmark was rerun, and no external waveform package was executed. The representative Task 2.8 checks use a different repaired pipeline and do not retroactively validate this old comparison.

Recommended final-stage action: remove the speculative speed/crossover argument. Omit this old figure as accuracy evidence unless it is repaired; if retained solely as an implementation diagnostic, accurately label its statistic, calibration, response differences, sampling and workload. A matched-accuracy eccentric comparison would be additional research requiring an explicit design decision.

## 5. Revision and response evidence to retain

At the final manuscript stage, implement the [4.5–4.7 plan](referee_task_4_subtasks.md). The reply should acknowledge the under-discussion of existing work, identify the added technical citations and explain the revised implementation contribution. It should describe actual changes rather than defending a blanket FD limitation.

Task 3 separately handles eccentricity measurability versus circular-template SNR loss. Task 5 supplies the physical applicability qualifications. Task 1 supplies release evidence. Task 2.9 supplies the revised full-grid numerical statements. These dependencies prevent the literature revision from becoming an unsupported science or performance claim.

Evidence preservation: seven pinned PDFs, layout and reading-order extraction, five inspected rendered pages, prepared bibliography additions, source/benchmark hashes and a saved research tracker. The initial two missing phrase locators were caused by a wording mismatch and two-column layout extraction; the initial record is retained under `history`, and the corrected reading-order extraction finds all locators. These checks support this review's provenance; they are not numerical physics validation.
