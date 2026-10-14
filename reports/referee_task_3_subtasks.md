# Task 3 Eccentricity Threshold Interpretation

Created 12 October 2026 for DV14615. Comment 3 requests that the chosen `1-FF=0.1` contour be distinguished from an eccentricity-detection or measurement threshold, whose statistical meaning depends on SNR. It also requests comparison with the existing eccentric Galactic-binary Bayesian analysis.

The [3.1–3.4 assessment](referee_task_3_1_3_4_threshold_interpretation.md) preserves the wording audit, statistical definitions, high-SNR assumptions and focused literature comparison. These research tasks are complete at the scopes below. No new waveform, noise or parameter-estimation simulation was performed.

| ID | Work | Completion evidence or requirement | Status |
|---|---|---|---|
| **3.1** | Audit threshold and measurability wording | Submitted PDF and current local source audited separately; affected passages and legitimate unrelated uses of “threshold” identified. The submitted abstract already identifies 10% SNR loss; its broader “adequate” language needs qualification. | **Complete at targeted wording-audit scope** |
| **3.2** | Define the FF contour as circular-template SNR loss | Coherent A/E inner product, declared circular family, common amplitude, recovered SNR and residual identities stated; median contours distinguished from individual-source/population thresholds. | **Complete at mathematical and interpretation-contract scope** |
| **3.3** | Explain SNR-dependent distinguishability | Primary waveform-accuracy, Fisher confidence-region and model-comparison references checked. Exact zero-noise residual/likelihood relation separated from approximate high-SNR criteria; no effective dimension or universal significance assigned. | **Complete at focused statistical-framework scope** |
| **3.4** | Compare with Moore et al. | Criterion, source/noise/response assumptions, duration, inference and frequency conventions compared. Eq. (21) evaluated within its inspected domain and checked against Fig. 3. These are published-fit reference values, not reproduced inference or matched-configuration agreement. | **Complete at focused literature-comparison scope** |
| **3.5** | Finalize the conclusions supported by our revised FF results | Use 2.9's revised contours to state the SNR-loss result, its family/reference dependence and limits. Preserve the distinction from eccentricity measurement; decide which literature reference values belong in the final discussion. | Pending; requires **2.9** for numerical conclusions |
| **3.6** | Apply edits, prepare the reply and verify consistency | Coordinate terminology, equations, captions, Table I, abstract and conclusions; cite the literature; tie the response to actual edits and final page references; compile and inspect. | Deferred to final manuscript/response stage; requires 3.5 |

**3.1–3.4 are complete at their stated research scopes; Comment 3 remains open.** A new Bayesian pipeline is not required by the adopted revision path. Task 3.5 awaits production results; 3.6 follows the user's preference to consolidate manuscript work at the end.

The [evidence directory](../outputs/referee/task_3_20261012) contains pinned primary papers, submitted/referee PDFs, wording locators, formula reference values, selected-page inspections and provenance. The [record script](../benchmarks/referee_task_3_record.py) performs extraction, scalar formula evaluation and evidence checks; it runs no waveform or inference package.
