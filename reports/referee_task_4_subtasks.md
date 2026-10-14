# Task 4: Literature and contribution assessment

Manuscript: *Efficient time-domain eccentric model for Galactic binaries in LISA* (DV14615). Created 12 October 2026.

Comment 4 requests fuller acknowledgement of existing eccentric waveform and LISA data-analysis work, an accurate discussion of frequency-domain constructions, and a clear statement of this paper's contribution. This is principally a literature and positioning task. No new physical ingredients or large numerical campaign are required to address the initial objection.

The user requested that manuscript work be consolidated at the end. The research below is complete at a focused assessment scope; manuscript edits, the finished response and final verification remain deferred.

## Subtask tracker

| ID | Work | Evidence and completion criterion | Dependency | Status |
|---|---|---|---|---|
| **4.1** | Map the referee's demands to existing claims and citation roles | Audit Moore's citation role, source-physics attribution, the end-of-Section-IV discussion and the benchmark supporting the performance discussion. Save affected source locations without applying edits. | Referee report and current source | **Complete at targeted claim-audit scope** |
| **4.2** | Verify the requested primary literature and close the precession gap | Check the four requested papers plus Tessmer–Gopakumar, Klein et al. (2018) and Klein (2021). Preserve pinned PDFs, text, section/equation locators and bibliographic identities. | 4.1 | **Complete at focused primary-literature scope** |
| **4.3** | Compare capabilities and observables | Separate source dynamics, detector response/TDI, numerical construction, Bayesian inference, harmonic confusion and fully maximized FF. Record approximation boundaries rather than generic yes/no feature claims. | 4.2 and the existing Task 5 source audit | **Complete at capability-assessment scope** |
| **4.4** | Define defensible contributions and unsupported claims | Identify eGB's specific source-to-link-to-TDI implementation, modularity and reference calculations. Record which claims require Task 1 release verification or Task 2.9 numerical results. Audit why the existing timing comparison cannot establish a general advantage. | 4.1–4.3 | **Complete at contribution-boundary scope** |
| **4.5** | Apply the literature and positioning revision | Expand Moore's discussion beyond astrophysical motivation; add the missing references; rewrite the frequency-domain paragraph; accurately scope or remove the old performance comparison. Keep package and source-physics claims consistent. | 4.4; 2.9 for any new numerical conclusions | Deferred to final manuscript stage |
| **4.6** | Prepare the response to Comment 4 | Acknowledge each point and identify actual revised passages and citations with final section/page references. Do not claim that an external-code equivalence benchmark or new inference study was performed. | 4.5 | Deferred to final response stage |
| **4.7** | Verify and close Comment 4 | Check citations, claim-to-evidence consistency, figure provenance and terminology; compile and inspect the revised manuscript; reconcile the reply with actual changes. | 4.5–4.6 and final results | Deferred to final manuscript checks |

**4.1–4.4 are complete at their stated research scopes. Comment 4 remains open until 4.5–4.7 are finished.** The [literature and contribution assessment](referee_task_4_literature_novelty.md) contains the evidence and proposed revision actions. Archived sources, the existing benchmark audit and a reproducible provenance script are in [the Task 4 evidence directory](../outputs/referee/task_4_20261012) and [record script](../benchmarks/referee_task_4_record.py).

## Order and decisions

The initial research proceeds independently of Task 2.9. At the final stage, implement 4.5, then draft 4.6 and verify 4.7. Task 3 will use the same literature but separately address the difference between circular-template SNR loss and SNR-dependent eccentricity inference.

Recommended direction: acknowledge the prior waveform/data-analysis results and describe the specific eGB implementation. Remove unsupported general statements about frequency-domain impracticality or performance crossover. This uses the currently authorized revision scope and needs no new scientific decision.

A separate decision would be needed to undertake a matched-accuracy external-package benchmark. Such a study is optional, must specify equivalent physics/response/output/accuracy, and would require reconciling the user's instruction to use only eGB for runs. No external waveform package was run for this task. The existing comparison should not be retained as an accuracy or speed validation without further work.

Planning estimate for the deferred work: several hours for coordinated text/citation edits and reply verification, assuming no new benchmark is requested. This is an effort estimate, not a measured runtime.
