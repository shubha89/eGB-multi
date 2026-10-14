# Task 5 Astrophysical Applicability Plan

Manuscript: *Efficient time-domain eccentric model for Galactic binaries in LISA* (DV14615). Created 12 October 2026.

Task 5 addresses the applicability of the implemented source physics to Galactic binaries. The referee asks us to acknowledge tidal and rotational apsidal motion in white dwarfs, interacting systems such as AM CVn binaries, and tertiary acceleration. The report permits acknowledgement without implementing these effects. Completion requires accurate model boundaries, supported manuscript changes and a reply tied to those changes.

The planned revision preserves the package's scope as a modular compact-binary waveform framework. It separates what the source model can generate from what the current numerical experiments demonstrate. Results obtained with isolated point-mass dynamics must be described as conditional on that source model. Detachment alone does not establish that finite-size effects are negligible.

## Subtask Tracker

| ID | Subtask | Deliverable and completion criterion | Depends on | Status |
|---|---|---|---|---|
| **5.1** | Audit the implemented source physics and manuscript claims | Included/omitted-physics table, NumPy/JAX dispatch checks, source references and affected manuscript passages are saved in the [audit report](referee_task_5_1_5_2_audit.md). Constant masses, absent stellar/environmental variables, fdot path asymmetry, auto-switch scope and orbital/carrier labeling are documented. Short runtime checks and inspected-source hashes are preserved. | None | **Complete at source-contract and targeted claim-audit scope** |
| **5.2** | Verify the primary literature for each omitted effect | Five primary documents, including the published precession article and erratum, have been checked at relevant sections/equations. The [evidence report](referee_task_5_1_5_2_audit.md) records assumptions and conventions; archived PDFs, locators and five prepared bibliography entries are saved. No physical omitted-effect forecast is inferred for our grid. | None; can overlap with 5.1 | **Complete at focused primary-literature scope** |
| **5.3** | Define applicability and the consequences of omissions | [Applicability assessment](referee_task_5_3_applicability.md) saved: physical-effect and source-class matrices, observation/frequency conventions, limits of free fdot, and separation of numerical convergence from physical model adequacy. Preserves the modular framework and qualifies the present point-mass experiments. No stellar parameters or universal frequency cutoff assigned. | 5.1–5.2 | **Complete at qualitative applicability-assessment scope** |
| **5.4** | Draft the astrophysical applicability text | A compact model-scope paragraph and a discussion passage covering all three objections, with primary citations. Explicitly distinguish GR apsidal motion from total stellar apsidal motion and isolated Peters evolution from interacting/environmental evolution. | 5.3 | Deferred to final manuscript stage |
| **5.5** | Apply the scope consistently across the paper and documentation | Update the introduction, model description, source-correction discussion and conclusions where claims are too broad. Check statements about dominant corrections, negligible eccentricity evolution and mass interpretation. Add an appropriate package-documentation limitation; flag related abstract wording for Task 7. | 5.4 | Deferred to final manuscript stage |
| **5.6** | Draft the point-by-point response to Comment 5 | A reply explaining the acknowledged effects, the adopted source-model scope and each actual manuscript change, with final section/page references. Do not claim new effects were implemented or their numerical impact was calculated unless that work exists. | 5.5 | Deferred to final manuscript and response stage |
| **5.7** | Verify and close Task 5 | Compile and inspect the revised PDF; verify citations and cross-references, match every reply claim to an actual edit, and check that no unsupported validity claim remains. Keep Task 5's scope edits distinct from Task 2's pending numerical results. | 5.5–5.6 | Deferred to final manuscript checks |

**5.1–5.3 are complete at their stated audit and qualitative assessment scopes. 5.4–5.7 are deferred to the final manuscript and response stage.** The user requested on 12 October that manuscript checks be left until the end. The [completed audit](referee_task_5_1_5_2_audit.md) and [applicability assessment](referee_task_5_3_applicability.md) preserve the research evidence for that consolidated stage.

## Implementation Order

With **5.1–5.3 complete**, continue the other research, literature and computation tasks. Scientific validation of code and runs continues as needed. At the final manuscript stage, draft and apply the text in **5.4–5.5**, prepare the reply in **5.6**, and perform compilation, citation, consistency and rendered-PDF checks in **5.7**.

Task 5 can proceed independently of the Task 2.9 production grid. It does not require that grid to finish before we document the source assumptions. Quantitative fitting-factor conclusions will still be finalized through Tasks 2.9–2.10, and the abstract through Task 7.

Planning estimate: approximately half a working day for the source audit, focused literature verification, edits and PDF/reply checks. This is an effort estimate, not a measured runtime. No large numerical campaign is part of the initial Task 5 scope.

## Verified Audit Findings

The current [source module](../src/egb_multi/eccentric.py) describes quadrupole polarizations evaluated on Newtonian or 1PN quasi-Keplerian orbital dynamics and supplies fixed/Peters evolution modes. The audit confirms that NumPy source functions ignore fdot, fixed JAX sources implement its prescribed chirp, and the non-fixed exact-link path rejects it. The circular-search envelope and native verifier explicitly use the JAX convention. An adjustable derivative must not be presented as a self-consistent tidal, mass-transfer or tertiary model.

The corresponding [JAX implementation](../src/egb_multi/eccentric_jax.py) has been checked at the source/dispatch contract. The [README](../README.md) and [manuscript source](Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex) still need the coordinated documentation edits in 5.5. The current production text also calls f0 an orbital frequency although its constructor uses the GW carrier convention.

The manuscript passages requiring attention include:

- The introduction's motivation from several Galactic-binary formation channels and its description of the package's applicability.
- The model's treatment of GR periastron advance and isolated radiation reaction.
- The discussion of precession and secular corrections, including claims about their importance for white-dwarf-like systems.
- The conclusions' statements about dominant corrections and applications to broader compact-binary populations.

## Literature and Physical Boundaries

| Effect | Verified primary reference | Established condition for the draft |
|---|---|---|
| Tidal and rotational apsidal motion | [Willems, Vecchio and Kalogera, arXiv:0706.3700](https://arxiv.org/abs/0706.3700) | Distinguish GR from total stellar precession. Small omitted effects require observation-specific checks and stellar inputs; detachment and frequency alone do not establish adequacy. The orbital/carrier convention and erratum are reconciled. |
| Mass transfer and tidal exchange in AM CVn systems | [Stroeer and Nelemans, arXiv:0909.1796](https://arxiv.org/abs/0909.1796) | State that coupled interacting evolution is omitted. A fitted chirp is a waveform approximation, not a mass-transfer or torque prescription. |
| Tertiary acceleration and parameter interpretation | [Xuan, Peng and Chen, arXiv:2012.00049](https://arxiv.org/abs/2012.00049) | Known detector Doppler motion does not supply source acceleration. An effect absorbed by fitted parameters can still bias their physical interpretation. |

The source-specific evidence table is complete in the [audit report](referee_task_5_1_5_2_audit.md), with an additional circular conservative-transfer reference, Wang, Cao and Zhang (2023). Bibliographic metadata and the published precession erratum have been checked. The [5.3 assessment](referee_task_5_3_applicability.md) reconciles applicability without claiming a numerical omitted-effect forecast. The [referee report](</Users/shubhanshutiwari/Desktop/Yourmanuscript DV14615 Tiwari.pdf>) supplies the requested revision scope.

## Decisions That May Need Input

The initial path is to acknowledge the omissions and qualify applicability without adding source physics. This follows the referee's stated allowance and does not require a new decision now.

A decision is needed only if we choose to add a quantitative omitted-effect diagnostic or extend the model. For a WD precession comparison, we would need declared radii/stellar structures and spin assumptions before claiming that tides or rotation are negligible in particular examples. If the audit exposes a conflict with a central physical interpretation, we will identify the affected claim and choose between a supported qualification and a targeted calculation.

## Completion Record

- 5.1 complete at source-contract and targeted manuscript claim-audit scope; existing fdot interface asymmetry documented without changing the implementation.
- 5.2 complete at focused primary-literature scope; five primary documents and five bibliography entries checked.
- 5.3 complete at qualitative applicability-assessment scope; no new physical validity boundary or stellar/environmental calculation claimed.
- 5.4–5.7 deferred to the final manuscript and response stage by the user's updated workflow preference.
- Manuscript changes: not yet applied.
- Referee reply: not yet drafted.
- Numerical source implementation: unchanged by this planning step.
