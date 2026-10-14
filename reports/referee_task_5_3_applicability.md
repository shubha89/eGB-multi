# Task 5 Astrophysical Applicability Assessment

Completed 12 October 2026. **Subtask 5.3 is complete as a qualitative applicability assessment grounded in the 5.1–5.2 code and literature audits.** It establishes the physical scope and the qualifications needed for 5.4–5.5. It does not calculate an omitted-effect error bound for the production grid or introduce additional source physics.

## Scope of the Present Model

eGB-multi provides a modular time-domain response framework for the source physics supplied to it. The current source implementation treats constant component masses with quadrupole polarizations on Newtonian or conservative 1PN quasi-Keplerian orbits, with optional leading Peters evolution. The fixed JAX path also supports a prescribed linear chirp. The [completed audit](referee_task_5_1_5_2_audit.md) records the backend-specific fdot contract.

The resulting waveforms are useful as controlled reference calculations within those approximations. Applying them to a particular astrophysical binary additionally requires that omitted stellar and environmental effects are sufficiently small for the observation and intended use. Accurate retarded links and TDI do not establish that condition. The framework's extensibility should be retained as a contribution, while the numerical experiments must be identified with their actual source and noise models.

The point-mass approximation also has its usual regime conditions: a bound orbit, weak-field orbital motion throughout the orbit, and sufficiently slow secular evolution for the adopted orbit-averaged Peters treatment. A general compact-binary application needs its own checks of neglected PN/spin terms and numerical settings. The representative Task 2 checks do not validate every source population or an unrestricted mass/frequency domain.

## Effects and Applicability Conditions

| Effect | What it can change | Applicability condition for the present baseline | Inputs needed for a quantitative check |
|---|---|---|---|
| Tidal and rotational apsidal motion | Total apsidal rate, phase evolution and harmonic sidebands | The unmodeled contribution must be small enough over the observation for the desired waveform accuracy. Detachment alone is insufficient. GR precession must be identified as the modeled contribution rather than total WD precession. [Willems et al.](https://doi.org/10.1103/PhysRevLett.100.041102) | Stellar radii, internal structure/apsidal constants and spin state, together with masses, orbital elements and observation duration |
| Tidal exchange of spin and orbital angular momentum | Frequency evolution and the physical interpretation of the chirp | Isolated Peters evolution is appropriate only where additional torques have negligible relevant effects or are intentionally excluded as a baseline assumption. [Stroeer and Nelemans](https://arxiv.org/abs/0909.1796) | A specified coupling prescription, spins, stellar structure and evolutionary state |
| Mass transfer and interacting systems | Component masses, mass ratio, orbital evolution and waveform amplitude | Roche-lobe overflow, accretion and mass exchange are outside the self-consistent present source model. A fitted chirp is not evidence that those processes are modeled. [Wang et al.](https://arxiv.org/abs/2302.08814) | Donor/accretor properties, transfer/loss law, angular-momentum transport and coupled evolution |
| Tertiary or other source CoM acceleration | Observed frequency and its time derivatives; parameter interpretation | The extra source Doppler evolution must be negligible for the application, or explicitly treated as a phenomenological approximation with checked residuals. LISA's known orbital Doppler response does not supply the source motion. [Xuan et al.](https://arxiv.org/abs/2012.00049) | Outer orbit or CoM trajectory, orientation, phase and observation interval; time variation of the acceleration where relevant |

These conditions are assessments obtained by reconciling the implementation with the cited studies. They are not newly measured validity thresholds. The effects also need not be independent: stellar tides can couple precession, spins and secular evolution, and interacting binaries require coupled source dynamics.

## Source Classes

| Source class or use | Assessment | Interpretation permitted by present evidence |
|---|---|---|
| Synthetic isolated point-mass injections | Directly represented within the selected Newtonian/1PN/Peters or prescribed-chirp contract | Study harmonic structure, implemented source corrections, detector response and recovery by the declared circular family. Numerical convergence remains conditional on the tested configuration. |
| Detached DWDs or WD–NS systems | A possible approximation whose adequacy must be checked for the actual stars and observation | Use WD-like mass examples as point-mass calculations. Do not infer that all detached systems have negligible finite-size effects, or that every synthetic grid point is a physically verified detached stellar binary. |
| AM CVn and other mass-transferring systems | Coupled stellar evolution is not represented | A generic short-observation chirp approximation may be useful after validation, but the current source does not predict donor/accretor evolution or support mass-transfer inference. |
| DWDs with tertiaries | Presence of a companion alone does not decide adequacy; the observable effect depends on its orbit and the observation | A sufficiently small source-motion effect can permit the baseline approximation. An effect absorbed by fitted parameters can still bias their astrophysical interpretation. |
| Other compact binaries, including NS/BH systems | The architecture remains applicable to source-model extensions and controlled point-mass examples | Check approximation order, omitted spin/environmental physics, search domain and source-specific numerical accuracy. The present WD-like production comparison is not a validation of a general NS/BH survey. |

No stellar radii, spin distribution, Roche-overflow calculation or outer-orbit population has been assigned to the synthetic production grid. Consequently, this assessment does not certify its astrophysical realizability point by point. That does not prevent reporting the controlled waveform comparison with its assumptions stated explicitly.

## What a Free Frequency Derivative Establishes

The fixed JAX prescription varies the carrier and phase while keeping component masses and eccentricity fixed. It is a phenomenological template freedom. It does not introduce an independent total apsidal rate, variable masses, a mass-transfer law, or a source CoM trajectory.

Fitted fdot can locally approximate a nearly constant extra observed chirp without identifying its cause. Time-varying acceleration can introduce higher-order phase terms, and parameter absorption can leave biased astrophysical inference. [Xuan et al.](https://arxiv.org/abs/2012.00049) supports this caution for circular-vacuum chirps; our independently fitted mass/fdot family needs its own analysis.

The NumPy source's ignored fdot and the non-fixed link path's rejection of nonzero fdot remain documented interface limitations. They have not been repaired in 5.3. The validated circular envelope and native verifier use the prescribed JAX convention explicitly.

## Observation and Frequency Conventions

Comparisons must use the actual observation duration and harmonics. A small omitted precession rate relative to the GR rate alone does not establish a small accumulated phase or waveform error over a long observation. Likewise, a slowly changing physical acceleration may be distinguishable through curvature, or remain degenerate at the available SNR. Relative rate comparisons are useful diagnostics, not complete accuracy criteria.

The manuscript grid's f0 follows the source carrier convention omega/pi. Orbital frequency is n/(2*pi), so comparisons require the appropriate frequency convention; in the Newtonian limit the carrier is twice the orbital frequency. The orbital-frequency examples in the WD precession literature cannot supply a direct cutoff in this grid's f0. Stellar assumptions and eccentricity also matter. No universal safe frequency is assigned here.

Fourier resolution 1/T is a scale for displaying spectral structure. It is not by itself a universal measurability criterion. The [published Willems erratum](https://doi.org/10.1103/PhysRevLett.101.219903) corrects the description of an earlier frequency-resolution estimate to a Fisher analysis. Its correction is consistent with keeping SNR-dependent measurement claims distinct from frequency-bin comparisons.

## Numerical Accuracy and Physical Model Adequacy

The working 0.001 absolute FF allowance in Task 2.8 concerns numerical convergence under the declared source family and instrumental-noise reference. It does not include a measured uncertainty from omitted tides, stellar rotation, mass transfer or environmental effects. The FF=0.9 contour in Task 2.9 will therefore remain a circular-template loss contour for that modeled injection family; it is not a universal threshold for the entire Galactic-binary population or eccentricity measurability.

If a source-specific omitted-effect claim is later needed, use the following validation procedure:

1. Declare the additional physical inputs and evolution model, including stellar properties or source motion. Check detachment/Roche overflow over the relevant orbit where applicable, rather than assuming it from the source label.
2. Generate the augmented and baseline responses with matching epochs, duration, units, detector/TDI convention, noise weighting and analysis band.
3. Separate the effect at fixed physical parameters from the effect after fitting the appropriate baseline source family. Record fitted-parameter shifts as well as residuals.
4. Use an application-specific accuracy or inference criterion. A small fitted residual alone does not establish unbiased parameters, and a source-model adequacy comparison is distinct from fitting an eccentric signal with circular templates.

This is a proposed validation procedure, not a completed calculation. It is unnecessary for the present acknowledgement-and-scope revision unless we decide to claim quantitative validity for particular real systems.

## Requirements for the Manuscript Draft

The 5.4 draft and 5.5 edits should implement these conclusions:

- State the isolated point-mass, constant-mass source assumptions and distinguish GR apsidal motion from total stellar apsidal motion.
- Explicitly acknowledge interacting binaries and source acceleration as omitted physics with possible consequences for phase evolution and inferred parameters.
- Restrict claims about dominant corrections, small eccentricity decay and the automatic switch to the implemented model and tested examples. Preserve the broader modular-framework contribution.
- Use the carrier-frequency convention consistently; coordinate the corrected grid labels and numerical results with Tasks 2.9–2.10.
- Describe future source extensions as extensions, without implying that the corresponding physics is currently implemented or numerically negligible.

## Completion and Next Step

The applicability matrix, observational qualifications and limits on astrophysical inference complete **5.3**. No new stellar parameters, omitted-effect forecasts or code changes were required. The [5.1–5.2 evidence](referee_task_5_1_5_2_audit.md) remains the input record. Next is **5.4**, drafting the model-scope and applicability text; actual manuscript edits and the reply remain 5.5–5.7.
