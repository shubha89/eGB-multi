# Task 5 Source Physics and Literature Audit

Completed 12 October 2026. **5.1 and 5.2 are complete at the source-contract and focused primary-literature audit scopes.** The manuscript changes, applicability assessment and referee reply remain 5.3–5.7. This audit changes no waveform implementation or production results.

## Source Model Contract

The eGB source consists of quadrupole polarizations evaluated on Newtonian or conservative 1PN quasi-Keplerian dynamics. Its secular evolution uses leading quadrupole Peters equations with fixed component masses. The detector response evaluates these polarizations at retarded link times. These are distinct pieces of the calculation: accurate detector geometry and TDI do not add omitted stellar or environmental dynamics.

| Ingredient | Implemented behavior | Code evidence and qualification |
|---|---|---|
| Newtonian and conservative 1PN orbit | Selectable orbital corrections and GR periastron advance | [Physics switches](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/src/egb_multi/eccentric.py:185) and [dynamics](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/src/egb_multi/eccentric.py:862). The precession coefficient is fixed by point-mass GR, not an independently adjustable total stellar apsidal rate. |
| Polarizations | Quadrupole expressions evaluated using the selected orbit | [Polarizations](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/src/egb_multi/eccentric.py:1012). This is not a complete general higher-PN amplitude model. |
| Radiation reaction | Peters radial-frequency/eccentricity evolution and integrated phase | [Derivatives and ODE](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/src/egb_multi/eccentric.py:608). Leading dissipative evolution is combined with the chosen conservative orbit; no stellar torque or mass-exchange equations are present. |
| Prescribed linear chirp | Fixed JAX sources use omega(t)=omega0+pi*fdot*(t-t0); NumPy source functions ignore fdot | [JAX kernel](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/src/egb_multi/eccentric_jax.py:358) and [NumPy dynamics](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/src/egb_multi/eccentric.py:862). This is a path-dependent phenomenological prescription, not an environmental model. |
| Non-fixed link evolution | Peters link fallback rejects nonzero prescribed fdot | [Fallback guard](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/src/egb_multi/eccentric_jax.py:234). A Peters trajectory and an additional fdot cannot be combined through this interface. |
| Component masses and stellar structure | Scalar, constant masses; no stellar radius, apsidal constant, spin or composition parameters | [Source dataclass](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/src/egb_multi/eccentric.py:23). Orbital separation r is not a WD stellar radius. Source polarization psi is not stellar rotation. |
| Tides and rotational distortion | Omitted | GR orbital precession contains no finite-size or stellar-spin terms. No tidal dissipation or spin/orbit angular-momentum exchange is integrated. |
| Interacting systems | Mass transfer and Roche-lobe overflow omitted | Constant masses and Peters derivatives do not model AM CVn donor/accretor evolution, changing mass ratio, or accretion torques. |
| Source environment | No source CoM worldline, tertiary orbit, acceleration or jerk | Fixed sky/distance and [link projection](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/src/egb_multi/eccentric_jax.py:234). LISA receiver/emitter motion supplies detector Doppler modulation, not source acceleration. |
| Automatic evolution switch | Compares fixed and Peters sources within the selected physics | [Switch calculation](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/src/egb_multi/eccentric.py:792). Its tolerance does not bound omitted tides, mass transfer or environmental effects. |

## Short Runtime Checks

The [audit driver](../benchmarks/referee_task_5_source_audit.py) used this checkout's eGB package, 2048 source samples over 100000 s and a deliberately large prescribed fdot=1e-11 Hz/s. This is a dispatch/sensitivity diagnostic, not an astrophysical injection or noise-weighted fitting factor.

| Check | Observed result |
|---|---|
| Change fdot in the NumPy Newtonian and 1PN fixed source | Both outputs remain bit-identical; relative L2 change zero |
| Change fdot in the NumPy Peters source | Output remains bit-identical; relative L2 change zero |
| Change fdot in fixed JAX sources | Relative L2 changes 0.14442482 (Newtonian) and 0.14442095 (1PN) |
| NumPy/JAX comparison at fdot=0 | Relative L2 differences 1.16e-15 and 1.32e-16 for the selected Newtonian and 1PN examples |
| Independent circular Newtonian prescribed-chirp formula versus JAX | Relative L2 difference 3.00e-14 |
| Nonzero fdot with Peters exact-link evolution | Explicit ValueError, matching the dispatch contract |

The fdot asymmetry is an existing interface limitation. The circular-search envelope already implements the prescribed JAX chirp explicitly ([envelope construction](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/src/egb_multi/circular_envelope.py:99)); its native winner verification calls the fixed JAX link path ([native response](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/src/egb_multi/native_validation.py:66)). The archived final high-PM Peters injection has fdot=0. The frozen 2.8 evidence hashes still verify. These checks explain why this finding does not by itself invalidate the validated circular search, while generic source documentation must avoid implying uniform fdot support.

For fixed JAX sources with nonzero fdot, eccentricity and component masses stay fixed but frequency, phase and frequency-dependent orbital quantities change. The radial/azimuthal chirp ratio uses the GR precession coefficient at the reference epoch. Thus the label fixed does not mean a strictly monochromatic source for this prescribed-chirp interface, and this prescription is not a self-consistent Peters or stellar-evolution solution.

Machine-readable results and inspected-source hashes are saved in [source_audit.json](../outputs/referee/task_5_1_5_2_20261012/source_audit.json). No source repair is claimed by this audit.

## Manuscript Claims Requiring Qualification

| Location in the current source | Finding | Required treatment in later subtasks |
|---|---|---|
| [Abstract](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:108) | Detached WD-like binaries are associated with a hierarchy in which eccentric harmonic structure dominates and eccentricity decay is negligible. | Attach the hierarchy to the tested point-mass model/examples. Detachment is not proof that total stellar precession or dissipative tides are negligible. Coordinate final abstract wording with Task 7. |
| [Introduction](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:146) | Several formation channels motivate general Galactic-binary use, without an explicit boundary for ongoing interactions. | Distinguish formation history from dynamics during observation; state the modeled source assumptions. |
| [Model motivation](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:300) | Broad applicability is suggested by analogy with relativistic pulsar binaries. | Keep the relativistic contribution, but acknowledge that WD finite-size contributions can compete with it. |
| [Evolution equations](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:396) | Equations contain Peters derivatives and 1PN phase reconstruction. | State constant masses, isolated point-mass radiation reaction and omitted environmental/stellar terms. |
| [Precession and frequency resolution](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:582) | GR splitting is compared with 1/T and described as resolvable. | Distinguish GR splitting from total stellar splitting. Fourier-bin comparison is a scale check, not a complete SNR-dependent measurability test; coordinate with Comment 3. |
| [Secular-correction discussion](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:743) and [optional corrections](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:935) | Radiation-reaction hierarchy and the auto switch are presented as an accuracy choice. | Qualify these as comparisons within the implemented fixed/GR/Peters family, rather than astrophysical completeness. |
| [Production grid description](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:1030) | f0 is called an initial orbital frequency. | The [grid source constructor](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/benchmarks/production_eccentric_circular_fdot_tdi2_mismatch_grid.py:98) sets mean_motion=pi*f0. Label it as the source GW carrier convention; reconcile this in 5.5/Task 2.10. |
| [Conclusions](/Users/shubhanshutiwari/IdeaProjects/eGB-multi/reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex:1246) | Small eccentricity decay and subdominant precession can read as general detached-binary results. | Retain demonstrated results, but identify the modeled parameter choices and omitted effects. Extensibility is not evidence that these effects are already implemented. |

## Verified Primary Literature

Full relevant sections and equations were checked in the archived PDFs, including visual checks of the key pages. Numerical population forecasts from these studies are not adopted as predictions for our grid.

| Source | Supporting locations | Verified finding and limits |
|---|---|---|
| [Willems, Vecchio and Kalogera 2008](https://doi.org/10.1103/PhysRevLett.100.041102) | Published pp. 041102-2 and 041102-4; Eqs. (1)–(3), Fig. 1 and astrophysical implications | DWD apsidal motion receives tidal, rotational and GR contributions. The stellar terms depend on radii, internal structure and spin. Fig. 1 uses orbital frequency nu=1/P, specified WD masses, zero-temperature radii, apsidal constants and periastron-synchronized rotation. Its crossover is not a universal frequency boundary. |
| [Published erratum](https://doi.org/10.1103/PhysRevLett.101.219903) | Entire one-page correction | Corrects the description of Seto's frequency-resolution method: it used a Fisher analysis. It does not revise the tidal/rotational equations or remove the finite-size applicability caveat. |
| [Stroeer and Nelemans 2009](https://arxiv.org/abs/0909.1796) | Sec. 2.1, Eqs. (1)–(8); Sec. 2.2, Eqs. (13)–(14); Sec. 4 | A coupled AM CVn model includes mass transfer and tidal spin/orbit exchange. Perturbations alter the frequency derivatives and their astrophysical interpretation. The model defines fGW=2/P; it is a specified interacting-system case study, not a universal prescription for eccentric DWDs. |
| [Wang, Cao and Zhang 2023](https://arxiv.org/abs/2302.08814) | Sec. 2, especially Eqs. (14)–(25) | A circular, conservative-transfer model changes the chirp relation even when GW radiation is the only external angular-momentum loss. It imposes restrictions on stellar rotation/spin exchange. This supports the mass-transfer caveat, not a claim that every tidal effect is modeled by that study. |
| [Xuan, Peng and Chen 2021](https://arxiv.org/abs/2012.00049) | Sec. 2, Eqs. (5)–(12); Sec. 3 confusion criteria; Sec. 7 | Source CoM acceleration changes the observed chirp and can bias mass/distance interpretation. The leading degeneracy is conditional on the observation and outer-orbit evolution; curvature can make acceleration distinguishable. Their chirp-mass relation assumes a circular vacuum binary, so it cannot be applied verbatim to our independently fitted fdot and masses. |

### Frequency Conventions

In this checkout, mean_motion is a source angular carrier in rad/s. With precession enabled it denotes the azimuthal omega; the radial angular frequency n is reconstructed at the retained PN order. Therefore fcarrier=omega/pi, whereas forbital=n/(2*pi). In the Newtonian limit, fcarrier=2*forbital. The precession literature's orbital-frequency axis must not be compared directly with the manuscript's f0 labels.

The package's GR coefficient k and the WD paper's stellar apsidal constants are different physical quantities. Neither a constant fdot nor a polarization angle supplies the missing stellar variables. These are implementation deductions, not claims that the omitted contributions are small.

### Citation Readiness

The existing manuscript's Willems2007 entry concerns eccentric-binary formation, not the 2008 finite-size precession study. Separate entries are prepared in [reference_additions.bib](../outputs/referee/task_5_1_5_2_20261012/reference_additions.bib), including the erratum. Journal metadata for the three MNRAS papers was checked against DOI records. The additions have not been inserted into the manuscript bibliography yet; that is part of 5.5.

The [literature inventory](../outputs/referee/task_5_1_5_2_20261012/literature_evidence.json) records versions, locators, metadata sources and PDF hashes. The [manifest](../outputs/referee/task_5_1_5_2_20261012/manifest.json) preserves the audit inputs and outputs.

## Next Step

Proceed to **5.3** using the code and literature tables above. No decision is required to acknowledge these omissions. A numerical claim that tides or rotation are negligible in specific WD examples would require explicit stellar-radius, structure and spin assumptions; it is not established by this audit. The revised applicability wording and any optional calculation must keep that distinction.
