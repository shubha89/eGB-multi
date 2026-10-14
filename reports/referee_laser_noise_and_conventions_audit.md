# Laser cancellation and LISA convention audit

## Outcome

**Follow-up qualification:** the [paired laser-on/off test](referee_laser_on_off_validation.md) passes with ideal clocks/ranges, but finds a material laser-dependent residual with raw default ranging noise. Disabling ranging noise removes that effect. The laser-only results below establish numerical cancellation under their stated assumptions; neither they nor low median all-default A/E coherence certify negligible laser noise with every raw default enabled.

The laser-dependent near-unity A/E coherence was reproduced with **laser noise alone, static unequal arms and perfect clocks**. It is caused by inadequate numerical interpolation in the tested chain. Both instrument propagation and TDI measurement interpolation must be improved; changing only one stage leaves a large residual. The instrument-to-PyTDI carrier adapter agrees exactly with PyTDI's native adapter.

Two independent response inconsistencies were also found and corrected in the revision pipeline: ecliptic source coordinates were used with native ICRS orbit positions, and moving links used instantaneous emitter positions rather than positions at emission. These are separate from the laser-interpolation problem.

The ideal-clock/ideal-ranging noise cancellation checks are complete over the stated 0.1–30 mHz band and finite runs. The paired follow-up leaves raw ranging calibration unresolved. These checks do not establish a clock-calibrated mission sensitivity or complete the production waveform/optimizer convergence task.

## Controlled experiment and causal isolation

The [driver](../benchmarks/referee_laser_cancellation.py) uses the local eGB checkout, LISA Instrument and PyTDI. LISA Instrument's `disable_all_noises(excluding=...)` selects contributions explicitly. Laser-only cases remove all clock imperfections, ranging noise, test-mass, optical metrology and other stochastic terms; a separate case retains default clocks. All retain the N1-12 laser-locking configuration. Carrier TDI includes SCI, TMI and REF, uses generation 2 and `unit='frequency'`, and computes delay derivatives.

Initial/refinement runs cover 327680 s (3.79 days), with 81920 s Welch segments. The band comparisons below use the nearest PSD bin to 10 mHz (9.99756 mHz). Endpoints without complete nested-delay/interpolation support are excluded using the recorded halo rule.

| Case | Instrument / TDI measurement orders | Cadence | A PSD at 9.99756 mHz, fractional frequency squared/Hz |
|---|---:|---:|---:|
| Static arms, laser only | 3 / 7 | 10 s | 1.9382e-32 |
| Moving geometric arms, laser only | 3 / 7 | 10 s | 1.9755e-32 |
| Moving arms, default clocks retained | 3 / 7 | 10 s | 2.1851e-32 |
| Improve TDI alone | 3 / 45 | 10 s | 2.0487e-32 |
| Improve instrument alone | 45 / 7 | 10 s | 8.3415e-37 |
| Improve both | 45 / 45 | 10 s | 1.2824e-49 |
| Refine cadence | 45 / 45 | 5 s | 4.7263e-50 |
| Refine cadence further | 45 / 45 | 2.5 s | 4.6276e-50 |

Orders 15, 31 and 45 were tested at 10 s. Even order 45 at 10 s leaves excessive residuals near 30 mHz; passing the 10 mHz check alone is insufficient. The 5 s and 2.5 s configurations suppress the residual throughout the tested band.

The [initial](../outputs/referee/laser_audit_20261007_initial/summary.json) and [refinement](../outputs/referee/laser_audit_20261007_refine/summary.json) results retain configurations, PSDs and carrier/delay adapter comparisons. These early experiments used a generated instantaneous-geometric-arm model to preserve the original diagnostic setup. They are superseded as references by the native-retarded-orbit confirmation below.

## Confirmation with native retarded orbits

Five [confirmation runs](../outputs/referee/laser_audit_20261007_confirm/summary.json) use native, unmodified analytic `lisaorbits` LTT fields and derivatives, duration 1310720 s (15.17 days), and 81920 s Welch segments. The generated analytic orbit file is not a mission ephemeris. Tests use 5 s/order 45, 2.5 s/order 45 and 5 s/order 63 for laser-only noise, plus 5 s/order 45 secondary-only and all-default noise models. The latter two keep every native default amplitude for the selected contributions.

Residual-to-secondary PSD ratios on the common frequency grid over **0.1–30 mHz** are:

| Laser-only configuration | A median / maximum | E median / maximum |
|---|---:|---:|
| 5 s, order 45 | 3.99e-8 / 5.69e-7 | 4.11e-8 / 5.41e-7 |
| 2.5 s, order 45 | 4.94e-9 / 3.34e-7 | 5.29e-9 / 4.07e-7 |
| 5 s, order 63 | 3.87e-8 / 6.24e-7 | 4.08e-8 / 6.52e-7 |

These are finite-realization PSD ratios, not exact population-noise transfer functions. Refinement changes the tiny residual floor, but it remains negligible compared with secondary noise. This validates adequate cancellation for these synthetic simulations and band, without claiming that the residual itself has converged to zero.

The corrected all-noise median magnitude-squared A/E coherence is **0.02890**, versus **0.02532** for secondary noise. There are 30 overlapping Welch segments; their approximate independent-segment estimator scale is 1/30, with overlap correlations preventing an exact interpretation as independent averages. Near-unity broadband coherence has disappeared. Coherence of the laser-only residual itself can still be large: once the residual is negligible, that is not evidence of inadequate suppression.

The [summary JSON](../outputs/referee/laser_audit_20261007_summary/summary.json) contains all ratios and response errors. The [four-panel plot](../outputs/referee/laser_audit_20261007_summary/laser_cancellation_and_coherence.png) uses unsmoothed PSD estimates and was visually inspected.

## Convention audit and fixes

| Boundary | Verified convention / action |
|---|---|
| Link indexing | `ij` means receiver i, emitter j. Local sender/receiver order maps to `21,32,13,31,23,12`; native file order is `12,23,31,13,32,21`. No swap required. |
| Instrument carrier data | SCI/TMI/REF carrier fluctuations agree exactly with `Data.from_instrument(..., signals='fluctuations')`. Native sidebands are additional calibration inputs and are not used in carrier-only Michelson TDI. |
| Delay data | MPRS and their numerical derivatives agree exactly with the native adapter. No sign or central-frequency factor is added to modern native instrument fluctuations. |
| TDI observable | Generation 2, frequency Doppler-delay operators, positive delays, retained derivatives; A=(Z-X)/sqrt(2), E=(X-2Y+Z)/sqrt(6), T=(X+Y+Z)/sqrt(3), as in PyTDI. |
| Units | GW links are dimensionless fractional frequency; native instrument streams are Hz. Noise PSD converts by dividing by the square of the recorded central laser frequency, 2.816e14 Hz. |
| Sky/position frame | Source beta/lambda and polarization basis use barycentric mean ecliptic J2000. `lisaorbits.compute_position` returns ICRS. `state_from_lisaorbits` now converts to ecliptic by default using Astropy's transformation, including frame bias. Explicit `frame='icrs'` retains native positions for archival reproduction. |
| Polarization | At identical Cartesian sky coordinates, the local cross tensor is minus the native LISA GW Response cross tensor. The independent check rotates the full physical polarization tensor into ICRS and projects into native plus/cross bases; it does not identify psi or phi blindly between packages. |
| Moving GW links | Optional `precompute_jax_link_geometry(..., orbits=provider)` uses native reception-time LTT and the emitter position at t-L. The revision driver enables this path. Historical instantaneous geometry remains available explicitly as an approximation. |
| Source epoch | GW strain is evaluated at barycentric retarded TCB times. Native response uses `shift_sun='never'`; no fitted time shift is used in the comparison. |

The [independent checks](../tests/test_lisa_conventions.py) compare the same local eccentric strain against LISA GW Response's response implementation at 1 mHz/e=0, 10 mHz/e=0.3 and 30 mHz/e=0.1, with nontrivial orientations. They independently test response conventions rather than replacing the eGB source dynamics.

Maximum relative L2 errors are **6.95e-14 in the six links** and **3.13e-12 in A/E**. The weak T channel has a largest relative error of 1.14e-9. Historical instantaneous geometry changes link responses by about 1.3–1.5e-4; omitting the frame conversion gives order-unity response differences in these fixed-source examples. These are waveform differences, not optimized fitting-factor losses.

## Code changes and validation

- [Noise helpers](../src/egb_multi/noise.py): include local metrology by default; use order-45 instrument and TDI interpolation; expose explicit native noise selection. Legacy science-only mapping requires `include_local_metrology=False`.
- [Orbit adapter](../src/egb_multi/external.py): explicit ICRS/ecliptic transformation; default output now matches the ecliptic source contract.
- [JAX geometry](../src/egb_multi/eccentric_jax.py): optional native retarded emitter geometry with state/provider consistency checks and explicit coordinate frame.
- [Native circular reference adapter](../src/egb_multi/reference.py): transform both sky position and polarization into ICRS. Tensor-level tests cover face-on, edge-on and general inclination, including the ecliptic pole.
- [Revision driver](../benchmarks/referee_task_2_objective_orientation.py): defaults to 5 s/order 45, native retarded geometry and ecliptic source coordinates. Its 262144 samples and 65536-sample Welch segments retain the original physical duration and PSD resolution.
- [Archival audit](../benchmarks/referee_task_2_audit.py): explicitly retains historical ICRS positions so old reported scores remain reproducible; it is not the corrected production response.
- [Noise regression](../tests/test_noise.py): default laser-only cancellation in a static unequal-arm interior band must suppress the 95th-percentile residual PSD/link PSD ratio below 1e-15.

The final complete test suite passed **117 tests** in 7.34 s; the [verification log](../outputs/referee/laser_audit_full_tests_verified.log) is saved. Six corrected fixed-intrinsic orientation checks also passed: maximum physical-basis error 1.233e-12, score discrepancy 1.222e-15, and circular fitted residual 1.699e-9. The [corrected results](../outputs/referee/task_2_3_2_4_20261007_corrected_frames_secondary/validation_summary.json) retain their source snapshot. The intermediate `corrected_secondary` run precedes the frame fix and is preserved as an intermediate result.

A matched-duration/resolution [corrected all-default PSD run](../outputs/referee/task_2_3_2_4_20261007_corrected_full_psd/psd_diagnostics.json), with the same six Welch segments as the original audit, has median coherence 0.13951 versus the original 0.99792. Thus the disappearance of near-unity coherence is not explained by increasing the number of Welch averages alone.

Confirmation runs archive source snapshots, hashes, versions and configurations. Existing scientific outputs and manuscript figures were preserved. The final driver conjugates SciPy's `csd(A,E)` to store the covariance convention S_AE=E[A E*] explicitly. Earlier archived driver CSVs store SciPy's conjugate convention; their coherence magnitudes are unchanged. Laser-audit CSVs label their SciPy convention explicitly as `CSD_AconjE`.

## Limits and next decisions

The helper still uses physics upsampling 1 and no antialias filter: it generates noise on a synthetic discrete grid. These tests isolate numerical laser cancellation, not realistic onboard decimation/filtering. Native TCB/LTT orbit mode identifies proper time with TCB. All-default clock imperfections generate SCET sampling errors; a direct TCB GW response is not a full SCET-aware or clock-calibrated signal/noise likelihood. Sideband calibration and a realistic filtering/time-coordinate model remain separate work if a complete mission-noise reference is selected.

The user adopted `secondary_plus_laser` with ideal clocks/ranges for Task 2 on 7 October 2026; the [change note](referee_noise_reference_change_note.md) records the assumptions and manuscript changes. This is an explicitly idealized instrumental-noise reference; the all-default output is raw carrier noise. Fixing laser interpolation does not make their physical meanings interchangeable. Final reference integration remains in 2.3, the coordinate correction requires production reruns in 2.9, and 2.8 must certify waveform/harmonic cadence, optimizer and PSD convergence for the actual science grid.

Reproduce from this checkout, with fresh output paths:

```bash
python benchmarks/referee_laser_cancellation.py --stage initial --output outputs/referee/repeat_laser_initial
python benchmarks/referee_laser_cancellation.py --stage refine --output outputs/referee/repeat_laser_refine
python benchmarks/referee_laser_cancellation.py --stage confirm --output outputs/referee/repeat_laser_confirm
python benchmarks/referee_task_2_objective_orientation.py --noise-model secondary_only --output outputs/referee/repeat_corrected_orientation
python -m pytest -q
```

Literature/source references: [laser residuals from interpolation and onboard processing](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.043040), [PyTDI conventions](https://lisa.pages.in2p3.fr/LDPG/wg6_inrep/pytdi/lisa-conventions.html), [frequency-delay operators](https://lisa.pages.in2p3.fr/LDPG/wg6_inrep/pytdi/content/tdi-combination.html). Exact installed source versions are recorded in the confirmation manifest; no upstream packages were edited.
