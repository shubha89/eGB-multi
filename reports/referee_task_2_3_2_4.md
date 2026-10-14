# Task 2 A E objective and circular orientation validation

Date: 7 October 2026. Manuscript: DV14615.

**Follow-up audits and adopted reference:** inadequate instrument/TDI interpolation was fixed, along with ICRS/ecliptic and retarded-emitter response inconsistencies; see the [laser/convention audit](referee_laser_noise_and_conventions_audit.md). The subsequent [paired laser-on/off test](referee_laser_on_off_validation.md) passes with ideal clocks/ranges but finds a separate material residual from raw default ranging noise. The adopted `secondary_plus_laser` model is recorded in the [change note](referee_noise_reference_change_note.md). The [final 2.3 integration](referee_task_2_3_reference_integration.md) now passes with deterministic clean responses and six laser-enabled orientation cases. The numerical examples and original settings below are preserved as the earlier validation record.

Task 2.4 is implemented and validated at fixed frequency, frequency derivative, sky and masses. Task 2.3's implementation/reference integration passes at its stated scope. The earlier all-default simulation developed strong laser-dependent A/E correlations; corrected interpolation suppresses that broadband effect, while the paired follow-up finds localized ranging-dependent residuals. Production PSD/covariance convergence remains Task 2.8; the new finite-covariance diagnostic materially shifts eccentric scores and must be resolved before production weighting is certified.

All waveform runs import this checkout's `src/egb_multi`. Scripts, tests, source snapshots, configurations, noise realizations, PSDs and validation results are saved under this project. The two [validation directories](../outputs/referee/) are `task_2_3_2_4_20261007_instrument_defaults` and `task_2_3_2_4_20261007_secondary_only`.

## Real A E comparison objective

The implementation in [comparison.py](../src/egb_multi/comparison.py) uses real, uniformly sampled A/E series and a one-sided density PSD. It defines

$$
\tilde a_c(f_k)=\Delta t\sum_{j=0}^{N-1}a_c(t_j)e^{-2\pi i k j/N},\qquad
\Delta f=(N\Delta t)^{-1},
$$

$$
(a|b)_{AE}=4\Delta f\operatorname{Re}\sum_{c\in\{A,E\}}\sum_{k\in B}
\frac{\tilde a_c(f_k)\tilde b_c^*(f_k)}{S_c(f_k)},\qquad
\mathcal O=\frac{(d|h)}{\sqrt{(d|d)(h|h)}}.
$$

The PSD, physical band, injection norm and detector timestamps stay fixed for every candidate. The band excludes DC and lies strictly below Nyquist, so neither endpoint requires a special half weight. Rectangular observation windows are used; there is no taper or implicit time shift. A/E weighting is diagonal, as in the existing analysis; covariance adequacy remains part of the noise-model assessment below.

The API rejects incompatible units or TDI conventions, shifted timestamps, nonfinite samples, nonpositive norms and invalid PSD bins. It refuses PSD extrapolation and refuses to truncate an analysis band at Nyquist. PSDs are linearly interpolated in frequency only within their recorded support.

For a fixed physical template, the best common positive amplitude is

$$
\widehat A=\max\left[0,\frac{(d|h)}{(h|h)}\right].
$$

The outputs distinguish signed overlap, raw relative residual and the relative residual after this amplitude fit. Residuals are computed from the waveform difference rather than by subtracting nearly equal norms. Negative overlap is not silently turned into positive overlap; physical phase/polarization variation belongs to the orientation fitter.

## Signal and PSD conventions

The validation uses second-generation Michelson TDI, measurement interpolation order 7 and delay interpolation order 5 for both signal and noise projection. A/E follows the package convention $A=(Z-X)/\sqrt2$, $E=(X-2Y+Z)/\sqrt6$. `TDIConvention` records generation, interpolation, orbit, delays, units and A/E normalization, and comparisons require equality of these records.

The generated noise uses SCI, TMI and REF carrier fluctuations. The previous noise adapter populated SCI only and set the local interferometers to zero. The revised adapter adds an explicit `include_local_metrology=True` option; the new driver always uses it. Historical science-only behavior remains the default for compatibility. GW-only link data still have zero TMI/REF content, which is appropriate for the source contribution. PyTDI's xi/eta construction uses the local interferometers, as verified in the [installed primary implementation](/Users/shubhanshutiwari/.local/lib/python3.13/site-packages/pytdi/intervar.py:72).

Instrument fluctuations are in Hz, whereas eGB GW links are dimensionless fractional-frequency response. The PSD conversion is explicitly

$$
S^{\rm fractional}_{c}(f)=S^{\rm Hz}_{c}(f)/\nu_0^2,
\qquad \nu_0=2.816\times10^{14}\ \mathrm{Hz}.
$$

The actual central frequency is obtained from the instantiated instrument and saved. The PSD stores both units. Tests verify that converting both signal and PSD preserves the absolute noise-weighted norm. PyTDI's `unit="frequency"` selects the frequency form of its delay operations; it does not supply this optical-frequency conversion. See the [PyTDI input interface](https://lisa.pages.in2p3.fr/LDPG/wg6_inrep/pytdi/content/interface.html) and [factorized Michelson documentation](https://lisa.pages.in2p3.fr/LDPG/wg6_inrep/pytdi/content/standard-combinations.html).

For these tests, the orbit file uses the local analytic trailing constellation and instantaneous geometric arm lengths, matching the existing eGB response approximation. It is a generated model input, not a mission ephemeris. Orbit samples are spaced by 600 s with a margin on either side. The signal and noise TDI evaluation use the same recorded instrument MPRS delay arrays through the new optional `link_delays` argument. This establishes shared analysis delays; it does not validate the underlying instantaneous-arm approximation against retarded mission geometry.

## Physical orientation basis

[circular_fit.py](../src/egb_multi/circular_fit.py) generates six circular eGB→links→TDI responses at edge-on inclination, two polarization angles and three orbital phases. Linear combinations extract cosine, sine and phase-independent responses for each of the two polarization tensors. The basis retains the small circular 1PN chirp term identified in Task 2.2.

Let $P,Q$ be the unrotated polarization tensors, $u=\cos\iota$, $\theta=2\delta\phi$, and $\gamma=2\psi$. With temporal responses proportional to $-Ka(t)\cos\Phi$, $-Ka(t)\sin\Phi$, and $-Kc_r(t)$, the physical coefficients are

$$
\begin{aligned}
v_{P\cos}&=(1+u^2)\cos\gamma\cos\theta+2u\sin\gamma\sin\theta,\\
v_{P\sin}&=-(1+u^2)\cos\gamma\sin\theta+2u\sin\gamma\cos\theta,\\
v_{Q\cos}&=-(1+u^2)\sin\gamma\cos\theta+2u\cos\gamma\sin\theta,\\
v_{Q\sin}&=(1+u^2)\sin\gamma\sin\theta+2u\cos\gamma\cos\theta,\\
v_{Pr}&=(1-u^2)\cos\gamma,\qquad
v_{Qr}=-(1-u^2)\sin\gamma.
\end{aligned}
$$

One common amplitude multiplies this entire coefficient vector in A and E. The implementation never fits six arbitrary amplitudes or fits channels independently. The full real Gram matrix retains unequal norms, nonorthogonal basis vectors and the phase-independent responses.

The overlap is evaluated as $v^Tp/\sqrt{(d|d)\,v^TGv}$, with $G_{ij}=(b_i|b_j)$ and $p_i=(d|b_i)$. A full-domain grid, two deterministic differential-evolution seeds and local refinements search $u\in[-1,1]$, $\psi\in[0,\pi)$ and orbital phase shift $\delta\phi\in[0,\pi)$. Common positive amplitude is profiled analytically afterwards. Each result records its seeds, evaluation count, Gram matrix, angles, scale and residual.

This is a physical extrinsic maximum at fixed intrinsic parameters. It is not the fitting factor. The validation sources use $t_0=0$; conversion to the midpoint coordinates required for the frequency/chirp search follows the verified Task 2.2 epoch identity. Face-on phase/polarization degeneracies mean waveform recovery, rather than unique angular recovery, is the relevant acceptance test.

## Validation results

The [run script](../benchmarks/referee_task_2_objective_orientation.py) was executed for both noise candidates. Each run uses 131072 samples at 10 s cadence, 15.17 days before cropping. A conservative 39-sample halo at each endpoint excludes TDI evaluations lacking full delay/interpolation support, leaving 130994 samples and 1309940 s. The rule is saved in PSD metadata. Welch segments contain 32768 samples with Hann windows, constant detrending and 50 percent overlap, giving $\Delta f_{\rm PSD}=3.05176\times10^{-6}$ Hz and six overlapping segments.

Both runs verify four circular injections spanning 0.1–3 mHz, face-on and edge-on limits, general orientations, positive and negative prescribed chirps, and a common amplitude change of 2.3. Two eccentric cases exercise fixed and Peters–Mathews evolution. The analysis band is $[0.2f_{\rm inj},10f_{\rm inj}]$, up to 30 mHz for the 3 mHz examples.

| Check | All-default noise candidate | Test-mass and optical-metrology candidate |
|---|---|---|
| Largest basis-versus-explicit weighted relative error | $1.2181\times10^{-12}$ | $1.7224\times10^{-12}$ |
| Largest basis-versus-explicit overlap difference | $3.1086\times10^{-15}$ | $1.4433\times10^{-15}$ |
| Largest circular residual after orientation/amplitude fitting | $8.8781\times10^{-10}$ | $7.4292\times10^{-10}$ |
| Largest overlap spread between two DE seeds | $1.1805\times10^{-11}$ | $3.0491\times10^{-12}$ |
| Validation cases passing the recorded checks | 6 of 6 | 6 of 6 |

For one general chirping circular injection, independently generated waveforms at six nearby physical orientations do not improve the recovered solution within the $10^{-9}$ overlap tolerance. Independent Euclidean tests additionally recover orientations with nonorthogonal basis vectors and an intentionally large phase-independent term, testing the physical constraint outside its weak astrophysical limit.

All **44 targeted tests passed** in 2.29 s. Tests cover independent direct-DFT and Parseval normalization, sign and amplitude treatment, unit conversion, invalid inputs, coherent physical orientation recovery, local-metrology mapping, shared delay arrays and the earlier Task 2.1/2.2 checks. The [validation record](../outputs/referee/task_2_3_2_4_validation_record.json) contains the command, current source hashes and the verification of 66 archived snapshot hashes. Numerical production convergence remains Task 2.8.

## Noise model decision

The all-default candidate retains the manuscript's enabled noise amplitudes, but uses the existing fast noise helper's sampling choices: physics upsampling 1, no antialias filter, Lagrange order 3 in the instrument simulation, and no low-frequency noise cutoff. These are distinct from the order-7 TDI analysis interpolation. The source snapshot and `instrument_defaults.json` preserve these settings and defaults. Sideband clock-noise calibration is not applied. This candidate must not be presented as a calibrated mission sensitivity.

The alternative keeps default test-mass and all optical-metrology noise levels and disables laser, clock, modulation, backlink, ranging, jitter, DWS and MOC time-correlation noise. It represents an idealized secondary-noise reference, not successful calibration of all those contributions. Neither candidate includes a Galactic confusion foreground.

The [PSD diagnostic script](../benchmarks/referee_psd_diagnostics.py) measures cross-channel coherence and tests linear versus constant Welch detrending without changing the saved PSD. The [PSD and coherence comparison plot](../outputs/referee/task_2_3_2_4_noise_comparison.png) shows all three noise configurations, with raw unsmoothed estimates and labelled units.

| Noise configuration | Median A/E magnitude-squared coherence over 0.02–30 mHz |
|---|---|
| All default noise amplitudes | 0.997924 |
| Test-mass and optical-metrology only | 0.134899 |
| All defaults with laser noise alone disabled, diagnostic only | 0.186502 |

Six overlapping Welch segments give a finite-estimator coherence scale of roughly $1/6$ even for independent channels; these medians do not establish exact population covariance. They nevertheless identify near-unity broadband correlation in the all-default candidate. Disabling laser noise removes that broad feature, indicating laser-dependent contamination. Linear detrending does not remove it. Its precise origin—such as numerical delay/interpolation residuals, sampling or instrument/TDI conventions—has not yet been isolated.

The noise choice materially changes the fixed-intrinsic eccentric comparisons:

| Injection | All-default candidate extrinsic maximum | Secondary-noise candidate extrinsic maximum |
|---|---|---|
| 1 mHz, $e=0.1$, fixed evolution | 0.914479 | 0.843778 |
| 3 mHz, $e=0.3$, Peters–Mathews evolution | 0.922386 | 0.545151 |

These are short validation examples at fixed circular frequency/derivative/sky/masses, with the derivative set to zero for these two templates. They are not revised manuscript fitting factors or thresholds.

The reference choice is settled: laser/test-mass/optical-metrology noise with ideal clocks/ranges. The [change note](referee_noise_reference_change_note.md) records the explicit revised description and the remaining integration/production checks. A complete mission instrument model would additionally require clock/ranging calibration, filtering and consistent time-coordinate treatment. The original manuscript curves and cached PSD remain archived as the historical baseline.

## Reproduction and remaining work

Use `/opt/miniconda3/bin/python` from this project root. Choose fresh output folders:

```bash
python benchmarks/referee_task_2_objective_orientation.py --noise-model instrument_defaults --output outputs/referee/repeat_full_noise
python benchmarks/referee_task_2_objective_orientation.py --noise-model secondary_only --output outputs/referee/repeat_secondary_noise
python benchmarks/referee_task_2_objective_orientation.py --noise-model instrument_defaults --psd-only --disable-laser-diagnostic --output outputs/referee/repeat_laser_disabled
python -m pytest tests/test_comparison.py tests/test_circular_fit.py tests/test_noise.py tests/test_tdi.py tests/test_referee_task_2_audit.py -q
```

Each output directory contains a hashed source snapshot, configuration, package versions, both PSD units, A/E cross-spectrum, the noise realization and shared delays. Existing directories are protected from overwrite; imports and output paths are restricted to this checkout.

The adopted-reference integration is recorded in the [2.3 follow-up](referee_task_2_3_reference_integration.md). The subsequent [2.5 validation](referee_task_2_5_frequency_chirp.md) passes bounded continuous frequency/chirp fitting at fixed sky and masses. Task 2.6 adds sky, mass and harmonic searches; 2.7 checks the complete optimizer; 2.8 certifies production settings, including PSD/covariance accuracy; 2.9–2.10 rerun results and revise the manuscript/reply.
