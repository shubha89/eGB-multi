# Continuous circular frequency and chirp fitting validation

**Task 2.5 completed at its bounded scope on 7 October 2026.** Continuous frequency/chirp fitting recovers two deliberately offset circular injections and gives stable maxima for two eccentric examples. Sky position and component masses are fixed. These results validate this stage of the optimizer; the full circular-family fitting factor requires the broader search and production convergence in Tasks 2.6–2.8.

All waveform code and runs use this eGB checkout. The [summary](../outputs/referee/task_2_5_20261007_summary/summary.json) verifies matching current-source and archived-snapshot hashes for 26 files in each of the four runs, including the fixed PSD and its metadata.

## Objective and search coordinates

Each search maximizes the real, normalized, coherent A/E inner product over circular frequency, frequency derivative and physical orientation. One positive amplitude is fitted analytically; inclination, polarization and orbital phase share the same physical coefficients across both channels. The exact six-response circular 1PN basis retains its phase-independent radial terms and full noise-weighted Gram matrix.

The frequency reference epoch is the midpoint of the retained detector timestamps. With $T=N_{\rm retained}\Delta t$, the dimensionless search coordinates are

$$
\alpha=(f_{\rm ref}-f_{\rm center})T,\qquad
\gamma=(\dot f-\dot f_{\rm center})T^2.
$$

The source epoch transformation rebases frequency and phase without shifting the detector timestamps. A fixed PSD, analysis band, timestamps and injection normalization apply to every candidate. Injection and template responses contain only the deterministic GW signal; the adopted laser/test-mass/OMS simulation supplies the weighting PSD.

The [search implementation](../src/egb_multi/circular_search.py) evaluates a 7 by 5 coarse grid, retains the original restricted candidates, and runs bounded Powell optimization from the best retained candidate and two leading grid points, removing duplicate starts. The physical orientation profile uses analytic gradients, multiple starts and a four-coefficient SVD seed. The SVD supplies a starting orientation only; the full six-response physical constraints remain in the objective. Independent global orientation fits check the final maximum, the restricted best candidate and an off-maximum candidate in every native case.

## Fixed validation settings

The four native cases use 16384 samples at 5 s cadence, generation-2 TDI, measurement interpolation order 45 and delay order 5. Cropping 198 samples from each end leaves 15988 samples and $T=79940$ s. Sky coordinates are $\beta=0.2$, $\lambda=-1$ rad; masses are $0.8$ and $0.6\,M_\odot$.

Each search covers $\alpha\in[-3.5,3.5]$ and $\gamma\in[-4,4]$, with zero central chirp. Nominal carrier frequencies are 1 or 3 mHz. The fixed bands are respectively 0.2–10 and 0.6–30 mHz. The complete search rectangle is checked for positive carrier frequency inside its band over all retarded source evaluation times. Invalid trials fail explicitly before waveform generation. These bounds cover one local carrier basin.

The reference is the finite diagonal PSD from the [2.3 integration run](../outputs/referee/task_2_3_reference_integrated_20261007/psd_metadata.json), using `secondary_plus_laser` with ideal clocks and ranging. Full-covariance search recovery is tested independently with analytic injections; the native runs reported here use diagonal weighting. Production PSD/covariance and bandwidth convergence remain Task 2.8.

The circular injections have hidden offsets $(\alpha,\gamma)=(1.371,1.213)$ and $(-0.917,-1.117)$, including both chirp signs and an edge-on case. Their imposed chirps are deliberately resolvable during this short observation. They test optimizer recovery rather than an astrophysical chirp distribution. The eccentric examples use $e=0.1$ with fixed evolution and $e=0.3$ with Peters–Mathews evolution.

## Results

The restricted comparison fixes frequency at its nominal value and uses the original nine chirp factors $(0,0.25,0.5,1,2,4,8,16,32)$ times the circular GR chirp. Both restricted and continuous searches use the same new coherent physical profile, PSD and analysis band. These are controlled comparisons of search restrictions, not reproductions of the submitted manuscript's archived scores.

| Injection | Restricted maximum | Continuous maximum | Unique frequency/chirp evaluations through refinement |
|---|---:|---:|---:|
| Circular with positive offsets | 0.325373858 | 1 within roundoff | 228 |
| Circular with negative offsets | 0.282517706 | 1 within roundoff | 313 |
| Eccentric with fixed evolution | 0.931775494 | 0.931778555 | 211 |
| Eccentric with Peters–Mathews evolution | 0.600701780 | 0.600717472 | 231 |

Each intrinsic evaluation generates six physical basis responses. Additional off-maximum and neighboring-point checks occur after the saved refinement count. Both circular recoveries have $|\Delta\alpha|<6.3\times10^{-9}$ and $|\Delta\gamma|<1.6\times10^{-7}$; fitted relative residuals are below $4.9\times10^{-8}$. The eccentric improvements are small, $3.06\times10^{-6}$ and $1.57\times10^{-5}$, conditional on this observation, band and weighting.

![Raw coarse-grid overlaps and continuous maxima](../outputs/referee/task_2_5_20261007_summary/frequency_chirp_validation.png)

The figure shows sampled coarse-grid values and the continuous maximum in each case, without interpolated contours. A [PDF copy](../outputs/referee/task_2_5_20261007_summary/frequency_chirp_validation.pdf) is available.

## Numerical verification

Templates cache native PyTDI delay operators in the [GW-only factorized evaluator](../src/egb_multi/tdi.py). Injections and explicit recovered templates use the original native factorized response path. Their largest weighted relative difference is $1.04\times10^{-15}$. Instrument-noise projection retains the full SCI/TMI/REF path.

Tightening the Powell tolerances from $(2\times10^{-5},10^{-10})$ to $(2\times10^{-6},10^{-12})$ and adding a local 3 by 3 grid changes the best overlap by at most $2.44\times10^{-12}$. Independent starting-point maxima differ by at most $1.66\times10^{-12}$. Explicit native responses and independent global orientation fits agree with the fast profile within $4.5\times10^{-16}$ at the maxima; off-maximum comparisons agree within $1.2\times10^{-16}$. No maximum reaches a declared search boundary, and neighboring-point checks find no improvement above tolerance.

All predeclared gates in the run manifests pass, including circular recovery, candidate retention, response agreement and refinement stability. The [full test log](../outputs/referee/task_2_5_full_tests.log) records **145 passing tests** and one upstream Astropy compatibility warning. Tests include analytic frequency/chirp recovery with diagonal and full complex-covariance weighting, epoch rebasing, invalid-domain rejection, physical orientation recovery, and native factorized TDI equivalence.

## Reproduction and next task

From the checkout root, the [validation driver](../benchmarks/referee_task_2_frequency_chirp.py) runs all four cases by default and requires a fresh output directory:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/miniconda3/bin/python \
  benchmarks/referee_task_2_frequency_chirp.py \
  --output outputs/referee/task_2_5_reproduction --samples 16384 --dt 5
```

The archived evidence uses four separate processes, selected with `--cases`. Each preserves its manifest, source/PSD snapshots, evaluation history and validation record. The [summary driver](../benchmarks/referee_task_2_5_summary.py) verifies these four runs and produces the table and plots.

The subsequent [2.6 validation](referee_task_2_6_sky_mass_harmonics.md) extends the search to sky position, both masses and finite carrier/harmonic basins, with bound expansion and multiple starts. Its controlled baselines are rerun at its own sampling settings; these archived 2.5 values remain separate. No mass-fixed reduction is certified. Next, Tasks 2.7–2.8 must certify the complete optimizer and production error budget before the science grid, fitting-factor claims and referee response are revised. See the [Task 2 tracker](referee_task_2_subtasks.md).
