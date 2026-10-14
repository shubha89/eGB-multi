"""Write the Task 2.6 report from verified numerical records."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'circular_unknown_sky_mass':'Circular with unknown sky and masses',
    'eccentric_fixed':'Eccentric with fixed evolution, e = 0.1',
    'eccentric_pm':'Eccentric with Peters–Mathews evolution, e = 0.3',
    'eccentric_high_e':'Eccentric with fixed evolution, e = 0.6'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(summary_folder,report):
    folder,report = summary_folder.resolve(),report.resolve()
    if not folder.is_relative_to(ROOT) or report.parent != ROOT/'reports':
        raise ValueError('Use this checkout and its reports folder')
    source = folder/'summary.json'
    summary = json.loads(source.read_text())
    if summary['status']!='passed_Task_2_6_staged_validation' or set(row['case'] for row in summary['rows'])!=set(NAMES):
        raise ValueError('Four verified passing native cases required')
    if report.exists():
        raise FileExistsError('Existing reports must be reviewed before replacement')
    rows = summary['rows']
    table = '\n'.join(f"| {NAMES[r['case']]} | {r['fixed_sky_mass_overlap']:.9f} | {r['expanded_family_overlap']:.9f} | {r['initial_basins']} / {r['expanded_basins']} | {1000*r['winning_frequency_hz']:.6f} |" for r in rows)
    circular = next(row for row in rows if row['case']=='circular_unknown_sky_mass')
    counts = '\n'.join(f"| {NAMES[r['case']]} | {r['initial_unique_intrinsic_evaluations']} | {r['expanded_unique_intrinsic_evaluations']} | {r['expansion_overlap_change']:.3g} |" for r in rows)
    max_expansion = max(row['expansion_overlap_change'] for row in rows)
    max_refinement = max(row['refinement_overlap_change'] for row in rows)
    max_response = max(row['native_vs_cached_weighted_shape_residual'] for row in rows)
    max_profile = max(abs(row['global_minus_profile_overlap']) for row in rows)
    max_native = max(abs(row['explicit_minus_profile_overlap']) for row in rows)
    max_spread = max(row['winning_basin_independent_start_spread'] for row in rows)
    mass_shape = max(row['mass_shape_max_weighted_relative_residual_at_best'] for row in rows)
    unsuccessful = sum(not value for row in rows for value in row['winning_basin_start_successes'])
    relative_folder = '../'+str(folder.relative_to(ROOT))
    old_failure = ROOT/'outputs/referee/task_2_6_20261007_circular_unknown_sky_mass/sky_mass_harmonic_validation.json'
    old = json.loads(old_failure.read_text())[0]
    old_score = old['final_refinement']['best']['profile']['overlap']
    text = f'''# Circular sky mass and harmonic search validation

**Task 2.6 implemented and validated within the finite search described below.** The exact circular 1PN template family now varies sky position and both component masses, in addition to frequency, chirp and coherent physical orientation. Four native cases pass the declared waveform, candidate-retention, bound-expansion and refinement gates. The full optimizer and production fitting-factor certification remain Tasks 2.7–2.8.

The [verified summary]({relative_folder}/summary.json) records current-source and archived-snapshot checks for {summary['verified_runs'][0]['snapshot_and_current_hashes_checked']} files per run. All eGB imports, scripts and outputs use this checkout. The adopted laser/test-mass/OMS reference with ideal clocks and ranging is unchanged.

## Physical family and numerical method

The [family search](../src/egb_multi/circular_family_search.py) profiles the same six-response physical orientation basis as Tasks 2.4–2.5. Inclination, polarization and orbital phase share coefficients across A and E; one positive common amplitude is fitted analytically. The chirping phase-independent 1PN terms remain in the objective. Each intrinsic candidate varies frequency, frequency derivative, sine of ecliptic latitude, periodic longitude and two mass coordinates.

The sky domain is the full sphere, including both poles. Equal-area Fibonacci points seed local fits. Each local longitude chart covers a full period and evaluation is canonical modulo 2π. Two coordinates on the unit square map onto the entire ordered mass triangle, 0.1 ≤ m2 ≤ m1 ≤ 1.4 solar masses. This is the computational domain in the parameter contract, not a population prior. Masses remain independent of the prescribed circular chirp.

Frequency/chirp coordinates are frequency offset times the retained duration and frequency derivative times duration squared, referred to the observation midpoint. All-sky retarded-time bounds use the norms of the actual sender and receiver positions, including emission light-travel times. The entire domain is checked for positive instantaneous frequency inside the fixed band before waveform generation.

Local SLSQP fits use measured diagonal curvature to scale their numerical coordinates. A curvature floor of 1e-6 limits scaling along flat directions; it changes neither the physical bounds nor the overlap. Physical finite-difference steps remain explicit. The first unscaled native circular search stopped at overlap {old_score:.12f}, despite small subsequent refinement changes, and failed the predeclared recovery gate. That [failed record](../outputs/referee/task_2_6_20261007_circular_unknown_sky_mass/validation_summary.json) remains archived. Curvature scaling resolves the failure. Three other incomplete unscaled runs have interruption records and remain separate from the passing evidence.

## Frequency basins and fixed comparison settings

Frequency seeds come from peaks of the actual noise-weighted A/E response spectrum. A Hann window suppresses leakage for seed selection only; the fit objective remains untapered. Every peak above the relative threshold is retained, and weaker peaks are added to reach the stated seed-spectrum power coverage. The nominal carrier is also retained. This accommodates the package's precessing radial/azimuthal sidebands without imposing exact half-integer frequency ratios.

The initial pass uses basin half-width 4/T, chirp bounds ±4/T², 16 equal-area sky points plus the poles, and two distinct sky starts per basin. Expansion uses half-width 6/T, chirp bounds ±8/T², 32 sky points plus the poles and three starts. Peak thresholds decrease from 1% to 0.5% of the strongest weighted peak, and target tapered-spectrum coverage increases from 99% to 99.9%. All initial basin winners are retained and reevaluated with the same objective. Seed-spectrum coverage is a diagnostic, not an upper bound on fitting-factor error or an operational detection procedure.

The native runs use {circular['samples_before_crop']} samples at {circular['dt_s']:g} s cadence, generation-2 TDI, measurement interpolation order 45 and delay order 5. Cropping 191 samples at either end gives T = {circular['duration_s']:g} s. One finite diagonal PSD from the adopted [2.3 reference](../outputs/referee/task_2_3_reference_integrated_20261007/psd_metadata.json) weights all candidates. Bands are fixed at 0.2–10 mHz for the 1 mHz cases and 0.6–30 mHz for the 3 mHz cases. Injection norm, timestamps, PSD and band never vary with a candidate or harmonic.

## Native results

The fixed-sky/mass baseline is rerun at these exact settings, with continuous frequency/chirp and the same physical orientation profile. It is a controlled restriction comparison; its numbers should not be compared directly with archived runs at different durations, cadences or weighting.

| Injection | Fixed sky and masses | Expanded family | Initial / expanded basins | Winning frequency in mHz |
|---|---:|---:|---:|---:|
{table}

The deliberately chirping circular injection has unknown sky and masses relative to the initial prototype. Its final overlap deficit is {1-circular['expanded_family_overlap']:.3g}, and its sky-direction chord error is {circular['circular_recovery_sky_direction_error']:.3g}. The imposed short-observation chirp tests optimization rather than an astrophysical population. Returned native mass values are not mass measurements: the largest amplitude-fitted endpoint mass-shape residual at the four best candidates is {mass_shape:.3g}. No mass-fixed reduction is certified by these checks.

![Best evaluated circular fits across frequency basins]({relative_folder}/sky_mass_harmonic_validation.png)

The plot compares staged basin winners under the same injection normalization; the black star includes all evaluated candidates and final refinement. It has no interpolated contours. A [PDF copy]({relative_folder}/sky_mass_harmonic_validation.pdf) is available.

| Injection | Initial intrinsic evaluations through refinement | Expanded intrinsic evaluations through refinement | Expansion change in overlap |
|---|---:|---:|---:|
{counts}

Each intrinsic evaluation generates six basis responses. Expanded counts include re-evaluated retained candidates; independent explicit-response and mass-audit calls are additional.

## Verification and remaining certification

The largest change from wider bounds, additional sky starts and increased peak coverage is {max_expansion:.3g}; final refinement changes the overlap by at most {max_refinement:.3g}. The maximum winning-basin starting-point spread is {max_spread:.3g}, with {unsuccessful} unsuccessful local starts in the winning basins. These records remain part of the comprehensive optimizer audit in 2.7; agreement of refinement alone is not proof of the global maximum.

Injections and explicit recovered templates use the original native factorized TDI path. Cached template responses agree in weighted shape within {max_response:.3g}. Explicit native overlaps agree with the profile within {max_native:.3g}; independent global orientation checks agree within {max_profile:.3g}. All predeclared gates in the passing run manifests hold.

The [full test log](../{summary['tests']['log']}) records **{summary['tests']['passed']} passing tests** and one upstream Astropy compatibility warning. Independent analytic responses verify joint recovery of frequency, chirp, sky and both masses with diagonal and full Hermitian covariance weighting. A separate two-line test selects the stronger frequency basin while preserving the full signal normalization. Ordered mass boundaries, longitude periodicity, poles, candidate retention and invalid-domain failures are also tested.

The [native driver](../benchmarks/referee_task_2_sky_mass_harmonics.py) runs all cases by default and requires a fresh local output directory:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/miniconda3/bin/python \\
  benchmarks/referee_task_2_sky_mass_harmonics.py \\
  --output outputs/referee/task_2_6_reproduction --samples 8192 --dt 10
```

The passing archive uses four processes selected with `--cases`. The [summary driver](../benchmarks/referee_task_2_6_summary.py) verifies inputs and creates the plots; the [report driver](../benchmarks/referee_task_2_6_report.py) writes this report from its numerical record.

Task 2.7 now checks the complete optimizer, including additional independent starts, the small-eccentricity limit and search-domain sufficiency. Task 2.8 certifies cadence, bandwidth, source evolution, PSD/covariance and the production error budget. The short native runs and tapered seed coverage do not replace those checks. See the [Task 2 tracker](referee_task_2_subtasks.md).
'''
    report.write_text(text)
    shutil.copy2(Path(__file__),folder/'referee_task_2_6_report.py')
    provenance = {'summary_sha256':digest(source),'report_path':str(report.relative_to(ROOT)),
        'report_sha256':digest(report),'report_script_sha256':digest(Path(__file__)),
        'archived_failed_circle_results_sha256':digest(old_failure)}
    (folder/'report_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--report',type=Path,default=ROOT/'reports/referee_task_2_6_sky_mass_harmonics.md')
    args = parser.parse_args()
    write_report(args.summary,args.report)
