"""Save the 12 October scoped scientific review and its immutable evidence.

This records the explicitly reviewed reference and empirical acceptance criteria.
It does not certify the full mission likelihood or an exact global optimum.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/referee'
OUT = BASE / 'task_2_8_20261012_completion'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run():
    review = read(BASE / 'task_2_8_20261009_closure_review/closure_review.json')
    inventory = read(BASE / 'task_2_8_20261008_summary/work_state.json')
    assert not review['pending_runs'] and not review['failed_runs']
    assert not any(r['hash_errors'] for r in inventory['provenance_audit'])
    assert inventory['tests']['passed'] == 205
    final_path = BASE / 'task_2_8_20261012_high_pm_final_metric_independent/annual_refinement.json'
    final = read(final_path)[0]
    assert final['independent_starts_only'] and final['retained_candidate_count'] == 0
    assert final['all_six_intrinsic_coordinates_free']
    assert final['independent_reference']['same_physical_metric_inputs_checked']
    assert len(final['local_runs']) == 6 and all(r['success'] for r in final['local_runs'])
    assert abs(final['independent_baseline_deficit']) < .0001
    assert abs(final['native_minus_spectral']) < 1e-10
    assert final['frequency_chirp_boundary_distance_scaled'] > 1
    assert sum(abs(r['overlap'] - final['search']['best']['profile']['overlap']) < .0001
               for r in final['local_runs']) >= 2
    band_path = BASE / 'task_2_8_20261012_high_pm_band_cadence/annual_refinement.json'
    bands = {r['level']['name']: r for r in read(band_path)}
    reference = final['independent_reference']['native_overlap']
    old = bands['same_metric_original_band_2_5']['native_winner_metrics']['overlap']
    wide = bands['extended_100_mHz_2_5']['native_winner_metrics']['overlap']
    fine = bands['extended_100_mHz_1_25']['native_winner_metrics']['overlap']
    assert all(r['all_six_intrinsic_coordinates_free'] and
               all(s['success'] for s in r['local_runs']) and
               r['frequency_chirp_boundary_distance_scaled'] > 1 for r in bands.values())
    assert abs(reference - old) < .0002 and abs(wide - reference) < .0002
    assert abs(fine - wide) < .0002
    prepared = BASE / 'task_2_8_20261012_followup_plan/prepared_production_configuration.json'
    config = read(prepared)
    assert abs(sum(config['working_allocations'].values()) - .001) < 1e-15
    for field in ('covariance_file', 'laser_covariance_file'):
        path = ROOT / config['reference'][field]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config['reference'][field + '_sha256']
    scope = config['scope']
    now = datetime.now(timezone.utc).isoformat()
    config.update(status='frozen_at_representative_annual_reference_scope',
                  frozen_utc=now, task_2_8_complete=True,
                  production_configuration_frozen=True, production_grid_completed=False,
                  prepared_configuration=str(prepared.relative_to(ROOT)))
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = OUT / 'frozen_production_configuration.json'
    save(frozen, config)
    save(OUT / 'reviewed_evidence_inventory.json', inventory)
    save(OUT / 'reviewed_closure_evidence.json', review)
    evidence = dict(review['completed_summary_hashes'])
    paths = [prepared, frozen, final_path, band_path,
             OUT / 'reviewed_evidence_inventory.json', OUT / 'reviewed_closure_evidence.json',
             ROOT / inventory['tests']['log'], Path(__file__).resolve()]
    for folder in ('task_2_8_20261012_high_pm_final_metric_independent',
                   'task_2_8_20261012_high_pm_band_cadence'):
        manifest_path = BASE / folder / 'manifest.json'
        paths.append(manifest_path)
        manifest = read(manifest_path)
        for relative, expected in manifest['sha256'].items():
            path = ROOT / relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
            evidence[relative] = expected
        evidence.update(manifest['inputs'])
    for field in ('covariance_file', 'laser_covariance_file'):
        evidence[config['reference'][field]] = config['reference'][field + '_sha256']
    for path in paths:
        evidence[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    gates = {
        '2.8a_noise_weighting': 'Complete for adopted stationary reference: full Hermitian moving-secondary plus primary covariance; nested-grid all-bin overlap bound 5.0404e-5; sampled filter/epoch/geometry and finite-covariance controls reviewed. No complete mission-noise certification.',
        '2.8b_laser_precision': 'Complete at reference scope: primary residual included; first locking arithmetic loss localized. Full sampled-instrument locking repair remains outside this closure.',
        '2.8c_annual_baselines': 'Complete: circular, small-e, near-contour, high-e, repaired high-PM, low-frequency/asymmetric and mid-e cases; historical missed branch preserved.',
        '2.8d_numerical_reoptimization': 'Complete: selected cadence/interpolation/envelope controls and nine repaired high-PM refinements; broad-band 5-to-2.5 s shift 4.2117e-5; optional 2.5-to-1.25 s shift 1.0829e-5.',
        '2.8e_band_leakage': 'Complete: eight combined-reference cases; final 2.5 s 40-to-80 mHz shift 1.70285e-4; 80-to-100 mHz shift 6.47145e-10. Earlier accepted 5 s exception retained as history.',
        '2.8f_evolution_optimizer': 'Complete empirically: PM tightening shifts below 1.7e-11; twelve repaired basins and final six blind free-coordinate starts; native agreement 5.4111e-9, no active frequency/chirp boundaries. Finite searches are not a global-maximum theorem.',
        '2.8g_contours_frozen_settings': 'Complete: reproducible configuration frozen; representative crossing bracket [0.089,0.089125] preserved under final reference. Working FF target 0.001 implies eccentricity uncertainty approximately 0.000482 here; revised production contours remain Task 2.9.'}
    decision = dict(status='completed_at_representative_annual_reference_scope',
        reviewed_utc=now, review_author='Codex scientific evidence review', scope=scope,
        task_2_8_complete=True, production_configuration_frozen=True,
        production_grid_completed=False, comment_2_response_completed=False,
        frozen_configuration=str(frozen.relative_to(ROOT)), gate_status=gates,
        working_absolute_FF_target=.001, working_allocations=config['working_allocations'],
        uncertainty_interpretation='Conservative working allocations supported by the declared representative empirical controls. Only specified all-bin metric comparisons have uniform mathematical bounds; no uniform physical-model or global-search error theorem is asserted.',
        final_independent_native_overlap=final['native_winner_metrics']['overlap'],
        independent_native_difference=-final['independent_baseline_deficit'],
        native_minus_spectral=final['native_minus_spectral'],
        blind_starts=6, retained_candidates=0,
        frequency_chirp_boundary_distance_scaled=final['frequency_chirp_boundary_distance_scaled'],
        optional_controls=dict(same_cadence_original_to_80_mHz_shift=reference-old,
            upper_80_to_100_mHz_shift=wide-reference, cadence_2_5_to_1_25_s_shift=fine-wide,
            cadence_refinement_also_doubles_envelope_intervals=True),
        representative_crossing=dict(bracket=[.089,.089125],
            final_reference_endpoint_overlaps=[.9000878015644455,.8998282949490907],
            local_slope=-2.0760546932514483,
            working_eccentricity_uncertainty=.001/2.0760546932514483,
            revised_manuscript_threshold=False),
        previous_band_exception=review['accepted_band_exception'],
        software_tests=inventory['tests'], provenance_hash_errors=0,
        completed_run_summaries=sum(r['summary'] is not None for r in inventory['runs'].values()),
        limitations=config['limitations'], next_tasks=['2.9 production grid and figures',
            '2.10 manuscript changes and referee response'], evidence_sha256=evidence)
    save(OUT / 'completion_decision.json', decision)
    print(json.dumps({k:decision[k] for k in ('status','task_2_8_complete',
        'production_configuration_frozen','final_independent_native_overlap','optional_controls')}, indent=2))


if __name__ == '__main__':
    run()
