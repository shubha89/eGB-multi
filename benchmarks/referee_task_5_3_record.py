"""Preserve the qualitative Task 5.3 assessment and verify its audit inputs."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/referee'
OUT=BASE/'task_5_3_20261012'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    prior_path=BASE/'task_5_1_5_2_20261012/manifest.json'
    prior=json.loads(prior_path.read_text())
    for relative,expected in prior['sha256'].items():
        assert digest(ROOT/relative)==expected,relative
    for relative,expected in prior['inspected_source_hashes'].items():
        assert digest(ROOT/relative)==expected,relative
    report=ROOT/'reports/referee_task_5_3_applicability.md'
    tracker=ROOT/'reports/referee_task_5_subtasks.md'
    revision=ROOT/'reports/referee_revision_table.md'
    for path in (report,tracker,revision):
        content=path.read_text()
        for raw in re.findall(r'\]\(([^)]+)\)',content):
            if raw.startswith(('http://','https://')):
                continue
            target=re.sub(r':\d+$','',raw.strip('<>'))
            local=Path(target) if target.startswith('/') else path.parent/target
            assert local.exists(),(str(path),target)
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(report,OUT/'applicability_assessment.md')
    shutil.copy2(ROOT/'reports/referee_task_5_1_5_2_audit.md',OUT/'input_source_and_literature_audit.md')
    shutil.copy2(tracker,OUT/'subtask_tracker_at_completion.md')
    state=dict(status='qualitative_applicability_assessment_completed',
        reviewed_utc=datetime.now(timezone.utc).isoformat(),task_5_3_complete=True,
        entire_task_5_complete=False,next_task='Other research tasks; 5.4–5.7 deferred to final manuscript stage',
        manuscript_workflow='User requested manuscript checks at the end on 12 October; research/run validation continues',
        completed_scope='Code-to-literature applicability matrices and interpretation conditions',
        numerical_omitted_effect_bound_established=False,
        universal_safe_frequency_assigned=False,stellar_inputs_assigned=False,
        new_source_physics_implemented=False,manuscript_edits_applied=False,
        source_classes=['synthetic_point_mass','detached_WD_containing','interacting_AM_CVn',
                        'tertiary_companion','other_compact_binaries'],
        effects_assessed=['stellar_apsidal_motion','tidal_angular_momentum_exchange',
                          'mass_transfer','source_CoM_motion'],
        interpretation_conditions=['preserve_modular_framework_scope',
            'distinguish_point_mass_examples_from_stellar_population_predictions',
            'physical_adequacy_is_separate_from_numerical_FF_convergence',
            'phenomenological_fdot_is_not_coupled_stellar_or_environmental_dynamics',
            'use_observation_duration_and_correct_frequency_conventions',
            'parameter_absorption_does_not_certify_unbiased_astrophysical_inference'],
        input_audit_provenance_hash_errors=0,
        unchanged_source_hashes=prior['inspected_source_hashes'])
    (OUT/'work_state.json').write_text(json.dumps(state,indent=2)+'\n')
    inputs=[prior_path,BASE/'task_5_1_5_2_20261012/source_audit.json',
            BASE/'task_5_1_5_2_20261012/literature_evidence.json',
            BASE/'task_2_8_20261012_completion/frozen_production_configuration.json',
            report,tracker,revision,Path(__file__).resolve()]
    outputs=[p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']
    manifest=dict(created_utc=state['reviewed_utc'],
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        output_sha256={str(p.relative_to(ROOT)):digest(p) for p in outputs},
        interpretation='Saved scientific scope assessment; not a numerical validity certificate')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(task_5_3_complete=True,entire_task_5_complete=False,
        prior_audit_hash_errors=0,manifest=str((OUT/'manifest.json').relative_to(ROOT))),indent=2))


if __name__=='__main__':
    run()
