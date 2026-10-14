"""Verify a saved, scoped evidence-review decision; never infer one from execution."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / 'outputs/referee/task_2_8_20261012_completion/completion_decision.json'


def reviewed_completion():
    if not DECISION.exists():
        return None
    decision = json.loads(DECISION.read_text())
    errors = []
    for relative, expected in decision['evidence_sha256'].items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(relative)
    configuration = ROOT / decision['frozen_configuration']
    config = json.loads(configuration.read_text()) if configuration.exists() else {}
    accepted = (
        decision.get('status') == 'completed_at_representative_annual_reference_scope'
        and decision.get('task_2_8_complete') is True
        and decision.get('production_configuration_frozen') is True
        and config.get('production_configuration_frozen') is True
        and config.get('task_2_8_complete') is True
        and not errors
    )
    return dict(accepted=accepted, hash_errors=errors,
                decision=str(DECISION.relative_to(ROOT)),
                frozen_configuration=decision['frozen_configuration'],
                scope=decision['scope'], gate_status=decision['gate_status'],
                production_grid_completed=False)
