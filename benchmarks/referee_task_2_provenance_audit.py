"""Verify listed snapshot bytes and immutable inputs for the validation report."""

from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def run():
    out=ROOT/'outputs/referee/task_2_7_2_8_20261008_summary'
    state=json.loads((out/'work_state.json').read_text());records=[]
    for entry in state['runs']:
        folder=ROOT/entry['folder'];manifest=folder/'manifest.json'
        if not manifest.exists():continue
        value=json.loads(manifest.read_text())
        hashes=value.get('sha256',value.get('source_hashes',{}))
        missing=[];mismatched=[]
        for rel,expected in hashes.items():
            path=folder/'snapshot'/rel
            if not path.exists():missing.append(rel)
            elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:mismatched.append(rel)
        inputs={}
        for key in ('input_result_sha256','input_evaluations_sha256'):
            if key in value:
                fallback='input_optimizer_validation.json' if key=='input_result_sha256' else 'input_evaluations.jsonl'
                path=folder/value.get('immutable_input_copy',fallback)
                inputs[key]=path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==value[key]
        passed=bool(hashes) and not missing and not mismatched and all(inputs.values())
        records.append({'run':entry['folder'],'listed_snapshot_hashes':len(hashes),
            'missing':missing,'mismatched':mismatched,'immutable_inputs':inputs,
            'status':'passed_listed_hash_checks' if passed else 'requires_investigation'})
    result={'updated_utc':datetime.now(timezone.utc).isoformat(),'runs':records,
        'limitation':'Checks listed archived bytes; does not establish unlisted dependency closure or scientific validity'}
    (out/'provenance_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    failures=[r['run'] for r in records if r['status']!='passed_listed_hash_checks']
    print({'runs_checked':len(records),'failures':failures})
    if failures:raise SystemExit(1)


if __name__=='__main__':run()
