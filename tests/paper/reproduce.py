"""Restore, verify and plot the checksummed paper data; no search is launched."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'tests/paper/reference_data_manifest.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked_path(name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '.git' in relative.parts:
        raise ValueError('Unsafe evidence path: ' + name)
    target = (ROOT / str(relative)).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError('Evidence escapes the checkout')
    return target


def verify():
    manifest = read(MANIFEST)
    missing, changed = [], []
    groups = manifest['source_sha256'], manifest['data_sha256']
    for group in groups:
        for name, expected in group.items():
            path = checked_path(name)
            if not path.is_file():
                missing.append(name)
            elif digest(path) != expected:
                changed.append(name)
    return dict(status='passed' if not missing and not changed else 'failed',
                source_files=len(groups[0]), data_files=len(groups[1]),
                missing=missing, changed=changed, scope=manifest['scope'],
                waveform_generation=False, broad_searches=0)


def restore(archive):
    """Check the complete archive before creating any data files."""
    expected = read(MANIFEST)['data_sha256']
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names) or set(names) != set(expected):
            raise ValueError('Archive member set differs from the evidence manifest')
        for member in members:
            if not member.isfile() or member.name.lower().endswith('.pdf'):
                raise ValueError('Only regular, non-PDF evidence files may be restored')
            target = checked_path(member.name)
            if target.exists() and digest(target) != expected[member.name]:
                raise ValueError('Refusing to overwrite changed evidence: ' + member.name)
            if hashlib.sha256(stream.extractfile(member).read()).hexdigest() != expected[member.name]:
                raise ValueError('Archive checksum mismatch: ' + member.name)
        for member in members:
            target = checked_path(member.name)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as destination:
                shutil.copyfileobj(stream.extractfile(member), destination)
    return verify()


def show():
    checked = verify()
    if checked['status'] != 'passed':
        raise ValueError('Restore and verify the evidence first')
    with (ROOT / 'tests/paper/data/representative_sources_updated.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    return dict(status='passed_saved_result_verification', nominal_sources=len(rows),
                broader_search_results=sum(bool(r['current_broad_search_match']) for r in rows),
                figures=6, scope=checked['scope'], waveform_generation=False,
                broad_searches_launched=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify', help='Check restored data and release source hashes')
    sub.add_parser('show', help='Summarize the saved paper results')
    p = sub.add_parser('restore', help='Restore the separately distributed data archive')
    p.add_argument('archive', type=Path)
    p = sub.add_parser('figures', help='Replay six figures from saved data; no numerical campaign')
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'restore':
        result = restore(args.archive)
    elif args.command == 'verify':
        result = verify()
    elif args.command == 'show':
        result = show()
    else:
        if verify()['status'] != 'passed':
            raise ValueError('Restore and verify the complete reference data first')
        script = ROOT / 'tests/paper/plot_figures.py'
        subprocess.run([sys.executable, str(script), '--output', str(args.output)], check=True)
        result = read(args.output.resolve() / 'figure_replay.json')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] != 'failed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
