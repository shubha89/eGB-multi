"""Repack the verified reference data without calculating any waveform."""
from pathlib import Path
import argparse
import gzip
import io
import json
import tarfile

from reproduce import MANIFEST, checked_path, digest, read, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checked = verify()
    if checked['status'] != 'passed':
        raise ValueError('Restore and verify the complete reference data first')
    manifest = read(MANIFEST)
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / manifest['data_archive_name']
    with archive.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as stream:
                for name, expected in sorted(manifest['data_sha256'].items()):
                    if name.lower().endswith('.pdf'):
                        raise ValueError('PDF files are excluded from the reference archive')
                    data = checked_path(name).read_bytes()
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    member.mode = 0o644
                    member.mtime = 0
                    stream.addfile(member, io.BytesIO(data))
    record = dict(archive=archive.name, sha256=digest(archive), bytes=archive.stat().st_size,
                  source_files=checked['source_files'], data_files=checked['data_files'],
                  numerical_recalculations=0, publication_performed=False)
    (args.output / 'reference_data_archive.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
