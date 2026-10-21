"""Verify public source aliases against the original executed inventories."""

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path


def read(path):
    data = path.read_bytes()
    if path.suffix == '.gz':
        data = gzip.decompress(data)
    return json.loads(data)


def verify(revision, rows):
    listing = subprocess.check_output(['git', 'ls-tree', '-r', '-z', revision])
    blobs = {}
    for entry in listing.split(b'\0'):
        if entry:
            metadata, name = entry.split(b'\t', 1)
            _, kind, oid = metadata.split()
            if kind == b'blob':
                blobs[name.decode()] = oid.decode()
    process = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for row in rows:
            assert row['path'] in blobs, row['path']
            process.stdin.write((blobs[row['path']] + '\n').encode())
            process.stdin.flush()
            _, kind, size = process.stdout.readline().split()
            assert kind == b'blob'
            data = process.stdout.read(int(size))
            assert process.stdout.read(1) == b'\n'
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], row['path']
    finally:
        process.stdin.close()
        process.stdout.close()
        assert process.wait() == 0
    return {'revision': revision, 'files': len(rows), 'bytes': sum(row['bytes'] for row in rows), 'passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--source-inventory', type=Path, action='append', default=[])
    parser.add_argument('--test-receipt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for path in args.source_inventory:
        result = verify(args.revision, read(path)['files'])
        results.append({'inventory': str(path), **result})
    if args.test_receipt is not None:
        rows = read(args.test_receipt)['test_dependencies_in_pinned_engine_commit']
        relative = [{**row, 'path': 'tests/' + row['path'].split('/tests/', 1)[1]} for row in rows]
        results.append({'test_receipt': str(args.test_receipt), **verify(args.revision, relative)})
    assert results
    report = {'passed': True, 'results': results, 'scope': 'Exact public Git blob identities for the executed scientific source and explicit test dependencies. Local checkout commit identities remain original provenance; captured research workers are replayed separately.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'inventories': len(results), 'verified_files': sum(row['files'] for row in results)}))


if __name__ == '__main__':
    main()
