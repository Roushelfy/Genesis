"""Publish the native continuous-wrench path, original-point budget checks and exact live failure."""
import hashlib
import json
import os
import shutil
from pathlib import Path

root = Path(__file__).parent
destination = Path('research/rigid_stress/evidence/20261011-nonconvex/native-continuous-wrench-checkpoint-v1')
assert not destination.exists(), destination
folders = ['native-wrench-diagnostic-v1','native-wrench-full-regression-v2',
           'native-wrench-certificate-audit-v1','native-wrench-online-b1024-v1']
files = []
skip = {'cache','tmp','__pycache__','.pytest_cache'}
for folder in folders:
    directory = root / folder
    assert directory.is_dir()
    for current, dirs, names in os.walk(directory):
        dirs[:] = [name for name in dirs if name not in skip]
        files.extend(Path(current) / name for name in names)
    for before in directory.rglob('source.sha256'):
        if not any(part in skip for part in before.relative_to(directory).parts):
            assert before.read_bytes() == before.with_name('source-after.sha256').read_bytes(), before
for backend in ('cpu','cuda'):
    assert '1 failed' in (root/f'native-wrench-diagnostic-v1/{backend}/tests.log').read_text()
    assert '12 passed' in (root/f'native-wrench-diagnostic-v1/{backend}/tests.log').read_text()
    assert '51 passed' in (root/f'native-wrench-full-regression-v2/{backend}/tests.log').read_text()
    audit = json.loads((root/f'native-wrench-certificate-audit-v1/{backend}/results/summary.json').read_text())
    assert audit['status'] == [[0,6]]
    assert audit['cpu_conservative_polar_upper_bound'] < -1e-4 and audit['certificate_margin'] > 1e-3
validation = json.loads((root/'native-wrench-online-b1024-v1/validation.json').read_text())
assert validation['completed_steps'] == 602 and validation['cpu_full_fields'] == 4096
assert all(validation['original_input_bitwise_checks'].values())
names = ['run_native_wrench_diagnostic_v1.sh','run_native_wrench_full_regression_v2.sh',
         'probe_native_wrench_certificate_v1.py','run_native_wrench_certificate_v1.sh',
         'run_native_wrench_online_b1024_v1.sh','check_native_wrench_online_v1.py',
         'run_check_native_wrench_online_v1.sh','check-native-wrench-online-v1.log',
         'extract_602_fixture_v1.py','run_extract_602_fixture_v1.sh','extract-602-fixture-v1.log',
         'extract_602_fixture_v2.py','run_extract_602_fixture_v2.sh','extract-602-fixture-v2.log',
         'archive_online_checkpoint_v13.py','run_archive_online_checkpoint_v13.sh']
files.extend(root/name for name in names)
for pattern in ('native-wrench-*allocation.log','extract-602-*allocation.log','check-native-wrench-*allocation.log'):
    files.extend(root.glob(pattern))
for original in files:
    assert original.is_file(), original
    assert original.stat().st_size < 20_000_000, original
destination.mkdir(parents=True)
entries = []
for original in sorted(set(files)):
    artifact = destination/'raw'/original.relative_to(root)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, artifact)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
    entries.append({'original':str(original),'artifact':str(artifact.relative_to(destination)),
                    'bytes':original.stat().st_size,'sha256':digest})
manifest = {'scope':__doc__,'source_base_revision':'87d04e451f18879836667b47632c654012d53ef9',
            'requirement_revision':'1d6babf77848c46ba616da8bbb7062260a7ccd09',
            'native_regression':{'cpu':51,'cuda':51},'live_validation':validation,
            'note':'Production six-component continuous diagnosis and original-point stopping are tested. B1024 aborts at602 on a certified incompatible original component; aggregate partitioning remains unselected. Full large-batch acceptance, matched four-scope performance and plateau are unfinished.',
            'files':entries}
path = destination/'manifest.json'
path.write_text(json.dumps(manifest,indent=2)+'\n')
for entry in entries:
    assert hashlib.sha256(Path(entry['original']).read_bytes()).hexdigest() == entry['sha256']
    assert hashlib.sha256((destination/entry['artifact']).read_bytes()).hexdigest() == entry['sha256']
print(json.dumps({'artifacts':len(entries),'bytes':sum(entry['bytes'] for entry in entries),
                  'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}),flush=True)
