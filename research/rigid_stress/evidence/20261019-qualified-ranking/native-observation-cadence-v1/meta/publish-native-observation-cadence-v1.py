"""Freeze terminal sparse-output pilots and parent-admission controls, preserving exact bytes."""
import gzip
import hashlib
import io
import json
import pathlib
import subprocess

RUN = pathlib.Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
WORK = pathlib.Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
DEST = WORK / 'research/rigid_stress/evidence/20261019-qualified-ranking'
META = RUN / 'meta'
SCRIPT = pathlib.Path(__file__)


def identity(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hasher.hexdigest()}


def freeze(name, files, external, scope):
    target = DEST / name
    target.mkdir(exist_ok=False)
    rows = []
    for source in sorted(set(files)):
        original = source.read_bytes()
        relative = source.relative_to(RUN)
        encoded = original
        compressed = bool(original) and (source.suffix == '.log' or (source.suffix == '.json' and len(original) > 8192) or source.suffix == '.patch')
        if compressed:
            memory = io.BytesIO()
            with gzip.GzipFile(filename='', mode='wb', fileobj=memory, mtime=0) as stream:
                stream.write(original)
            encoded = memory.getvalue()
            relative = pathlib.Path(str(relative) + '.gz')
        copy = target / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(encoded)
        rows.append({'source': str(source), 'copy': str(relative), 'bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest(), 'encoding': 'gzip' if compressed else 'identity', 'original_bytes': len(original), 'original_sha256': hashlib.sha256(original).hexdigest()})
    manifest = {'files': rows, 'external_files': [identity(path) for path in external], 'scope': scope, 'payloads': len(rows), 'payload_bytes': sum(row['bytes'] for row in rows)}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (target / 'README.md').write_text('# ' + name + '\n\n' + scope + '\n\nManifest records encoded and original SHA-256/size. Gzip uses mtime0 and an empty filename. Decode to a separate working directory when replaying; immutable archive bytes stay unchanged. Large trajectory/field arrays remain at the explicit data-side locations in the receipts. The original1ff39b077 engine can be reconstructed from the public-base overlay in [the source packet](../native-public-source-overlay-v1/README.md). Captured worker modules retain publication-time variants. Live fine histories and pending cadence-campaign outputs are excluded.\n')
    return {'packet': name, 'payloads': len(rows), 'bytes': manifest['payload_bytes'], 'external_arrays': len(external)}


def captured(campaign, modules):
    root = RUN / campaign
    files = [root / n for n in ('submission.json', 'source.json', 'source.patch', 'analysis-files.json', 'tasks.json', 'job.sh')]
    files += list(root.glob('workload-*.sh'))
    for module in modules:
        files.append(root / 'analysis-source/research/rigid_stress' / module)
    return files


def terminal(campaign):
    root = RUN / campaign
    return [p for p in root.glob('group*/**/*') if p.is_file() and p.suffix in ('.json', '.txt', '.log') and 'cache' not in p.parts and 'tmp' not in p.parts] + list(root.glob('logs/*.log'))


expected = {f'{job}_{group}': ('COMPLETED', '0:0') for job in ('729467', '729470') for group in range(2)}
expected['729473_0'] = ('COMPLETED', '0:0')
accounting = subprocess.check_output(['sacct', '-j', '729467,729470,729473', '-P', '--noheader', '--format=JobID,State,ExitCode'], text=True)
observed = {parts[0]: tuple(parts[1:3]) for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
assert all(observed.get(job) == status for job, status in expected.items()), observed
result = json.loads((RUN / 'native-observation-cadence-audit-v1/group0/audit/result.json').read_text())
assert result['passed'] and len(result['checks']) == 521 and result['case_count'] == 26 and len(result['timing_rows']) == 26
receipt = META / 'native-observation-cadence-terminal-v1.json'
receipt.write_text(json.dumps({'expected': expected, 'observed': observed, 'sacct': accounting, 'checks_passed': 521, 'case_count': 26, 'source_files_rehashed': result['source_files_rehashed'], 'passed': True, 'physical_reference_qualified': False}, indent=2) + '\n')
modules = ['qipc_comparison/artifacts.py', 'qipc_comparison/native_observation_cadence.py', 'qipc_comparison/native_sparse_production.py', 'qipc_comparison/native_production_controls.py', 'qipc_comparison/audit_native_production_controls.py', 'qipc_comparison/audit_native_observation_cadence.py']
files = [SCRIPT, receipt, META / 'native-sparse-observation-plan-v1.json', META / 'spatial-parent-dependency-split-v1.json', META / 'canonical125-spatial-parent-admission-v1-tasks.json']
external = []
for name in ('native-observation-cadence-conformance-v1', 'native-observation-cadence-primary-v1', 'native-observation-cadence-audit-v1'):
    files += captured(name, modules) + terminal(name)
    external += list((RUN / name).glob('group*/*/trajectory.npz'))
files += list((RUN / 'native-observation-cadence-audit-v1/group0/audit').glob('*.csv'))
files += captured('canonical125-spatial-parent-admission-v1', ['qipc_comparison/artifacts.py', 'qipc_comparison/require_solid_pad_admission.py'])
row = freeze('native-observation-cadence-v1', files, sorted(external), 'Terminal B8/B12813-mode primary/conformance campaigns and independent521-check audit, with26 matched timing rows, all motion exactly matching pure rigid and sampled stress within predeclared budgets. Three warm within-process1200-step windows per case,900 warmup,dt0.01s/substeps1, GPU outputs, unchanged native acceptance and repeated frozen eight-condition inputs. Sparse observations do not cover unsampled peaks, diverse-SCR reliability, policy transitions or physical collider error. New spatial-parent preflight/dependency split is immutable pending input only; all live fine outputs remain excluded.')
(META / 'native-observation-cadence-publication-v1.json').write_text(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))
