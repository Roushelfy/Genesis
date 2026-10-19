"""Freeze source-bound scene/module/memory measurements and original reader failures."""
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
    (target / 'README.md').write_text('# ' + name + '\n\n' + scope + '\n\nManifest records encoded and original SHA-256/size. Gzip uses mtime0 and an empty filename. Decode to a separate working directory when replaying; immutable archive bytes stay unchanged. Large trajectory/field arrays remain at the explicit data-side locations in the receipts. The original1ff39b077 engine can be reconstructed from the public-base overlay in [the source packet](../native-public-source-overlay-v1/README.md). Captured worker modules retain publication-time variants. Live fine histories and pending method/ranking/figure outputs are excluded.\n')
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

expected = {'730471_0': ('FAILED', '1:0'), '730473_0': ('COMPLETED', '0:0'), '730460_0': ('COMPLETED', '0:0'), '730461_0': ('COMPLETED', '0:0'), '730463_0': ('COMPLETED', '0:0'), '730464_0': ('FAILED', '1:0'), '730498_0': ('COMPLETED', '0:0')}
accounting = subprocess.check_output(['sacct', '-j', '730471,730473,730460,730461,730463,730464,730498', '-P', '--noheader', '--format=JobID,State,ExitCode,Elapsed'], text=True)
observed = {parts[0]: tuple(parts[1:3]) for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
assert all(observed.get(job) == status for job, status in expected.items()), observed
summary = json.loads((RUN / 'native-scene-summary-v2/group0/summary/result.json').read_text())
module = json.loads((RUN / 'native-module-replay-canonical-audit-v1/group0/audit/result.json').read_text())
memory = json.loads((RUN / 'native-module-memory-canonical-audit-v2/group0/audit/result.json').read_text())
assert summary['passed'] and len(summary['rows']) == 61 and len(summary['checks']) == 14454 and all(row['passed'] for row in summary['checks'])
assert len(summary['receipts']) == 4267 and sum(row['sampled_process_memory'] is not None for row in summary['rows']) == 20
assert module['passed'] and len(module['rows']) == 4 and module['engine_revision'] == '1ff39b0775200120737225fdfb8631c9b1fc5fa7'
assert memory['passed'] and len(memory['rows']) == 4 and all(row['passed'] for row in memory['checks'])
contract = json.loads((META / 'native-module-canonical-contract-v1.json').read_text())
for name in ['native_production_controls.py', 'native_module_memory.py']:
    campaign = 'native-module-memory-canonical-v1' if name == 'native_module_memory.py' else 'native-module-replay-canonical-v2'
    assert identity(RUN / campaign / 'analysis-source/research/rigid_stress/qipc_comparison' / name)['sha256'] == contract['workers'][name]
strengthening = json.loads((META / 'native-module-canonical-audit-strengthening-v1.json').read_text())
for name, campaign in [('native_module_replay_audit.py', 'native-module-replay-canonical-audit-v1'), ('audit_native_module_memory.py', 'native-module-memory-canonical-audit-v1')]:
    assert identity(RUN / campaign / 'analysis-source/research/rigid_stress/qipc_comparison' / name)['sha256'] == strengthening['reader_sha256'][name]
fix = json.loads((META / 'native-module-memory-audit-receipt-fix-v1.json').read_text())
assert identity(RUN / 'native-module-memory-canonical-audit-v2/analysis-source/research/rigid_stress/qipc_comparison/audit_native_module_memory.py')['sha256'] == fix['replacement_reader_sha256']
(META / 'native-performance-terminal-v1.json').write_text(json.dumps({'accounting': accounting, 'expected': expected, 'scene_checks': len(summary['checks']), 'scene_rows': len(summary['rows']), 'scene_receipts': len(summary['receipts']), 'module_rows': len(module['rows']), 'module_memory_checks': len(memory['checks']), 'module_memory_rows': len(memory['rows']), 'whole_goal_complete': False}, indent=2) + '\n')
files = [SCRIPT]
files += [META / name for name in ('native-performance-terminal-v1.json', 'native-module-canonical-v2-tasks.json', 'native-module-memory-canonical-v1-tasks.json', 'native-module-canonical-audit-v1-tasks.json', 'native-module-memory-canonical-audit-v1-tasks.json', 'native-module-canonical-contract-v1.json', 'native-module-canonical-preflight-failure-v1.json', 'native-module-canonical-audit-strengthening-v1.json', 'native-module-pending-walltime-v1.json', 'native-module-memory-audit-receipt-fix-v1.json', 'native-performance-b8-comment-equivalence-v1.json', 'native-performance-array-class-b8-reconstruction-v1.py', 'native-scene-summary-v1-tasks.json')]
files += list(META.glob('native-module-memory-*-v1-task.json'))
modules = ['qipc_comparison/' + name for name in ('artifacts.py', 'native_production_controls.py', 'native_observation_cadence.py', 'native_module_memory.py', 'native_module_replay_audit.py', 'audit_native_module_memory.py', 'native_scene_summary.py')]
modules += ['shell_comparison/memory.py']
for campaign in ['native-scene-summary-v1', 'native-scene-summary-v2', 'native-module-replay-canonical-v2', 'native-module-memory-canonical-v1', 'native-module-replay-canonical-audit-v1', 'native-module-memory-canonical-audit-v1', 'native-module-memory-canonical-audit-v2']:
    present = [rel for rel in modules if (RUN / campaign / 'analysis-source/research/rigid_stress' / rel).is_file()]
    files += captured(campaign, present) + terminal(campaign)
    files += list((RUN / campaign).glob('group*/**/*.csv'))
external = []
for campaign in ['native-module-replay-canonical-v2', 'native-module-memory-canonical-v1']:
    external += list((RUN / campaign).glob('group*/**/*.npz'))
row = freeze('native-performance-summary-v1', files, external, 'Terminal source-bound native performance:61 scene/cadence configurations and20 independent sampled-memory profiles pass14454 checks with4267 file identities. Exact B8 comment-only source equivalence preserves original revision/patch. New1ff engine B8 module replay and four matched separate process-memory runs pass complete-history/reset/typed-buffer/source and configuration audits; all primary rates exclude scene dynamics and retain device restoration/native acceptance. Actual memory counts/timestamps/gaps are explicit, including long gaps in earlier profiles; no capacity maximum, policy throughput or physical-pad error is inferred. Original reader/preflight failures and scheduler limit updates remain intact. Fine references, final friction errors/ranks/figures and full-goal completion remain pending.')
(META / 'native-performance-publication-v1.json').write_text(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))
