"""Freeze admitted normal configuration/error/cost associations and strict friction guards."""
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


expected = {'730051_0': ('COMPLETED', '0:0'), '730051_1': ('FAILED', '1:0')}
accounting = subprocess.check_output(['sacct', '-j', '730051', '-P', '--noheader', '--format=JobID,State,ExitCode'], text=True)
observed = {parts[0]: tuple(parts[1:3]) for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
assert all(observed.get(job) == status for job, status in expected.items()), observed
positive = json.loads((RUN / 'pad-cost-accuracy-controls-v1/group0/normal/result.json').read_text())
negative = json.loads((RUN / 'pad-cost-accuracy-controls-v1/group1/unqualified-friction/result.json').read_text())
assert positive['passed'] and len(positive['checks']) == 478 and len(positive['matched_configurations']) == 24 and len(positive['reference_case_error_cost_rows']) == 120
assert not negative['passed'] and not negative['matched_configurations'] and not negative['reference_case_error_cost_rows']
assert not list((RUN / 'pad-cost-accuracy-controls-v1/group1/unqualified-friction').glob('*.csv'))
constructors = json.loads((META / 'pad-cost-constructor-provenance-v1.json').read_text())
assert constructors['passed'] and len(constructors['comparisons']) == 28
for receipt in constructors['receipts']:
    assert identity(pathlib.Path(receipt['path'])) == receipt
contract = json.loads((META / 'pad-cost-accuracy-contract-v1.json').read_text())
queued = RUN / 'pad-friction-cost-accuracy-v1/analysis-source/research/rigid_stress/qipc_comparison/pad_cost_accuracy.py'
assert identity(queued)['sha256'] == contract['module_sha256']
files = [SCRIPT] + [META / name for name in ('pad-cost-accuracy-terminal-v1.json', 'pad-cost-accuracy-contract-v1.json', 'pad-cost-accuracy-controls-v1-tasks.json', 'pad-friction-cost-accuracy-v1-tasks.json', 'audit-pad-cost-constructor-provenance-v1.py', 'pad-cost-constructor-provenance-v1.json', 'canonical125-halfdt-walltime-extension-v1.json')]
modules = ['qipc_comparison/' + name for name in ('artifacts.py', 'pad_cost_matrix.py', 'pad_cost_accuracy.py')]
files += captured('pad-cost-accuracy-controls-v1', modules) + terminal('pad-cost-accuracy-controls-v1')
files += list((RUN / 'pad-cost-accuracy-controls-v1/group0/normal').glob('*.csv'))
files += captured('pad-friction-cost-accuracy-v1', modules)
files += [pathlib.Path(receipt['path']) for receipt in constructors['receipts']]
for campaign in ('native-pad-full-source-stability-v1', 'compliant-pad-fine-coarse-history-v1', 'pad-primary-cost-v1', 'pad-primary-batch4-v1'):
    filename = 'extra-source0.json' if campaign == 'compliant-pad-fine-coarse-history-v1' else 'source.json'
    files.append(RUN / campaign / filename)
    if (RUN / campaign / 'extra-source0.json').exists():
        files.append(RUN / campaign / 'extra-source0.json')
row = freeze('pad-cost-accuracy-v1', files, [], 'Terminal478 checks bind24 admitted normal research-workflow timing configurations to120 applicable qualified reference-case metric associations. Runtime libraries, captured64-bit precision, exact masses/meshes/assets/solver parameters, complete own-run audits and separate memory provenance are verified;28 material/recovery constructor comparisons and15 changed-configuration rejections are retained. Max-only rows associate raw Mises magnitude only, sparse memory stays unavailable, and reference errors at0.2s are not relabelled as exact timed-endpoint errors. An unqualified friction input writes no joined CSV. Final four-configuration/36-row friction consolidation waits for actual fine-reference/error admission. Exact denied half-dt time-extension resource evidence is retained without admitting or moving live reference data. Production collider speeds inherit no pad errors; the full goal remains active.')
(META / 'pad-cost-accuracy-publication-v1.json').write_text(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))
