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


expected = {'729545_0': ('COMPLETED', '0:0'), '729549_0': ('COMPLETED', '0:0'), '729572_0': ('FAILED', '1:0')}
accounting = subprocess.check_output(['sacct', '-j', '729545,729549,729572', '-P', '--noheader', '--format=JobID,State,ExitCode'], text=True)
observed = {parts[0]: tuple(parts[1:3]) for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
assert all(observed.get(job) == status for job, status in expected.items()), observed
measures = json.loads((RUN / 'stress-error-measures-controls-v1/group0/controls/result.json').read_text())
admission = json.loads((RUN / 'pad-candidate-admission-controls-v1/group0/admission/result.json').read_text())
negative = json.loads((RUN / 'stvk-ranking-admission-negative-v1/group0/unqualified-reference/result.json').read_text())
assert measures['passed'] and len(measures['checks']) == 99 and measures['independent_vector_bound_cases'] == 2400
assert admission['passed'] and len(admission['checks']) == 30 and len(admission['retained_every_step_failures']) == 2
assert not negative['passed'] and not negative['errors_admitted'] and not negative['rows'] and not negative['conditional_rankings']
contract = json.loads((META / 'canonical-stvk-ranking-contract-v1.json').read_text())
for name, expected_hash in contract['source_sha256'].items():
    path = RUN / 'canonical-stvk-method-ranking-v1/analysis-source/research/rigid_stress/qipc_comparison' / name
    assert identity(path)['sha256'] == expected_hash, name
receipt = META / 'stvk-ranking-reader-controls-terminal-v1.json'
receipt.write_text(json.dumps({'expected': expected, 'observed': observed, 'sacct': accounting, 'passed': True, 'dimensional_error_controls': 99, 'candidate_admission_controls': 30, 'vector_bound_cases': 2400, 'unqualified_reference_rejected_without_error_rows_or_ranks': True, 'actual_fine_errors_and_ranks_pending': True}, indent=2) + '\n')
files = [SCRIPT, receipt, META / 'stvk-ranking-controls-terminal-v1.json', META / 'canonical-stvk-ranking-contract-v1.json', META / 'canonical-stvk-surface-export-update-v1.json']
files += [META / name for name in ('stress-error-measures-controls-v1-tasks.json', 'pad-candidate-admission-controls-v1-tasks.json', 'stvk-ranking-admission-negative-v1-tasks.json', 'canonical-stvk-method-fields-v2-tasks.json', 'canonical-stvk-method-ranking-v1-tasks.json')]
files += captured('stress-error-measures-controls-v1', ['qipc_comparison/artifacts.py', 'qipc_comparison/stress_error_measures.py', 'qipc_comparison/check_stress_error_measures.py']) + terminal('stress-error-measures-controls-v1')
files += captured('pad-candidate-admission-controls-v1', ['qipc_comparison/artifacts.py', 'qipc_comparison/pad_candidate_admission.py', 'qipc_comparison/check_pad_candidate_admission.py']) + terminal('pad-candidate-admission-controls-v1')
modules = ['qipc_comparison/artifacts.py', 'qipc_comparison/stress_error_measures.py', 'qipc_comparison/pad_candidate_admission.py', 'qipc_comparison/rank_stvk_pad_methods.py']
files += captured('stvk-ranking-admission-negative-v1', modules) + terminal('stvk-ranking-admission-negative-v1')
files += captured('canonical-stvk-method-fields-v2', ['qipc_comparison/contact_fem_exact_columns.py'])
files += captured('canonical-stvk-method-ranking-v1', modules)
row = freeze('stvk-ranking-reader-controls-v1', files, [], 'Terminal99 complete-array/dimensional/invariant/interval checks,30 candidate own-admission checks and an intentionally rejected unqualified-reference rank attempt. All81 published normal errors are reproduced, full q1 surface/resultant arrays and raw peaks/locations independently checked, and2400 vector norm cases validate only the interval mathematics. Exact worker variants and the prospective18-case q9/q25 surface-export/final54-row ranking plans are retained. Full original arrays remain on data with hashes in the control receipts. No live fine outputs, final physical errors/ranks or final figures are admitted here; every-step native failures remain separately unranked.')
(META / 'stvk-ranking-reader-controls-publication-v1.json').write_text(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))
