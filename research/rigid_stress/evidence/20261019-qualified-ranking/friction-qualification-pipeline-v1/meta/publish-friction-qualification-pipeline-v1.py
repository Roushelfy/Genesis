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
    (target / 'README.md').write_text('# ' + name + '\n\n' + scope + '\n\nManifest records encoded and original SHA-256/size. Gzip uses mtime0 and an empty filename. Decode to a separate working directory when replaying; immutable archive bytes stay unchanged. Large trajectory/field arrays remain at the explicit data-side locations in the receipts. The original1ff39b077 engine can be reconstructed from the public-base overlay in [the source packet](../native-public-source-overlay-v1/README.md). Captured worker modules retain publication-time variants. Live fine histories and pending qualification/method/plot-data outputs are excluded.\n')
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


expected = {'729509_0': ('COMPLETED', '0:0'), '729509_1': ('FAILED', '1:0'), '729509_2': ('COMPLETED', '0:0'), '729516_0': ('COMPLETED', '0:0')}
accounting = subprocess.check_output(['sacct', '-j', '729509,729516', '-P', '--noheader', '--format=JobID,State,ExitCode'], text=True)
observed = {parts[0]: tuple(parts[1:3]) for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
assert all(observed.get(job) == state for job, state in expected.items()), observed
controls = json.loads((RUN / 'friction-reference-gate-controls-v1/group0/gates/result.json').read_text())
negative = json.loads((RUN / 'friction-reference-gate-controls-v1/group1/incomplete-parent-matrix/result.json').read_text())
parent = json.loads((RUN / 'friction-reference-gate-controls-v1/group2/surface500-parent/result.json').read_text())
plot = json.loads((RUN / 'friction-plot-data-cpu-control-v1/group0/data/result.json').read_text())
assert controls['passed'] and len(controls['checks']) == 19 and all(controls['checks'].values())
assert not negative['passed'] and not negative['fine_reference_qualified'] and not negative['accepted_reference_frames'] and negative['error'] == 'all_seven_required_parent_histories'
assert parent['passed'] and all(parent['runs'][0]['checks'].values())
assert plot['passed'] and len(plot['checks']) == 5 and plot['trajectory_rows'] == 1600 and len(plot['traction']) == 8
receipt = META / 'friction-qualification-pipeline-terminal-v1.json'
receipt.write_text(json.dumps({'expected': expected, 'observed': observed, 'sacct': accounting, 'passed': True, 'gate_checks': 19, 'historical_parent_checks': 10, 'cpu_data_reader_checks': 5, 'incomplete_matrix_intentionally_rejected': True, 'fine_reference_qualified': False}, indent=2) + '\n')
files = [SCRIPT, receipt, META / 'friction-reference-qualification-contract-v1.json', META / 'friction-reference-gate-controls-terminal-v1.json']
files += [META / name for name in ('friction-reference-gate-controls-v1-tasks.json', 'friction-plot-data-cpu-control-v1-tasks.json', 'canonical-friction-reference-qualification-v1-tasks.json', 'canonical-stvk-method-fields-v1-tasks.json', 'canonical-friction-plot-data-v1-tasks.json')]
modules = ['qipc_comparison/artifacts.py', 'qipc_comparison/friction_reference_gates.py', 'qipc_comparison/check_friction_reference_gates.py', 'qipc_comparison/qualify_friction_reference.py', 'qipc_comparison/pad_normal_qualification.py', 'qipc_comparison/require_solid_pad_admission.py']
files += captured('friction-reference-gate-controls-v1', modules) + terminal('friction-reference-gate-controls-v1')
files += captured('friction-plot-data-cpu-control-v1', ['qipc_comparison/artifacts.py', 'qipc_comparison/friction_plot_data.py']) + terminal('friction-plot-data-cpu-control-v1')
files += captured('canonical-friction-reference-qualification-v1', modules)
files += captured('canonical-stvk-method-fields-v1', ['qipc_comparison/contact_fem_exact_columns.py'])
files += captured('canonical-friction-plot-data-v1', ['qipc_comparison/artifacts.py', 'qipc_comparison/friction_plot_data.py'])
external = sorted((RUN / 'friction-plot-data-cpu-control-v1/group0/data').glob('*.npz')) + list((RUN / 'friction-plot-data-cpu-control-v1/group0/data').glob('*.csv'))
row = freeze('friction-qualification-pipeline-v1', files, external, 'Terminal19-check real-data/negative-contract gate controls, an intentionally rejected incomplete parent matrix, ten passing historical500um parent gates and five passing CPU-only force-data-reader checks. Queued seven-parent/fourteen-frame/36-comparison fine qualification and18 normal/friction method-field plus actual-force-data tasks are immutable plans only. Existing2%/5% spatial and0.5%/1% control targets, all source/field identities and original failures are preserved. No live fine outputs, final method errors/ranks or final comparative figures are admitted or frozen here.')
(META / 'friction-qualification-pipeline-publication-v1.json').write_text(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))
