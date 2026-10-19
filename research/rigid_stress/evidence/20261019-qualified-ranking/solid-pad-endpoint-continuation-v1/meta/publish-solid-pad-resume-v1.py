"""Freeze verified endpoint continuation and exact terminal/live source receipts."""
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


expected = {'730362_0': ('COMPLETED', '0:0'), '730363_0': ('COMPLETED', '0:0'), '730366_0': ('COMPLETED', '0:0')}
accounting = subprocess.check_output(['sacct', '-j', '730362,730363,730366', '-P', '--noheader', '--format=JobID,State,ExitCode'], text=True)
observed = {parts[0]: tuple(parts[1:3]) for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
assert all(observed.get(job) == status for job, status in expected.items()), observed
run = RUN / 'solid-pad-resume-coarse-control-v1/group0/reference'
audit = json.loads((RUN / 'solid-pad-resume-equivalence-v1/group0/audit/result.json').read_text())
controls = json.loads((RUN / 'solid-pad-resume-input-controls-v2/group0/controls/result.json').read_text())
assert audit['passed'] and len(audit['checks']) == 33 and all(audit['checks'].values())
assert controls['passed'] and len(controls['checks']) == 15 and all(controls['checks'].values())
proof = json.loads((RUN / 'solid-pad-resume-preparation-v1/result.json').read_text())
negative = json.loads((RUN / 'solid-pad-resume-live-negative-v1/result.json').read_text())
assert proof['passed'] and proof['prefix_steps'] == 200 and not negative['passed'] and negative['terminal_scheduler_state'] == 'RUNNING'
for receipt in proof['receipts']:
    assert identity(pathlib.Path(receipt['path'])) == receipt
strengthening = json.loads((META / 'solid-pad-resume-provenance-strengthening-v1.json').read_text())
assert strengthening['restoration_function_and_driver_unchanged']
contract = json.loads((META / 'solid-pad-resume-contract-v1.json').read_text())
for name, digest in contract['source_sha256'].items():
    assert identity(run.parent.parent / 'analysis-source/research/rigid_stress/qipc_comparison' / name)['sha256'] == digest
files = [SCRIPT] + [META / name for name in ('solid-pad-resume-terminal-v1.json', 'solid-pad-resume-contract-v1.json', 'solid-pad-resume-provenance-strengthening-v1.json', 'solid-pad-resume-coarse-control-v1-tasks.json', 'solid-pad-resume-input-controls-v2-tasks.json', 'solid-pad-resume-equivalence-v1-tasks.json', 'solid-pad-resume-live-negative-v1.log')]
files += [RUN / name / 'result.json' for name in ('solid-pad-resume-preparation-v1', 'solid-pad-resume-live-negative-v1')]
modules = ['qipc_comparison/' + name for name in ('artifacts.py', 'plates.py', 'plate_solid_pad.py', 'solid_pad_checkpoint.py', 'check_solid_pad_checkpoint.py')]
files += captured('solid-pad-resume-coarse-control-v1', modules + ['qipc_comparison/contact_fem_exact_columns.py', 'qipc_comparison/audit_solid_pad.py']) + terminal('solid-pad-resume-coarse-control-v1')
files += captured('solid-pad-resume-input-controls-v2', modules) + terminal('solid-pad-resume-input-controls-v2')
files += captured('solid-pad-resume-equivalence-v1', modules + ['qipc_comparison/audit_solid_pad_resume.py', 'qipc_comparison/friction_reference_gates.py', 'qipc_comparison/pad_normal_qualification.py']) + terminal('solid-pad-resume-equivalence-v1')
files += [pathlib.Path(row['file']['path']) for row in proof['physics_modules']]
external = list((RUN / 'solid-pad-resume-coarse-control-v1/group0').glob('*/**/*.npz'))
row = freeze('solid-pad-endpoint-continuation-v1', files, external, 'Terminal400-step coarse physical reference versus its exact200-step saved prefix plus200 newly integrated steps:33 independent history/momentum/complete-array/common-field equivalence checks and15 strengthened provenance controls pass. Complete float64 body/plate/lagged-force state, original global clock/loading, prefix bytes, exact source modules and separate inherited/new-step accounting are retained. Actual live-parent preparation is rejected. Full field/peak changes remain below predeclared1e-8 targets, with maximum nodal velocity difference2.47e-12m/s below1e-10. Stronger module-coverage checks preserve identical restoration AST and driver bytes. Large states/columns stay on data with hashes. No live fine task is resumed, changed or admitted; final fine convergence, error ranks/figures and matched friction costs remain pending.')
(META / 'solid-pad-resume-publication-v1.json').write_text(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))
