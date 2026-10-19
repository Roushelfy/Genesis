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
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


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


expected = {'729460_0': ('COMPLETED', '0:0'), '729460_1': ('COMPLETED', '0:0'), '729460_2': ('COMPLETED', '0:0'), '729476_0': ('COMPLETED', '0:0'), '729476_1': ('FAILED', '1:0')}
accounting = subprocess.check_output(['sacct', '-j', '729460,729476', '-P', '--noheader', '--format=JobID,State,ExitCode'], text=True)
observed = {parts[0]: tuple(parts[1:3]) for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
assert all(observed.get(job) == status for job, status in expected.items()), observed
receipt = META / 'cadence-parent-terminal-v1.json'
receipt.write_text(json.dumps({'expected': expected, 'observed': observed, 'sacct': accounting, 'passed': True, 'fine_reference_qualified': False}, indent=2) + '\n')
base = [SCRIPT, receipt]
files = captured('native-sparse-observation-pilot-v1', ['qipc_comparison/artifacts.py', 'qipc_comparison/native_observation_cadence.py', 'qipc_comparison/native_sparse_production.py', 'qipc_comparison/native_production_controls.py'])
files += terminal('native-sparse-observation-pilot-v1') + base
files += [META / p for p in ('native-sparse-observation-plan-v1.json', 'native-sparse-observation-pilot-v1-tasks.json', 'native-observation-cadence-conformance-v1-tasks.json', 'native-observation-cadence-primary-v1-tasks.json', 'native-observation-cadence-audit-v1-tasks.json')]
for name in ('native-observation-cadence-conformance-v1', 'native-observation-cadence-primary-v1', 'native-observation-cadence-audit-v1'):
    files += captured(name, [])
rows = [freeze('native-observation-cadence-pilot-v1', files, sorted((RUN / 'native-sparse-observation-pilot-v1').glob('group*/*/trajectory.npz')), 'Terminal B8 DPR/full/stride10 and SCR/full/stride50 output/partial-reset pilots, plus a separate primary-timing pilot. All requested samples are valid;1080/1176 unsampled full-field holding/reset checks pass. The paired B8/B128 campaigns and independent audit are publication-time pending; no sparse speedup, diverse reliability or physical error is admitted here.')]
files = captured('solid-pad-parent-admission-controls-v1', ['qipc_comparison/artifacts.py', 'qipc_comparison/require_solid_pad_admission.py'])
files += terminal('solid-pad-parent-admission-controls-v1') + base
files += [META / p for p in ('fine-field-dependency-guard-correction-v1.json', 'canonical125-parent-admission-v1-tasks.json', 'solid-pad-parent-admission-controls-v1-tasks.json', 'stvk-canonical125-fields-v1-tasks.json', 'stvk-canonical125-time-contact-fields-v1-tasks.json')]
for name in ('canonical125-parent-admission-v1', 'stvk-canonical125-fields-v1', 'stvk-canonical125-time-contact-fields-v1', 'stvk-canonical125-fields-v2', 'stvk-canonical125-time-contact-fields-v2'):
    files += captured(name, [])
rows += [freeze('fine-parent-admission-guard-v1', files, [], 'Explicit admission preflight passes all ten checks on the admitted400-step coarse parent and intentionally rejects the old125um physical-history failure despite its successful process/scheduler receipt. Pending unguarded analyses were cancelled before execution. Six-parent preflight and replacement field plans are immutable inputs only; new fine histories/fields remain pending. No physical gate, existing payload, tolerance or source is changed by the guard.')]
(META / 'cadence-parent-publication-v1.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows))
