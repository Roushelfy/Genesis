"""Freeze immutable pending contact-order pair scheduling without admitting any new science."""
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

import datetime
import re

contract=json.loads((META/'contact125-pair-scheduling-contract-v1.json').read_text())
release=json.loads((META/'contact125-pair-dependency-release-v1.json').read_text())
assert release['passed'] and all(release['checks'].values())
assert not release['physics_modified'] and not release['tasks_resubmitted']
for path,sha in contract['captured_hashes'].items():
    assert identity(pathlib.Path(path))['sha256']==sha
state=lambda j:subprocess.check_output(['scontrol','show','job',j,'-o'],text=True)
dep=lambda s:re.search(r'(?:^| )Dependency=(\S+)',s).group(1)
current={j:state(j) for j in contract['original_states']}
assert all('JobState=PENDING' in current[j] and 'afterok:730699_' in dep(current[j]) for j in contract['target_job_elements'])
assert all(dep(current[j])==dep(contract['original_states'][j]) for j in current if j not in contract['target_job_elements'])
guard=state('730699_0')
assert 'JobState=PENDING' in guard and all(j in dep(guard) for j in ['afterok:729312_1','afterok:729314_0'])
assert not (RUN/'contact125-pair-parent-admission-v1/group0').exists()
source=RUN/'contact125-pair-parent-admission-v1/analysis-source/research/rigid_stress/qipc_comparison/require_solid_pad_admission.py'
assert identity(source)['sha256']==contract['unchanged_admission_module_sha256']
live={j:state(j) for j in ['729312_1','729314_0','729314_1']}
assert all('JobState=RUNNING' in value for value in live.values())
receipt={'recorded_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'guard_job':'730699','guard_state':guard,'selected_jobs':contract['target_job_elements'],'all_original_capture_hashes_unchanged':True,'new_admission_worker_exactly_original':True,'nonselected_dependencies_unchanged':True,'current_original_states':current,'live_fine_tasks':live,'new_scientific_results':False,'fine_reference_qualified':False,'accepted_reference_frames':[],'whole_goal_complete':False}
proof=META/'contact125-pair-scheduling-verified-v1.json'
assert not proof.exists();proof.write_text(json.dumps(receipt,indent=2)+'\n')
files=[SCRIPT,proof]+[META/name for name in ['contact125-pair-scheduling-contract-v1.json','contact125-pair-parent-admission-v1-tasks.json','contact125-pair-dependency-release-v1.json']]
files+=captured('contact125-pair-parent-admission-v1',['qipc_comparison/require_solid_pad_admission.py','qipc_comparison/artifacts.py'])
files+=[pathlib.Path(p) for p in contract['captured_hashes']]
row=freeze('contact125-pair-scheduling-v1',files,[], 'Immutable pending scheduling only: guard730699 waits for actual successful729312_1/729314_0, then reuses the exact original strict complete-history/momentum/two-endpoint admission worker. Only original contact-order field array tasks729480_0/_1 now wait on this guard instead of the unrelated800-step half-dt parent. Their q9/depth2, q9/depth4 and q25/depth4 commands, source hashes, tolerances and output paths remain unchanged. The two half-dt tasks and all full-qualification/ranking/figure/cost dependencies retain original gates. This packet contains no live physical output or new scientific result, no admitted fine reference or method rank, and no completed-goal claim.')
report=META/'contact125-pair-scheduling-publication-v1.json'
assert not report.exists();report.write_text(json.dumps(row,indent=2)+'\n')
print(json.dumps(row))
