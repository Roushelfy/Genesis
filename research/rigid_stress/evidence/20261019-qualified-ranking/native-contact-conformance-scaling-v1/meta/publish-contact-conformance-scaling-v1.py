import gzip
import hashlib
import json
from pathlib import Path
import subprocess
N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
W = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
E = W / 'research/rigid_stress/evidence/20261019-qualified-ranking'

def identity(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda : f.read(1024 * 1024), b''):
            h.update(part)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()}

def terminal(job, expected):
    result = subprocess.check_output(['sacct', '-j', job, '-X', '-n', '-P', '--format=JobID,State,ExitCode'], text=True)
    assert result.strip() == job + '|' + expected, result
    return result

def campaign(paths, name, done=True, groups=None):
    root = N / name
    for pattern in ('submission.json', 'source*.json', 'source*.patch', 'tasks.json', 'analysis-files.json', 'job.sh', 'workload-*.sh'):
        paths.update((p for p in root.glob(pattern) if p.is_file()))
    if done:
        for group in sorted(root.glob('group*')):
            if not group.is_dir() or (groups is not None and int(group.name[5:]) not in groups):
                continue
            for pattern in ('returncode.txt', '*.returncode.txt', '*.log', '*.json', '**/result.json', '**/start.json', '**/constructed.json', '**/restarted.json', '**/step*.json', '**/conformance.json', '**/junit.xml'):
                paths.update((p for p in group.glob(pattern) if p.is_file()))
        for p in (root / 'logs').glob('*.log'):
            if groups is None or int(p.stem.split('_')[-1]) in groups:
                paths.add(p)

def publish(name, paths, scope):
    paths.add(Path(__file__))
    dest = E / name
    dest.mkdir(exist_ok=True)
    rows = []
    for source in sorted(paths):
        data = source.read_bytes()
        assert len(data) < 32000000, str(source)
        compressed = len(data) > 100000 or source.suffix == '.log'
        payload = gzip.compress(data, mtime=0) if compressed else data
        rel = str(source.relative_to(N)) + ('.gz' if compressed else '')
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        assert (gzip.decompress(payload) if compressed else payload) == data
        rows.append({'source': str(source), 'copy': rel, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(), 'encoding': 'gzip' if compressed else 'identity', 'original_bytes': len(data), 'original_sha256': hashlib.sha256(data).hexdigest()})
    m = {'files': rows, 'verified_file_count': len(rows), 'verified_bytes': sum((r['bytes'] for r in rows)), 'whole_goal_complete': False, 'scope': scope}
    (dest / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    (dest / 'README.md').write_text('# ' + name + '\n\n' + scope + '\n\nLarge JSON and raw logs use deterministic gzip; the manifest preserves original and encoded SHA-256/size. Decode a working copy with Python gzip before inspecting raw records. Full NPZ arrays/traces remain on data with exact identities. No live reference output is included.\n')
    print(name, len(rows), m['verified_bytes'])

paths=set()
states={}
for job in ('729118_0','729118_1','729137_0','729137_1','729201_0','729202_0','729216_0','729216_1','728589_0'):
    states[job]=terminal(job,'COMPLETED|0:0')
for job in ('728930_0','728930_1'):
    state=subprocess.check_output(['sacct','-j',job,'-X','-n','-P','--format=JobID,State,ExitCode'],text=True)
    assert 'PREEMPTED' in state,state
    states[job]=state
p=N/'meta/native-contact-conformance-scaling-terminal-v1.json';p.write_text(json.dumps(states,indent=2)+'\n');paths.add(p)
for name in ('native-portal-point-b1-b32-v1','native-portal-point-combined-audit-v1','native-production-scaling128-512-v1','native-production-scaling128-512-audit-v1','native-independent-condition-banks-v1'):
    campaign(paths,name)
for name in ('native-production-canonical-pilot-v1','native-portal-point-b128-b512-v1'):
    campaign(paths,name,False)
paths.add(N/'native-portal-point-combined-v1/assembly.json')
for name in ('native-portal-point-b128-b512-v1-plan.json','native-portal-point-b128-b512-v1-tasks.json','native-portal-point-b128-b512-audit-v1-tasks.json','native-independent-condition-banks-v1-plan.json','native-independent-condition-banks-v1-tasks.json'):
    paths.add(N/'meta'/name)
for name in ('native-portal-point-combined-audit-v1','native-production-scaling128-512-audit-v1'):
    d=json.loads((N/name/'group0/audit/result.json').read_text())
    assert d.get('passed',d.get('timing_receipts_passed',False))
    for receipt in d['receipts']:
        p=Path(receipt['path'])
        if p.suffix in ('.json','.txt','.patch'):
            assert identity(p)==receipt
            paths.add(p)
    if name=='native-portal-point-combined-audit-v1':assert len(d['checks'])==349
for module,camp in (('production_throughput_audit.py','native-production-scaling128-512-audit-v1'),('audit_native_production_controls.py','native-portal-point-b128-b512-v1')):
    paths.add(N/camp/'analysis-source/research/rigid_stress/qipc_comparison'/module)
campaign(paths,'solid-pad-stvk-final125-reference-v2',True,[0])
campaign(paths,'solid-pad-stvk-final125-time-contact-v1',False)
paths.add(N/'solid-pad-stvk-final125-reference-v2/group0/reference/steps.jsonl')
reference=json.loads((N/'solid-pad-stvk-final125-reference-v2/group0/reference/result.json').read_text())
assert reference['completed'] and not reference['mechanical_history_passed']
assert not reference['history_checks']['actual_friction']
publish('native-contact-conformance-scaling-v1',paths,'Terminal349-check B1/B8/B32 repaired-source conformance, B128/B512 primary timing receipts and independent one-step condition-bank construction; immutable canonical-timing/larger-conformance inputs only. Larger motion/output conformance, total process GPU memory, diverse full histories and configuration-matched collider accuracy remain pending. Also retain the terminal125um two-wall reference whole-history friction rejection and immutable preempted time/contact inputs. Settled windows/current moment do not override the transient failure. Prior localization and static-control packets are unchanged. No full-goal completion claim.')
