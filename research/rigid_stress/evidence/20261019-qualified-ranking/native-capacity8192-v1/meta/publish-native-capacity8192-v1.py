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
    dest.mkdir(exist_ok=False)
    rows = []
    for source in sorted(paths):
        data = source.read_bytes()
        assert len(data) < 32000000, str(source)
        compressed = len(data) > 100000 or source.suffix == '.log' or (source.suffix == '.patch' and bool(data))
        payload = gzip.compress(data, mtime=0) if compressed else data
        rel = str(source.relative_to(N)) + ('.gz' if compressed else '')
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        assert (gzip.decompress(payload) if compressed else payload) == data
        rows.append({'source': str(source), 'copy': rel, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(), 'encoding': 'gzip' if compressed else 'identity', 'original_bytes': len(data), 'original_sha256': hashlib.sha256(data).hexdigest()})
    m = {'files': rows, 'verified_file_count': len(rows), 'verified_bytes': sum((r['bytes'] for r in rows)), 'whole_goal_complete': False, 'scope': scope}
    (dest / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    (dest / 'README.md').write_text('# ' + name + '\n\n' + scope + '\n\nExact nonempty patches, large JSON and raw logs use deterministic gzip; the manifest preserves original and encoded SHA-256/size. Decode a working copy with Python gzip before inspecting raw records. Full NPZ arrays/traces remain on data with exact identities. No live reference output is included.\n')
    print(name, len(rows), m['verified_bytes'])

states={}
for job in ('729261_0','729261_1','729291_0','729291_1','729348_0','729349_0','729349_1','729371_0','729409_0'):
    states[job]=terminal(job,'COMPLETED|0:0')
p=N/'meta/native-capacity8192-terminal-v1.json';assert not p.exists();p.write_text(json.dumps(states,indent=2)+'\n');paths={p}
campaign(paths,'native-production-scaling2048-8192-v1',False)
for name in ('native-production-scaling2048-8192-audit-v1','native-portal-point-b2048-b8192-v1','native-portal-point-b2048-b8192-audit-v1','native-production-memory2048-8192-v1','native-production-memory2048-8192-audit-v1'):campaign(paths,name)
for name in ('native-production-scaling2048-8192-v1-plan.json','native-portal-point-b2048-b8192-v1-plan.json'):
    paths.add(N/'meta'/name)
for name in ('native-production-scaling2048-8192-audit-v1','native-portal-point-b2048-b8192-audit-v1','native-production-memory2048-8192-audit-v1'):
    p=N/name/'group0/audit/result.json';d=json.loads(p.read_text());assert d.get('passed',d.get('timing_receipts_passed',False))
    for receipt in d['receipts']:
        p=Path(receipt['path'])
        if p.suffix=='.json' and p.stat().st_size<1_000_000:
            assert identity(p)==receipt;paths.add(p)
paths.add(N/'native-production-scaling2048-8192-audit-v1/group0/audit/production.csv')
paths.add(N/'native-production-memory2048-8192-audit-v1/group0/audit/memory.csv')
for camp,module in (('native-production-scaling2048-8192-audit-v1','production_throughput_audit.py'),('native-portal-point-b2048-b8192-audit-v1','audit_native_production_controls.py'),('native-production-memory2048-8192-audit-v1','audit_native_production_memory.py')):
    paths.add(N/camp/'analysis-source/research/rigid_stress/qipc_comparison'/module)
publish('native-capacity8192-v1',paths,'Terminal five-mode B2048/B8192 primary timing receipts,159 complete motion/output/batch checks and10 separate source/configuration-matched sampled NVIDIA process-memory profiles. Every timed environment is retained, runtime invalid counters are zero, full output stays on GPU and primary windows exclude intrusive arrays/audits/memory sampling. Source858 files are rehashed; full trajectories and large raw benchmark JSON remain on data with exact IDs. These repeat the preexisting eight-condition bank; no universal diverse SCR stability, physical collider accuracy, policy-inference or hardware maximum-capacity claim. Full goal remains active.')
