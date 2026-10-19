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

paths=set();states={}
for job in ('729194_0','729194_1','729194_2','729236_0'):
    states[job]=terminal(job,'COMPLETED|0:0')
p=N/'meta/native-canonical-production-terminal-v1.json';assert not p.exists();p.write_text(json.dumps(states,indent=2)+'\n');paths.add(p)
for name in ('native-production-canonical-pilot-v1','native-production-canonical-audit-v1'):campaign(paths,name)
for name in ('native-production-memory128-512-v1',):campaign(paths,name,False)
for p in (N/'meta').glob('native-production-memory-b*.json'):paths.add(p)
d=json.loads((N/'native-production-canonical-audit-v1/group0/audit/result.json').read_text());assert d['timing_receipts_passed'] and len(d['rows'])==15
for receipt in d['receipts']:
    p=Path(receipt['path'])
    if p.suffix in ('.json','.txt','.patch'):
        assert identity(p)==receipt;paths.add(p)
paths.add(N/'native-production-canonical-audit-v1/group0/audit/production.csv')
publish('native-canonical-production-v1',paths,'Fifteen terminal fixed-horizon native production timings on repaired engine1ff39b077, three warm1200-step repeats per mode/batch at B1/B8/B32, all runtime counters valid and858-file/source receipts rehashed. Paired complete motion/output conformance is the separate349-check contact packet. Few-collider nominal6mm source errors remain pending; no pad-error transfer, host full-output or policy-inference throughput claim. Separate total process-memory work contributes immutable inputs only. Full goal remains active.')
