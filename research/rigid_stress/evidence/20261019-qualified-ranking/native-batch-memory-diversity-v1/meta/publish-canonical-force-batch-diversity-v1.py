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
for job in ('729265_0','729267_0','729344_0'):
    states[job]=terminal(job,'COMPLETED|0:0')
for job in ('729250_0','729252_0','729255_0','729263_0'):
    states[job]=terminal(job,'FAILED|1:0')
p=N/'meta/modal-canonical-force-terminal-v1.json';assert not p.exists();p.write_text(json.dumps(states,indent=2)+'\n');paths={p}
for name in ('modal-physical-force-before-v1','modal-physical-force-after-v1','modal-physical-force-after-v2','modal-physical-force-after-v3','modal-physical-force-after-v4','solid-pad-stvk-physical-force-coarse-v1','modal-physical-force-audit-v1'):campaign(paths,name)
for name in ('solid-pad-stvk-canonical125-reference-v1','solid-pad-stvk-canonical125-time-contact-v1'):campaign(paths,name,False)
for name in ('modal-physical-force-variant-v1-plan.json','solid-pad-stvk-physical-force-coarse-v1-plan.json','solid-pad-stvk-canonical125-reference-v1-plan.json','solid-pad-stvk-canonical125-time-contact-v1-plan.json','stvk-canonical125-fields-v1-tasks.json','stvk-canonical125-time-contact-fields-v1-tasks.json','stvk-canonical-friction-method-fields-v1-tasks.json'):
    paths.add(N/'meta'/name)
d=json.loads((N/'modal-physical-force-audit-v1/group0/audit/result.json').read_text());assert d['passed'] and len(d['checks'])==28
for receipt in d['receipts']:
    p=Path(receipt['path'])
    if p.suffix=='.json':
        assert identity(p)==receipt;paths.add(p)
for camp,modules in (
    ('modal-physical-force-before-v1',('modal_pad_dynamics.py','check_modal_physical_force.py')),
    ('modal-physical-force-after-v1',('modal_pad_dynamics.py','check_matrix_free_pad.py')),
    ('modal-physical-force-after-v2',('check_matrix_free_pad.py',)),
    ('modal-physical-force-after-v3',('check_matrix_free_pad.py',)),
    ('modal-physical-force-after-v4',('modal_pad_dynamics.py','check_modal_physical_force.py','check_matrix_free_pad.py')),
    ('solid-pad-stvk-physical-force-coarse-v1',('plate_solid_pad.py','solid_pad_step.py','modal_pad_dynamics.py','condensed_pad_dynamics.py')),
    ('modal-physical-force-audit-v1',('audit_modal_physical_force.py',))):
    for module in modules:paths.add(N/camp/'analysis-source/research/rigid_stress/qipc_comparison'/module)
publish('modal-canonical-force-v1',paths,'Original near-detachment force/velocity mismatch,12 repaired analytic detachment cases,24 contact controls, unchanged400-step physical/momentum admission and q9/depth4 same-configuration fields under the predeclared1e-8 target. Aggregate28 checks pass; earlier test/CPU-metadata failures remain retained. Canonical physical impulse is evaluated at returned full velocity under the original momentum budget; no source/friction/material/acceptance tolerance changes. Fine125um reference/time/contact work contributes immutable inputs only. Full arrays remain on data and fine qualification/full-goal completion remain open.')

states={}
for job in ('729223_0','729223_1','729241_0','729228_0','729228_1','729260_0','729281_0','729281_1','729323_0','729323_1','729351_0','729351_1','729357_0','729357_1'):
    states[job]=terminal(job,'COMPLETED|0:0')
for job in ('729247_0','729247_1','729278_0','729278_1','729300_0','729300_1','729317_0','729317_1','729336_0','729336_1','729342_0','729342_1','729346_0','729346_1','729323_2','729357_2'):
    states[job]=terminal(job,'FAILED|1:0')
p=N/'meta/native-batch-memory-diversity-terminal-v1.json';assert not p.exists();p.write_text(json.dumps(states,indent=2)+'\n');paths={p}
for name in ('native-portal-point-b128-b512-v1','native-portal-point-b128-b512-audit-v1','native-production-memory128-512-v1','native-production-memory128-512-audit-v1','native-diverse-conformance128-512-v1','native-diverse-dpr-audit-v1','native-diverse-dpr-audit-v2','native-diverse-rejection-capture-v1','native-diverse-rejection-capture-v2','native-diverse-rejection-audit-v1','native-diverse-rejection-audit-v2','native-diverse-rejection-audit-v3','native-diverse-rejection-audit-v4','native-diverse-rejection-audit-v5','native-rejection-reader-relocation-v1'):campaign(paths,name)
for name in ('native-production-scaling2048-8192-v1','native-portal-point-b2048-b8192-v1','native-production-memory2048-8192-v1'):campaign(paths,name,False)
for i in (0,1):
    paths.add(N/f'native-diverse-rejection-capture-v2/group{i}/scr-max/rejected-state.npz')
    paths.update((N/f'native-diverse-rejection-capture-v2/group{i}/scr-max').glob('rejected-step.json'))
    paths.add(N/f'native-independent-condition-banks-v1/group{i}/conditions.npz')
for name in ('native-diverse-rejection-capture-v1','native-diverse-rejection-capture-v2','native-diverse-rejection-audit-v4','native-diverse-rejection-audit-v5'):
    paths.update((N/name).glob('group*/**/rejected-step.json'))
    paths.update((N/name).glob('group*/**/observed.json'))
for name in ('native-production-memory128-512-audit-v1','native-portal-point-b128-b512-audit-v1'):
    result=N/name/'group0/audit/result.json';d=json.loads(result.read_text());assert d['passed']
    for receipt in d['receipts']:
        p=Path(receipt['path'])
        if p.suffix=='.json' and p.stat().st_size<1_000_000:
            assert identity(p)==receipt;paths.add(p)
paths.add(N/'native-production-memory128-512-audit-v1/group0/audit/memory.csv')
for name in ('rejected-state.npz','rejected-step.json'):
    paths.add(N/'meta/native-rejection-reader-fixtures-v1/bad0'/name)
for camp,modules in (
    ('native-production-memory128-512-v1',('native_production_memory.py',)),
    ('native-production-memory128-512-audit-v1',('audit_native_production_memory.py',)),
    ('native-diverse-dpr-audit-v2',('audit_native_production_controls.py',)),
    ('native-diverse-rejection-capture-v2',('native_production_controls.py',)),
    ('native-diverse-rejection-audit-v1',('audit_native_rejection.py',)),
    ('native-diverse-rejection-audit-v5',('audit_native_rejection.py',)),
    ('native-rejection-reader-relocation-v1',('audit_native_rejection.py',))):
    for module in modules:paths.add(N/camp/'analysis-source/research/rigid_stress/qipc_comparison'/module)
publish('native-batch-memory-diversity-v1',paths,'Terminal159-check B128/B512 complete conformance plus10 separately source-matched NVIDIA process-memory profiles. Diverse rigid/DPR max/full cohorts retain every environment and pass64 checks, with a repeated-bank diversity negative control. Diverse SCR max/full original failures and timely first-rejection captures are retained. Independent full-wrench certificates verify all captured failed environments on conservative continuous supports, with portable replay and a corrupted-byte negative control; earlier delayed-capture/reader failures remain explicit. Larger scaling contributes immutable inputs only. No general SCR stability, configuration-mismatched pad error, policy throughput or full-goal completion claim.')
