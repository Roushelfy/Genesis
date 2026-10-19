"""Freeze the first terminal125um surface parent and its strict admission only."""
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

import ast

accounting = subprocess.check_output(['sacct','-j','729312,730592','-P','--noheader','--format=JobID,State,ExitCode,Elapsed'],text=True)
observed = {p[0]: tuple(p[1:3]) for line in accounting.splitlines() if len(p := line.split('|')) >= 3}
assert observed['729312_0'] == ('COMPLETED','0:0') and observed['730592_0'] == ('COMPLETED','0:0')
parent = RUN / 'solid-pad-stvk-canonical125-reference-v1'
run = parent / 'group0/reference'
result = json.loads((run / 'result.json').read_text())
admission = json.loads((RUN / 'canonical125-surface-parent-admission-v1/group0/admission/result.json').read_text())
moment = json.loads((parent / 'group0/moment/result.json').read_text())
assert result['completed'] and result['steps'] == 400 and result['mechanical_history_passed'] and not result['physical_reference_qualified']
assert admission['passed'] and len(admission['runs']) == 1 and len(admission['runs'][0]['checks']) == 10 and all(admission['runs'][0]['checks'].values())
assert moment['passed'] and moment['history_admitted']
for receipt in admission['receipts']:
    assert identity(pathlib.Path(receipt['path'])) == receipt
source = parent / 'analysis-source'
manifest = {row['path']:row for row in json.loads((parent/'analysis-files.json').read_text())}
queue = ['research.rigid_stress.qipc_comparison.'+name for name in ['plate_solid_pad','audit_solid_pad','check_matrix_free_pad','check_p2_stvk']]
modules = set()
while queue:
    module = queue.pop()
    rel = pathlib.Path(*module.split('.')).with_suffix('.py')
    path = source / rel
    if str(rel) in modules or not path.is_file():
        continue
    modules.add(str(rel))
    row = manifest[str(rel)]
    actual = identity(path)
    assert actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256']
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            base = '.'.join(module.split('.')[:-node.level]) if node.level else ''
            target = '.'.join(part for part in [base,node.module] if part)
            if target.startswith('research.rigid_stress'):
                queue.append(target)
                queue.extend(target+'.'+alias.name for alias in node.names)
        elif isinstance(node,ast.Import):
            queue.extend(alias.name for alias in node.names if alias.name.startswith('research.rigid_stress'))
files = [SCRIPT,META/'canonical125-surface-parent-admission-v1-tasks.json',META/'canonical125-surface-terminal-observation-v1.json']
files += captured('solid-pad-stvk-canonical125-reference-v1',[str(pathlib.Path(m).relative_to('research/rigid_stress')) for m in sorted(modules)])
files += [p for p in (parent/'group0').rglob('*') if p.is_file() and p.suffix in ['.json','.txt','.log'] and 'cache' not in p.parts and 'tmp' not in p.parts]
files.append(parent/'logs/729312_0.log')
files += captured('canonical125-surface-parent-admission-v1',['qipc_comparison/require_solid_pad_admission.py','qipc_comparison/artifacts.py']) + terminal('canonical125-surface-parent-admission-v1')
start = json.loads((run/'start.json').read_text())
external = list((parent/'group0').rglob('*.npz'))
if isinstance(start.get('mesh'),dict):
    external.append(pathlib.Path(start['mesh']['path']))
metadata = {'accounting':accounting,'steps':400,'strict_checks':10,'physics_research_modules':sorted(modules),'mechanical_history_passed':True,'independent_momentum_passed':True,'physical_reference_qualified':False,'other_three_fine_writers_excluded':True,'whole_goal_complete':False}
(META/'canonical125-surface-terminal-publication-receipt-v1.json').write_text(json.dumps(metadata,indent=2)+'\n')
files.append(META/'canonical125-surface-terminal-publication-receipt-v1.json')
row = freeze('canonical125-surface-terminal-v1',files,list(set(external)),'First repaired125um/two-wall/q10 physical reference completes all400 steps, original six every-step law/equation/boundary/current-moment checks and both51-sample normal/friction equilibrium windows. Own independent current momentum and exact endpoint transfer pass. Strict preflight730592 passes all10 complete-journal/result/momentum/two-field receipt checks. Actual terminal Slurm states, original coupled driver and its complete captured research import cone remain traceable. Full endpoints/mesh remain on data with hashes. This is mechanical parent admission only: no spatial/wall/time/contact qualification, reference uncertainty, final method error/rank or final friction figure is inferred. The other three fine writers, original pending DAG, old failed125um baselines and numerical/physical targets remain unchanged; all live group1 outputs are excluded.')
(META/'canonical125-surface-terminal-publication-v1.json').write_text(json.dumps(row,indent=2)+'\n')
print(json.dumps(row))
