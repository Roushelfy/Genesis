"""Freeze early completed125um surface comparisons without qualifying the full reference."""
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

accounting = subprocess.check_output(['sacct','-j','729479,730607,730619,730622','-P','--noheader','--format=JobID,State,ExitCode,Elapsed'],text=True)
observed = {p[0]:tuple(p[1:3]) for line in accounting.splitlines() if len(p := line.split('|')) >= 3}
for job in ['729479_0','729479_1','730607_0','730622_0']:
    assert observed[job] == ('COMPLETED','0:0'),(job,observed.get(job))
for job in ['730622_1','730622_2']:
    assert observed[job] == ('FAILED','1:0')
assert any(job.startswith('730619') and status[0].startswith('CANCELLED') for job,status in observed.items())
parent = json.loads((RUN/'surface2-pair-parent-admission-v1/group0/admission/result.json').read_text())
assert parent['passed'] and len(parent['runs']) == 2 and all(len(r['checks']) == 10 and all(r['checks'].values()) for r in parent['runs'])
positive = json.loads((RUN/'surface2-component-assessment-v2/group0/assessment/result.json').read_text())
assert positive['passed'] and len(positive['checks']) == 65 and all(positive['checks'].values()) and len(positive['rows']) == 2
assert not positive['fine_reference_qualified'] and positive['accepted_reference_frames'] == [] and not positive['whole_goal_complete']
for group,name in [(1,'negative-row-passed'),(2,'negative-empty-checks')]:
    negative = json.loads((RUN/'surface2-component-assessment-v2'/f'group{group}'/name/'result.json').read_text())
    assert not negative['passed'] and negative['error'] == 'exact_complete_parent_pair' and negative['rows'] == [] and not negative['fine_reference_qualified'] and negative['accepted_reference_frames'] == []
for receipt in positive['receipts']:
    assert identity(pathlib.Path(receipt['path'])) == receipt
contract = json.loads((META/'surface2-pair-scheduling-contract-v1.json').read_text())
for name,row in contract['immutable_capture_files'].items():
    current = identity(RUN/'stvk-canonical125-fields-v2'/name)
    assert current['bytes'] == row['bytes'] and current['sha256'] == row['sha256']
reader = json.loads((META/'surface2-component-reader-contract-v2.json').read_text())
assert reader['original_core_AST_unchanged']
assert identity(RUN/'surface2-component-assessment-v2/analysis-source/research/rigid_stress/qipc_comparison/friction_reference_gates.py')['sha256'] == reader['source_sha256']
proof = json.loads((META/'surface2-pair-release-verification-v1.json').read_text())
assert proof['selected_tasks_running'] and proof['other_six_tasks_original_dependency_unchanged'] and proof['all_captured_inputs_unchanged']
files = [SCRIPT] + [META/name for name in ['surface2-pair-parent-admission-v1-tasks.json','surface2-pair-scheduling-contract-v1.json','surface2-pair-dependency-release-v1.json','surface2-pair-release-verification-v1.json','surface2-component-reader-contract-v1.json','surface2-component-reader-contract-v2.json','surface2-component-assessment-v1-tasks.json','surface2-component-assessment-v2-tasks.json','negative-row-passed-v1.json','negative-empty-checks-v1.json']]
files += captured('surface2-pair-parent-admission-v1',['qipc_comparison/require_solid_pad_admission.py','qipc_comparison/artifacts.py']) + terminal('surface2-pair-parent-admission-v1')
for campaign in ['surface2-component-assessment-v1','surface2-component-assessment-v2']:
    files += captured(campaign,['qipc_comparison/friction_reference_gates.py','qipc_comparison/pad_normal_qualification.py','qipc_comparison/artifacts.py']) + terminal(campaign)
source = RUN/'stvk-canonical125-fields-v2/analysis-source'
manifest = {r['path']:r for r in json.loads((RUN/'stvk-canonical125-fields-v2/analysis-files.json').read_text())}
queue = ['research.rigid_stress.qipc_comparison.contact_fem_exact_columns']
modules = set()
while queue:
    module = queue.pop();rel=pathlib.Path(*module.split('.')).with_suffix('.py');path=source/rel
    if str(rel) in modules or not path.is_file():continue
    modules.add(str(rel));actual=identity(path);row=manifest[str(rel)];assert all(actual[k] == row[k] for k in ['bytes','sha256'])
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node,ast.ImportFrom):
            base='.'.join(module.split('.')[:-node.level]) if node.level else ''
            target='.'.join(p for p in [base,node.module] if p)
            if target.startswith('research.rigid_stress'):
                queue.append(target);queue.extend(target+'.'+name.name for name in node.names)
        elif isinstance(node,ast.Import):queue.extend(name.name for name in node.names if name.name.startswith('research.rigid_stress'))
files += captured('stvk-canonical125-fields-v2',[str(pathlib.Path(m).relative_to('research/rigid_stress')) for m in sorted(modules)])
external=[]
for group in [0,1]:
    root=RUN/'stvk-canonical125-fields-v2'/f'group{group}'
    files += [p for p in root.rglob('*') if p.is_file() and p.suffix in ['.json','.txt','.log'] and 'cache' not in p.parts and 'tmp' not in p.parts]
    files.append(RUN/'stvk-canonical125-fields-v2/logs'/f'729479_{group}.log')
    external += list(root.rglob('*.npz'))
terminal_receipt={'accounting':accounting,'pair_parent_checks':20,'component_checks':65,'component_passed':True,'rejected_input_controls':2,'full_reference_qualified':False,'accepted_reference_frames':[],'unchanged_six_other_task_dependencies':True,'immutable_original_field_capture_unchanged':True,'captured_field_research_import_cone':sorted(modules),'whole_goal_complete':False}
(META/'surface125-component-terminal-v1.json').write_text(json.dumps(terminal_receipt,indent=2)+'\n');files.append(META/'surface125-component-terminal-v1.json')
row=freeze('surface125-component-v1',files,external,'Actual250-to125um two-wall normal/friction surface comparisons at q9/q25/depth4 satisfy original2% complete-field/5% raw-and-common-peak and angular targets. Two terminal400-step parents pass20 strict admission checks; component730622_0 passes65 checks, and two invalid parent-receipt controls are rejected. Only original array tasks729479_0/_1 were released early after pair admission; original tasks/source/output identities, six remaining task dependencies and the full seven-parent/36-comparison qualifier are preserved. A transient cleared-dependency postcondition was verified against actual running handles without resubmission. The unused first component reader was cancelled before execution; scalar numerical gate AST remains unchanged. Four complete column arrays remain externally hashed on data. This component publishes no accepted fine-reference frames, uncertainty envelope, method error/rank or final friction figure; other three physical histories and remaining refinement/control components stay pending.')
(META/'surface125-component-publication-v1.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
