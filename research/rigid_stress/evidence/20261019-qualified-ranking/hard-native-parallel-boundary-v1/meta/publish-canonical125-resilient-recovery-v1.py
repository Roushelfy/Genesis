"""Freeze actual preemptions, completed periodic-state controls and the explicitly rebound reference consumers."""
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
        compressed = bool(original) and (source.suffix in ('.log', '.jsonl') or (source.suffix == '.json' and len(original) > 8192) or source.suffix == '.patch')
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
import datetime

read=lambda p:json.loads(p.read_text())
physical=read(META/'canonical125-separated-resume-launch-v1.json')
dag=read(META/'canonical125-resilient-dag-launch-v1.json')
verified=read(META/'canonical125-resilient-recovery-verified-v1.json')
assert verified['passed'] and len(verified['checks'])==55 and all(verified['checks'].values())
control=read(RUN/'solid-pad-periodic-state-reinspection-v1/group0/admission/result.json')
assert control['passed'] and len(control['checks'])==99 and all(control['checks'].values())
for receipt in control['receipts']:assert identity(pathlib.Path(receipt['path']))==receipt
assert (RUN/'solid-pad-periodic-checkpoint-control-v1/group0/returncode.txt').read_text().strip()=='1'
accounting=subprocess.check_output(['sacct','--array','-j','729312,729314,730868,730903,730913,730941,730971,730990,730978,730979,730980','-P','--noheader','--format=JobID,State,ExitCode,Elapsed,End'],text=True)
observed={v[0]:(v[1].split()[0],v[2]) for line in accounting.splitlines() if len(v:=line.split('|'))>=3}
for job in ['729312_1','729314_0','729314_1','730903_0']:assert observed[job][0]=='PREEMPTED',(job,observed.get(job))
assert observed['730868_0'][0].startswith('CANCELLED')
assert observed['730913_0']==('COMPLETED','0:0') and observed['730941_0']==('FAILED','1:0')
assert observed['730971_0']==('FAILED','1:0') and observed['730990_0']==('COMPLETED','0:0')
for i in range(1,5):assert observed[f'730971_{i}']==('COMPLETED','0:0')
states={role:subprocess.check_output(['scontrol','show','job',job+'_0','-o'],text=True) for role,job in physical['jobs'].items()}
assert all('JobState=RUNNING' in state or 'JobState=PENDING' in state for state in states.values())
proofs=[RUN/name/'result.json' for name in ['solid-pad-stvk-canonical125-final-resume-preparation-v1','solid-pad-stvk-canonical125-contact-resume-preparation-v1','solid-pad-stvk-canonical125-halfdt-resume-preparation-v1']]
for path,count in zip(proofs,[200,200,400]):
    proof=read(path);assert proof['passed'] and proof['prefix_steps']==count and proof['terminal_scheduler_state']=='PREEMPTED'
    for row in proof['receipts']:assert identity(pathlib.Path(row['path']))==row
assert not (RUN/'solid-pad-stvk-canonical125-final-resume-v1/group0/reference/start.json').exists()
assert not (RUN/'solid-pad-stvk-canonical125-final-resume-v1/group0/reference/steps.jsonl').exists()
assert not (RUN/'solid-pad-stvk-canonical125-final-resume-v2/group0/reference/start.json').exists()

files=[SCRIPT,*proofs]
meta_names=['canonical125-final-preemption-resume-contract-v1.json','canonical125-final-resume-v1-launch-correction.json','solid-pad-stvk-canonical125-final-resume-v1-tasks.json','solid-pad-stvk-canonical125-final-resume-v2-tasks.json','solid-pad-resume-lineage-controls-v1-tasks.json','canonical125-recovered-parent-overrides-v1.json','canonical125-recovered-dag-contract-v1.json','canonical125-recovered-dag-events-v1.jsonl','canonical125-recovered-dag-launch-v1.json','canonical125-recovered-dag-v1-prelaunch-observation.json','canonical125-preemption-recovery-verified-v1.json','rebuild-canonical125-resumed-dag-v1.py','rebuild-canonical125-resumed-dag-v2.py','publish-canonical125-preemption-resume-v1.py','halfdt-normal-keyframe-observed-v1.json','periodic-checkpoint-preemption-contract-v1.json','periodic-state-reader-recovery-v1.json','solid-pad-periodic-checkpoint-control-v1-tasks.json','solid-pad-periodic-state-reinspection-v1-tasks.json','submit-canonical125-separated-resumes-v1.py','canonical125-separated-resume-contract-v1.json','canonical125-separated-resume-launch-v1.json','canonical125-separated-resume-events-v1.jsonl','canonical125-separated-resume-guard-rebinding-v1.json','rebuild-canonical125-resilient-dag-v1.py','canonical125-resilient-parent-overrides-v1.json','canonical125-resilient-dag-contract-v1.json','canonical125-resilient-dag-events-v1.jsonl','canonical125-resilient-dag-launch-v1.json','canonical125-resilient-recovery-verified-v1.json','periodic-reference-arithmetic-equivalence-v1.json']
for contract_name,key in [('canonical125-recovered-dag-contract-v1.json','new_task_plans'),('canonical125-resilient-dag-contract-v1.json','new_task_files')]:meta_names += [pathlib.Path(path).name for path in read(META/contract_name)[key].values()]
meta_names += [pathlib.Path(plan['tasks']).name for plan in physical['plans']]
files += [META/name for name in meta_names]
files += [pathlib.Path(read(META/name)['assembly']) for name in ['canonical125-recovered-dag-launch-v1.json','canonical125-resilient-dag-launch-v1.json']]
cones={}
def cone(campaign):
    root=RUN/campaign;source=root/'analysis-source';inventory={row['path']:row for row in read(root/'analysis-files.json')}
    queue=[task['module'] for group in read(root/'tasks.json') for task in group];seen=set()
    while queue:
        module=queue.pop();relative=pathlib.Path(*module.split('.')).with_suffix('.py');path=source/relative
        if str(relative) in seen or not path.is_file():continue
        seen.add(str(relative));actual=identity(path);row=inventory[str(relative)];assert all(actual[key]==row[key] for key in ['bytes','sha256'])
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom):
                base='.'.join(module.split('.')[:-node.level]) if node.level else ''
                target='.'.join(part for part in [base,node.module] if part)
                if target.startswith('research.rigid_stress'):
                    queue.append(target);queue.extend(target+'.'+name.name for name in node.names)
            elif isinstance(node,ast.Import):queue.extend(name.name for name in node.names if name.name.startswith('research.rigid_stress'))
    cones[campaign]=sorted(seen)
    return captured(campaign,[str(pathlib.Path(path).relative_to('research/rigid_stress')) for path in sorted(seen)])
campaigns=['solid-pad-stvk-canonical125-reference-v1','solid-pad-stvk-canonical125-time-contact-v1','solid-pad-stvk-canonical125-final-resume-v1','solid-pad-stvk-canonical125-final-resume-v2','solid-pad-resume-lineage-controls-v1','recovered-figure-admission-negative-v1','solid-pad-periodic-checkpoint-control-v1','solid-pad-periodic-state-reinspection-v1']
campaigns += [plan['campaign'] for plan in physical['plans']]
for launch_name in ['canonical125-recovered-dag-launch-v1.json','canonical125-resilient-dag-launch-v1.json']:
    capture=read(META/launch_name)
    naming=capture.get('campaign_names',read(META/'canonical125-recovered-dag-contract-v1.json')['new_campaign_names'])
    campaigns += [naming[role] for role in capture['jobs']]
for campaign in sorted(set(campaigns)):files += cone(campaign)
for campaign,group in [('solid-pad-stvk-canonical125-reference-v1',1),('solid-pad-stvk-canonical125-time-contact-v1',0),('solid-pad-stvk-canonical125-time-contact-v1',1)]:
    root=RUN/campaign/f'group{group}'
    files += [p for p in root.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.txt','.log'] and 'cache' not in p.parts and 'tmp' not in p.parts]
    original_job='729312' if 'reference-v1' in campaign else '729314'
    files.append(RUN/campaign/'logs'/f'{original_job}_{group}.log')
for campaign in ['solid-pad-stvk-canonical125-final-resume-v1','solid-pad-stvk-canonical125-final-resume-v2','solid-pad-resume-lineage-controls-v1','recovered-figure-admission-negative-v1','solid-pad-periodic-checkpoint-control-v1','solid-pad-periodic-state-reinspection-v1']:files += terminal(campaign)
files.append(RUN/'solid-pad-periodic-checkpoint-control-v1/group0/reference/steps.jsonl')
external={pathlib.Path(row['path']) for row in control['receipts'] if row['path'].endswith('.npz')}
for path in proofs:
    proof=read(path);external.update(pathlib.Path(row['path']) for row in proof['receipts'] if row['path'].endswith('.npz'))
    external.add(pathlib.Path(read(pathlib.Path(proof['run'])/'start.json')['mesh']['path']))
receipt={'recorded_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accounting':accounting,'actual_new_physical_states':states,'physical_jobs':physical['jobs'],'consumer_jobs':dag['jobs'],'original_journal_counts':[321,342,418],'inherited_steps':[200,200,400],'periodic_stride':25,'actual_coarse_equivalence_checks':33,'actual_fine_prefix_input_checks':45,'actual_lineage_controls':26,'actual_independent_state_and_receipt_checks':99,'actual_wiring_and_placement_checks':55,'captured_import_cones':cones,'old_wrapper_JSON_failure_preserved':True,'old_captured_sources_and_journals_not_rewritten':True,'live_scientific_outputs_excluded':True,'fine_reference_qualified':False,'accepted_reference_frames':[],'whole_goal_complete':False}
p=META/'canonical125-resilient-recovery-terminal-v1.json';assert not p.exists();p.write_text(json.dumps(receipt,indent=2)+'\n');files.append(p)
row=freeze('canonical125-resilient-recovery-v1',files,sorted(external),'Actual node-wide preemptions retain original final/contact/half-dt prefixes321/342/418 and their full normal endpoints. Terminal-only preparations preserve200/200/400 physical steps. The first corrected initialization was also preempted; an earlier in-group terminal-check launch was cancelled before integration. A periodic-state output option is validated by33 actual coarse complete-history/array/common-field comparisons; three fine prefix controls pass45 checks, lineage passes26 and independent state/control reinspection passes99. The first periodic reader failed only at NumPy-bool JSON encoding; its failed wrapper/log/source remains intact and only read-only reinspection reruns. New CPU continuations730978/730979/730980 use disjoint recorded node pools, exclude the repeatedly preempted node and save physical states every25 steps. Current-state placement/source/CLI/graph checks pass55. Explicit aliases bind all three continued parents and retain every original physical/resolution gate; two completed surface2 groups are reused via relative data-side links. The complete seven-parent/36-comparison/18-envelope,54-error/18-rank, figure and matching-cost chain is requeued. No live fine output, continued-parent admission, final fine qualification, method rank or final artwork is frozen or admitted here. Periodic-state packing is validated; non-endpoint restoration requires its own subsequent provenance/trajectory control.')
(META/'canonical125-resilient-recovery-publication-v1.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
