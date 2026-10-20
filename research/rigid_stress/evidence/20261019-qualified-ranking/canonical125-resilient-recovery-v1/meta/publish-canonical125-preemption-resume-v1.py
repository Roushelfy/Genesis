"""Freeze terminal fine preemption, validated continuation lineage and explicit replacement consumers."""
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

contract=json.loads((META/'canonical125-recovered-dag-contract-v1.json').read_text())
launch=json.loads((META/'canonical125-recovered-dag-launch-v1.json').read_text())
verified=json.loads((META/'canonical125-preemption-recovery-verified-v1.json').read_text())
assert verified['passed'] and len(verified['checks'])==28 and all(verified['checks'].values())
accounting=subprocess.check_output(['sacct','--array','-j','729312,729314,730868,730903,730913,730941,729487,729478,730699,729479,729480,729513,729539,729519,729835,729938,730062','-P','--noheader','--format=JobID,State,ExitCode,Elapsed,End'],text=True)
observed={v[0]:(v[1].split()[0],v[2]) for line in accounting.splitlines() if len(v:=line.split('|'))>=3}
assert observed['729312_1'][0]=='PREEMPTED'
assert observed['730868_0'][0].startswith('CANCELLED')
assert observed['730913_0']==('COMPLETED','0:0') and observed['730941_0']==('FAILED','1:0')
assert observed['729479_0']==('COMPLETED','0:0') and observed['729479_1']==('COMPLETED','0:0')
for job in ['729487_0','729478_0','730699_0',*[f'729479_{i}' for i in range(2,8)],*[f'729480_{i}' for i in range(4)],'729513_0',*[f'729539_{i}' for i in range(6)],'729519_0','729835_0','729938_0','730062_0']:
    assert observed[job][0].startswith('CANCELLED'),(job,observed.get(job))
live={job:subprocess.check_output(['scontrol','show','job',job,'-o'],text=True) for job in ['730903_0','729314_0','729314_1']}
assert all('JobState=RUNNING' in value for value in live.values())
original=RUN/'solid-pad-stvk-canonical125-reference-v1/group1/reference'
assert len((original/'steps.jsonl').read_text().splitlines())==321
assert not (RUN/'solid-pad-stvk-canonical125-final-resume-v1/group0/reference/start.json').exists()
assert not (RUN/'solid-pad-stvk-canonical125-final-resume-v1/group0/reference/steps.jsonl').exists()
proof=json.loads((RUN/'solid-pad-stvk-canonical125-final-resume-preparation-v1/result.json').read_text())
assert proof['passed'] and proof['prefix_steps']==200 and proof['terminal_scheduler_state']=='PREEMPTED'
for row in proof['receipts']:assert identity(pathlib.Path(row['path']))==row
for path,sha in contract['original_capture_hashes'].items():assert identity(pathlib.Path(path))['sha256']==sha
controls=json.loads((RUN/'solid-pad-resume-lineage-controls-v1/group0/controls/result.json').read_text())
assert controls['passed'] and len(controls['checks'])==26 and all(controls['checks'].values())
negative=json.loads((RUN/'recovered-figure-admission-negative-v1/group0/figures/result.json').read_text())
assert not negative['passed'] and negative['error']=='explicit_fine_friction_reference_qualified' and negative['figures']=={}

files=[SCRIPT]
meta_names=['canonical125-final-preemption-resume-contract-v1.json','canonical125-final-resume-v1-launch-correction.json','solid-pad-stvk-canonical125-final-resume-v1-tasks.json','solid-pad-stvk-canonical125-final-resume-v2-tasks.json','solid-pad-resume-lineage-controls-v1-tasks.json','canonical125-recovered-parent-overrides-v1.json','canonical125-recovered-dag-contract-v1.json','canonical125-recovered-dag-events-v1.jsonl','canonical125-recovered-dag-launch-v1.json','canonical125-recovered-dag-v1-prelaunch-observation.json','canonical125-preemption-recovery-verified-v1.json','rebuild-canonical125-resumed-dag-v1.py','rebuild-canonical125-resumed-dag-v2.py','watch-canonical125-recovered-v1.py']
meta_names += [pathlib.Path(path).name for path in contract['new_task_plans'].values()]
files += [META/name for name in meta_names]
files.append(RUN/launch['assembly'].removeprefix(str(RUN)+'/'))

cones={}
def cone(campaign):
    folder=RUN/campaign;source=folder/'analysis-source'
    manifest={row['path']:row for row in json.loads((folder/'analysis-files.json').read_text())}
    groups=json.loads((folder/'tasks.json').read_text())
    queue=[task['module'] for group in groups for task in group];seen=set()
    while queue:
        module=queue.pop();relative=pathlib.Path(*module.split('.')).with_suffix('.py');path=source/relative
        if str(relative) in seen or not path.is_file():continue
        seen.add(str(relative));actual=identity(path);row=manifest[str(relative)]
        assert all(actual[key]==row[key] for key in ['bytes','sha256'])
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom):
                base='.'.join(module.split('.')[:-node.level]) if node.level else ''
                target='.'.join(p for p in [base,node.module] if p)
                if target.startswith('research.rigid_stress'):
                    queue.append(target);queue.extend(target+'.'+name.name for name in node.names)
            elif isinstance(node,ast.Import):queue.extend(name.name for name in node.names if name.name.startswith('research.rigid_stress'))
    cones[campaign]=sorted(seen)
    return captured(campaign,[str(pathlib.Path(path).relative_to('research/rigid_stress')) for path in sorted(seen)])

campaigns=['solid-pad-stvk-canonical125-reference-v1','solid-pad-stvk-canonical125-final-resume-v1','solid-pad-stvk-canonical125-final-resume-v2','solid-pad-resume-lineage-controls-v1',*[value for key,value in contract['new_campaign_names'].items() if key!='spatial_view']]
for campaign in campaigns:files += cone(campaign)
files += [path for path in original.parent.rglob('*') if path.is_file() and path.suffix in ('.json','.jsonl','.txt','.log') and 'cache' not in path.parts and 'tmp' not in path.parts]
files.append(RUN/'solid-pad-stvk-canonical125-reference-v1/logs/729312_1.log')
for campaign in ['solid-pad-stvk-canonical125-final-resume-v1','solid-pad-resume-lineage-controls-v1','recovered-figure-admission-negative-v1']:files+=terminal(campaign)
files.append(RUN/'solid-pad-stvk-canonical125-final-resume-preparation-v1/result.json')
for campaign in ['canonical125-spatial-parent-admission-v1','canonical125-parent-admission-v1','contact125-pair-parent-admission-v1','stvk-canonical125-fields-v2','stvk-canonical125-time-contact-fields-v2','canonical-friction-reference-qualification-v1','canonical-stvk-method-fields-v2','canonical-friction-plot-data-v1','canonical-stvk-method-ranking-v1','canonical-stvk-figures-v1','pad-friction-cost-accuracy-v1']:
    files += [RUN/campaign/name for name in ['tasks.json','submission.json']]
external={pathlib.Path(row['path']) for row in proof['receipts'] if row['path'].endswith('.npz')}
external.update(pathlib.Path(row['path']) for row in controls['receipts'] if row['path'].endswith('.npz'))
external.add(pathlib.Path(json.loads((original/'start.json').read_text())['mesh']['path']))
receipt={'recorded_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accounting':accounting,'live_reference_jobs':live,'original_complete_records':321,'inherited_normal_prefix_steps':200,'new_steps_required':200,'actual_fine_input_controls':15,'continuation_lineage_controls':26,'replacement_wiring_and_source_checks':28,'new_figure_reader_rejects_normal_only_reference':True,'captured_research_import_cones':cones,'all_original_captures_and_scientific_targets_preserved':True,'no_live_scientific_outputs_frozen':True,'fine_reference_qualified':False,'accepted_reference_frames':[],'whole_goal_complete':False}
p=META/'canonical125-final-preemption-terminal-v1.json';assert not p.exists();p.write_text(json.dumps(receipt,indent=2)+'\n');files.append(p)
row=freeze('canonical125-final-preemption-resume-v1',files,sorted(external),'Actual final125um/three-wall q10 parent729312_1 is PREEMPTED after321 complete records. Original journal, normal endpoint and failure records remain intact. Terminal-only preparation binds the complete200-step normal state and all ten numerical modules. Fine input controls pass15; corrected CPU-only continuation730903 uses the unchanged validated driver/state restoration and new data outputs. Initial730868 launch was cancelled before any integration to remove an invalid in-group terminal check; its inputs and15 passing controls remain preserved. New continued-parent lineage passes26 actual/negative controls, recovered source/wiring passes28 checks, and the figure entry with an explicit fields campaign rejects a normal-only reference. Two already completed surface2 groups are reused through a recorded relative data-side view. Only obsolete never-started consumers are cancelled; the explicit replacement retains seven physical histories,36 comparisons,18 envelopes,54 method-error rows/18 per-metric ranks, final figures and matched costs. Other two original fine writers and the active continuation remain live; only their immutable captures are frozen. No continued fine parent, full reference, final rank or final artwork is admitted here.')
(META/'canonical125-final-preemption-publication-v1.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
