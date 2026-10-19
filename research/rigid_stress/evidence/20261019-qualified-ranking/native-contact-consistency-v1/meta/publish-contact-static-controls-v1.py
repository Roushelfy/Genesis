"""Freeze terminal contact-consistency and physical-pad control evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

N=Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
W=Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
E=W/'research/rigid_stress/evidence/20261019-qualified-ranking'

def identity(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}

def terminal(job,expected):
    result=subprocess.check_output(['sacct','-j',job,'-X','-n','-P','--format=JobID,State,ExitCode'],text=True)
    assert result.strip()==job+'|'+expected,result
    return result

def campaign(paths,name,done=True,groups=None):
    root=N/name
    for pattern in ('submission.json','source*.json','source*.patch','tasks.json','analysis-files.json','job.sh','workload-*.sh'):
        paths.update(p for p in root.glob(pattern) if p.is_file())
    if done:
        for group in sorted(root.glob('group*')):
            if not group.is_dir() or (groups is not None and int(group.name[5:]) not in groups):continue
            for pattern in ('returncode.txt','*.returncode.txt','*.log','*.json','**/result.json','**/start.json','**/constructed.json','**/restarted.json','**/step*.json','**/conformance.json','**/junit.xml'):
                paths.update(p for p in group.glob(pattern) if p.is_file())
        for p in (root/'logs').glob('*.log'):
            if groups is None or int(p.stem.split('_')[-1]) in groups:paths.add(p)

def publish(name,paths,scope):
    paths.add(Path(__file__))
    dest=E/name;dest.mkdir(exist_ok=False);rows=[]
    for source in sorted(paths):
        data=source.read_bytes();assert len(data)<3_000_000,str(source)
        compressed=len(data)>100000 or source.suffix=='.log'
        payload=gzip.compress(data,mtime=0) if compressed else data
        rel=str(source.relative_to(N))+('.gz' if compressed else '')
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
        assert (gzip.decompress(payload) if compressed else payload)==data
        rows.append({'source':str(source),'copy':rel,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'encoding':'gzip' if compressed else 'identity','original_bytes':len(data),'original_sha256':hashlib.sha256(data).hexdigest()})
    m={'files':rows,'verified_file_count':len(rows),'verified_bytes':sum(r['bytes'] for r in rows),'whole_goal_complete':False,'scope':scope}
    (dest/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (dest/'README.md').write_text('# '+name+'\n\n'+scope+'\n\nLarge JSON and raw logs use deterministic gzip; the manifest preserves original and encoded SHA-256/size. Decode a working copy with Python gzip before inspecting raw records. Full NPZ arrays/traces remain on data with exact identities. No live reference output is included.\n')
    print(name,len(rows),m['verified_bytes'])

states={}
for j in ['729031_0','729070_0',*(f'729033_{i}' for i in range(3)),*(f'729057_{i}' for i in (3,4,5)),*(f'729086_{i}' for i in range(5)),'729114_0']:
    states[j]=terminal(j,'COMPLETED|0:0')
for j in (0,1,2,6,7):states[f'729057_{j}']=terminal(f'729057_{j}','FAILED|1:0')
p=N/'meta/shell-pad-control-terminal-v1.json';p.write_text(json.dumps(states,indent=2)+'\n');paths={p}
d=json.loads((N/'physical-pad-shell250-control-audit-v1/group0/audit/result.json').read_text());assert d['passed'] and len(d['checks'])==151
for r in d['receipts']:
    assert identity(Path(r['path']))==r
    if Path(r['path']).suffix=='.json':paths.add(Path(r['path']))
for name in ('static-inertia-relief-controls-v1','column-external-volume-controls-v1','physical-pad-shell250-static-controls-v2','physical-pad-shell250-control-fields-v1','physical-pad-shell250-control-fields-v2','physical-pad-shell250-control-audit-v1'):campaign(paths,name)
for name in ('physical-pad-shell250-static-controls-v2-plan.json','physical-pad-shell250-control-fields-v1-plan.json','physical-pad-shell250-control-fields-v2-plan.json','physical-pad-shell250-control-audit-v1-preflight.json'):paths.add(N/'meta'/name)
for rel in ('static_inertia_relief.py','check_static_inertia_relief.py','finite_area_shell.py','shell_load_replay.py'):
    paths.add(N/'physical-pad-shell250-static-controls-v2/analysis-source/research/rigid_stress/qipc_comparison'/rel)
for rel in ('column_decomposition.py','contact_fem_exact_columns.py'):
    paths.add(N/'physical-pad-shell250-control-fields-v2/analysis-source/research/rigid_stress/qipc_comparison'/rel)
paths.add(N/'column-external-volume-controls-v1/analysis-source/research/rigid_stress/qipc_comparison/check_column_decomposition.py')
paths.add(N/'physical-pad-shell250-control-audit-v1/analysis-source/research/rigid_stress/qipc_comparison/audit_shell_pad_controls.py')
publish('shell-pad-load-separation-v1',paths,'Six admitted250um static Shell replay/own-pad controls, nine independent rigid-mass-projection controls, eight signed-tensor/volume controls, q9/q25 matched continuous/static/replay/solid comparisons and151 passed aggregation checks with56 rehashed receipts. Full fields/columns stay on data. Tiny explicit body corrections preserve actual point loads and the original zero-wrench/static gates. The solid carrier is the same-source250um centroid-pressure control; qualified125um normal errors remain separate. Original integration-guard failures remain retained.')

states={}
for j in ('729028_0','729032_0','729036_0','729040_0','729044_0','729048_0','729071_0','729090_0','729091_0','729100_0','729106_0'):states[j]=terminal(j,'COMPLETED|0:0')
for j in ('729022_0','729025_0','729027_0','729067_0'):states[j]=terminal(j,'FAILED|1:0')
p=N/'meta/native-contact-consistency-terminal-v1.json';p.write_text(json.dumps(states,indent=2)+'\n');paths={p}
d=json.loads((N/'native-portal-point-audit-v1/group0/audit/result.json').read_text());assert d['passed'] and len(d['checks'])==89
for r in d['receipts']:assert identity(Path(r['path']))==r
for name in ('native-construction-probe-v1','native-construction-probe-v2','native-construction-probe-v3','native-construction-arrays-v2','native-construction-arrays-v3','mpr-witness-position-v1','mpr-witness-normalization-v1','native-first-contact-state-v1','native-first-contact-comparison-v1','native-scr-changing-trace-v1','native-scr-changing-trace-summary-v1','native-scr-changing-trace-summary-v2','native-scr-workspace-tuning-v1','native-scr-workspace-audit-v1','native-portal-point-conformance-v1','native-portal-point-audit-v1','native-witness-focused-tests-v1'):campaign(paths,name)
for name in ('native-portal-point-b1-b32-v1','native-production-scaling128-512-v1'):campaign(paths,name,False)
for name in ('native-construction-probe-v1-cancel.json','native-construction-probe-v2-terminal.json','native-first-contact-state-v1-plan.json','mpr-witness-normalization-v1-plan.json','mpr-persistent-portal-points-v1-plan.json','native-scr-changing-trace-v1-terminal.json','native-scr-workspace-tuning-v1-plan.json','native-witness-documentation-ast-v1.json','native-witness-variant-source-v1.json','native-witness-promotion-v1.json','native-portal-point-audit-v1-preflight.json','native-portal-point-b1-b32-v1-plan.json','native-production-scaling128-512-v1-plan.json','native-witness-variant-v1.patch'):
    paths.add(N/'meta'/name)
for campaign_name,module in (('native-construction-probe-v3','native_production_controls.py'),('native-construction-arrays-v3','compare_physics_snapshots.py'),('mpr-witness-position-v1','check_mpr_witness_position.py'),('native-first-contact-comparison-v1','compare_contact_state_steps.py'),('native-scr-changing-trace-v1','native_recovery_trace.py'),('native-scr-changing-trace-summary-v2','summarize_native_trace.py'),('native-scr-workspace-audit-v1','audit_native_workspace_tuning.py'),('native-portal-point-audit-v1','audit_native_production_controls.py')):
    paths.add(N/campaign_name/'analysis-source/research/rigid_stress/qipc_comparison'/module)
publish('native-contact-consistency-v1',paths,'Initial346-array/configuration comparisons, first-divergence contact snapshots,384-portal MPR counterexamples, retained failed reciprocal candidate, persistent-portal-point B8 repair with89 passed strict checks and MPR/GJK lifecycle regressions. Trace metadata/parser failures remain explicit; complete raw trace is retained on data. Smaller-workspace numerical checks pass without an isolated speedup claim. B1/B32 conformance and B128/B512 production work contribute immutable inputs only. No physical-error or full-goal completion claim.')
