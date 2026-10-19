"""Freeze terminal evidence and immutable new-job inputs without copying full fields."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
W = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
E = W / 'research/rigid_stress/evidence/20261019-qualified-ranking'

def identity(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}

def terminal(job):
    state = subprocess.check_output(['sacct','-j',job,'-X','-n','-P','--format=JobID,State,ExitCode'],text=True)
    assert state.strip() == job+'|COMPLETED|0:0', state
    return state

def campaign(paths, name, complete=False):
    root = N / name
    patterns = ['submission.json','tasks.json','source*.json','source*.patch','analysis-files.json','job.sh','workload-*.sh']
    if complete:
        patterns += ['group*/returncode.txt','group*/*.log','group*/**/result.json','group*/**/start.json','group*/**/*.csv','logs/*.log']
    for pattern in patterns:
        paths.update(p for p in root.glob(pattern) if p.is_file())

def publish(name, paths, scope):
    paths.add(Path(__file__))
    dest = E / name
    dest.mkdir(exist_ok=False)
    rows=[]
    for source in sorted(paths):
        assert source.is_file() and source.stat().st_size < 1_000_000, str(source)
        target = dest / source.relative_to(N)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        a,b=identity(source),identity(target)
        assert a['bytes']==b['bytes'] and a['sha256']==b['sha256']
        rows.append({'source':str(source),'copy':str(target.relative_to(dest)),'bytes':a['bytes'],'sha256':a['sha256']})
    m={'files':rows,'verified_file_count':len(rows),'verified_bytes':sum(r['bytes'] for r in rows),'whole_goal_complete':False,'scope':scope}
    (dest/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (dest/'README.md').write_text('# '+name+'\n\n'+scope+'\n\nThe manifest records exact original paths, sizes and SHA-256 identities. Full arrays remain on data.\n')
    print(name,len(rows),m['verified_bytes'])

result=json.loads((N/'prescribed-method-comparison-v1/group0/audit/result.json').read_text())
assert result['passed'] and all(result['checks'].values()) and len(result['errors'])==27 and len(result['decomposition'])==15
assert len(result['independent_resultant_array_checks'])==12
states={j:terminal(j) for j in ('728893_0','728893_1','728918_0','728933_0')}
p=N/'meta/prescribed-method-comparison-v1-terminal.json';p.write_text(json.dumps(states,indent=2)+'\n');paths={p}
for r in result['receipts']:
    src=Path(r['path']);assert identity(src)==r
    if src.suffix=='.json': paths.add(src)
for name in ('finite-area-prescribed-native-decomposition-v1','finite-area-prescribed-shell-decomposition-v1','prescribed-method-comparison-v1'):
    campaign(paths,name,True)
for rel in ('prescribed_method_comparison.py','artifacts.py','contact_fem_exact_columns.py','column_decomposition.py'):
    paths.add(N/'prescribed-method-comparison-v1/analysis-source/research/rigid_stress/qipc_comparison'/rel)
publish('prescribed-method-ranking-v1',paths,'Qualified prescribed-C2 method comparison:27 errors,15 signed tensor splits,12 independent full-resultant-array checks and62 rehashed input identities. This is a static prescribed-load control, not the compliant-pad cost/accuracy configuration.')

states={j:terminal(j) for j in ('728464_0','728950_0','728950_1')}
p=N/'meta/stvk-wall250-terminal-v1.json';p.write_text(json.dumps(states,indent=2)+'\n');paths={p}
history=json.loads((N/'meta/stvk-wall250-complete-history-v1-audit.json').read_text());assert history['passed'] and history['steps']==400
assert identity(Path(history['journal']['path']))==history['journal']
ref=json.loads((N/'solid-pad-stvk-wall250x3-cache-v1/group0/reference/result.json').read_text());assert ref['completed'] and ref['mechanical_history_passed']
moment=json.loads((N/'solid-pad-stvk-wall250x3-cache-v1/group0/moment/result.json').read_text());assert moment['passed']
for name in ('solid-pad-stvk-wall250x3-cache-v1','stvk-wall250-fields-v1'): campaign(paths,name,True)
for p in (N/'stvk-wall250-fields-v1').glob('group*/*/result.json'):
    d=json.loads(p.read_text())
    for case in d['cases']:
        for raw in case['raw']:
            for key in ('mesh','field'): assert identity(Path(raw[key]['path']))==raw[key]
for name in ('solid-pad-stvk-final125-time-contact-v1','physical-pad-shell250-static-controls-v1'):
    campaign(paths,name)
for name in ('stvk-wall250-fields-v1-preflight.json','stvk-wall250-complete-history-v1-audit.json','stvk-final125-time-contact-v1-tasks.json','stvk-final125-time-contact-v1-targets.json','cpu-solid-reference-metadata-v1-audit.json','stvk-final125-time-contact-fields-v1-tasks.json','stvk-final125-time-contact-field-plan-v1.json','physical-pad-shell250-static-controls-v1-plan.json'):
    paths.add(N/'meta'/name)
publish('friction-wall250-reference-v1',paths,'Complete400-step250um three-wall StVK reference with unchanged history/momentum admission and q9/q25 wall2-to3 comparisons. Fine125um spatial/time/contact qualification is still pending; only immutable inputs from new controls are included.')

paths=set()
for name in ('production-throughput-pilot-v1','production-throughput-pilot-v2'): campaign(paths,name)
for name in ('production-throughput-pilot-v1-tasks.json','production-throughput-pilot-v1-plan.json','production-throughput-pilot-v1-cancel.json','source-capture-guard-v1-check.json','production-throughput-v2-preflight-source.json','production-throughput-v2-preflight-source.patch'):
    paths.add(N/'meta'/name)
paths.add(N/'production-throughput-pilot-v2/analysis-source/research/rigid_stress/shell_comparison/submit.py')
publish('production-throughput-pilot-inputs-v1',paths,'Immutable15-task native production-throughput pilot inputs,858-file source preflight, source-capture guard checks and cancelled invalid-provenance first submission. No active run output or rate is admitted in this packet. B1/B8/B32 paired rigid/DPR/SCR max/full use identical1200-step horizons and three warm repeats. Collider-load physical errors remain pending.')
