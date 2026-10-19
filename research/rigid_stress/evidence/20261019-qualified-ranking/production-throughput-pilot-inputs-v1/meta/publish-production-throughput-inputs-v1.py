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
    assert not (dest / "manifest.json").exists()
    dest.mkdir(exist_ok=True)
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

paths=set()
for name in ('production-throughput-pilot-v1','production-throughput-pilot-v2'): campaign(paths,name)
omitted=N/'production-throughput-pilot-v1/source.patch'
paths.remove(omitted)
p=N/'meta/production-throughput-cancelled-patch-identity.json'
p.write_text(json.dumps({'identity':identity(omitted),'reason':'Invalid empty-index provenance patch; remains on data, not needed to reconstruct the admitted replacement.'},indent=2)+'\n')
paths.add(p)
for name in ('production-throughput-pilot-v1-tasks.json','production-throughput-pilot-v1-plan.json','production-throughput-pilot-v1-cancel.json','source-capture-guard-v1-check.json','production-throughput-v2-preflight-source.json','production-throughput-v2-preflight-source.patch'):
    paths.add(N/'meta'/name)
paths.add(N/'production-throughput-pilot-v2/analysis-source/research/rigid_stress/shell_comparison/submit.py')
publish('production-throughput-pilot-inputs-v1',paths,'Immutable15-task native production-throughput pilot inputs,858-file source preflight, source-capture guard checks and cancelled invalid-provenance first submission. No active run output or rate is admitted. B1/B8/B32 paired rigid/DPR/SCR max/full use identical1200-step horizons and three warm repeats. Collider-load physical errors remain pending. The cancelled empty-index patch remains on data with its identity; it is excluded from the source of the replacement.')
