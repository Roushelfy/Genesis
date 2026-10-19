"""Publish complete timing/control receipts, retaining failed conformance and original profile failure."""
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
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}

def state(job, expected):
    text=subprocess.check_output(['sacct','-j',job,'-X','-n','-P','--format=JobID,State,ExitCode'],text=True)
    assert text.strip()==job+'|'+expected,text
    return text

def add(paths,name,terminal=True):
    patterns=['submission.json','tasks.json','source*.json','source*.patch','analysis-files.json','job.sh','workload-*.sh']
    if terminal:patterns+=['group*/returncode.txt','group*/*.returncode.txt','group*/*.log','group*/*.json','group*/**/result.json','group*/**/start.json','group*/**/conformance.json','group*/**/*.csv','logs/*.log']
    for pat in patterns:paths.update(p for p in (N/name).glob(pat) if p.is_file())

def publish(name, paths, scope):
    paths.add(Path(__file__))
    dest=E/name;dest.mkdir(exist_ok=False);rows=[]
    for source in sorted(paths):
        data=source.read_bytes();assert len(data)<3_000_000,str(source)
        rel=source.relative_to(N);encoded=len(data)>300_000 or source.suffix=='.log'
        payload=gzip.compress(data,mtime=0) if encoded else data
        target=dest/(str(rel)+'.gz' if encoded else str(rel));target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
        assert (gzip.decompress(target.read_bytes()) if encoded else target.read_bytes())==data
        rows.append({'source':str(source),'copy':str(target.relative_to(dest)),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'encoding':'gzip' if encoded else 'identity','original_bytes':len(data),'original_sha256':hashlib.sha256(data).hexdigest()})
    manifest={'files':rows,'verified_file_count':len(rows),'verified_bytes':sum(r['bytes'] for r in rows),'whole_goal_complete':False,'scope':scope}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (dest/'README.md').write_text('# '+name+'\n\n'+scope+'\n\nLarge JSON and raw logs use deterministic gzip. Decompression preserves every original byte; the manifest records both encoded and original SHA-256/size. Use `python -m gzip -d FILE.gz` on a separate working copy to inspect those records. Full trajectory/source-history NPZ files remain on data with exact identities.\n')
    print(name,len(rows),manifest['verified_bytes'])

states={}
for job in ('728966_0','728966_1','728966_2','728986_0','728971_0','728971_1','728971_2','728989_0','728988_0','728996_0'):states[job]=state(job,'COMPLETED|0:0')
for job in ('728977_0','728985_0','728991_0'):states[job]=state(job,'FAILED|1:0')
p=N/'meta/native-production-terminal-v1.json';p.write_text(json.dumps(states,indent=2)+'\n')
performance=json.loads((N/'production-throughput-audit-v1/group0/audit/result.json').read_text());assert performance['timing_receipts_passed'] and len(performance['rows'])==15
conformance=json.loads((N/'native-production-full-audit-v1/group0/audit/result.json').read_text());assert not conformance['passed'] and conformance['case_count']==15
paths={p}
for d in (performance,conformance):
    for receipt in d['receipts']:
        assert identity(Path(receipt['path']))==receipt
for name in ('production-throughput-pilot-v2','production-throughput-audit-v1','native-production-controls-v1','native-production-b1-b8-audit-v1','native-production-full-audit-v1','production-stage-profile-v1','production-stage-profile-scr-v2','native-motion-diagnostics-v1','native-motion-diagnosis-v1'):add(paths,name)
for name in ('native-controller-inputs-v1',):add(paths,name,False)
for name in ('native-production-controls-v1-plan.json','native-production-b1-b8-audit-v1-preflight.json','production-throughput-audit-v1-preflight.json','production-profile-scale-observer-v1-check.json','native-motion-diagnostics-v1-plan.json'):
    paths.add(N/'meta'/name)
for campaign,module in [('native-production-controls-v1','native_production_controls.py'),('production-throughput-audit-v1','production_throughput_audit.py'),('native-production-full-audit-v1','audit_native_production_controls.py'),('native-motion-diagnosis-v1','native_motion_diagnosis.py')]:
    paths.add(N/campaign/'analysis-source/research/rigid_stress/qipc_comparison'/module)
publish('native-production-measurements-v1',paths,'Terminal15-case B1/B8/B32 rigid/DPR/SCR max/full scene timings, complete motion/output audits and separate stage profiles. Zero invalid-step counts and timing receipts pass, but strict motion conformance fails; these are measured baseline rates, not a fully admitted production recommendation. Retain the original SCR profile-normalization failure and corrected observer-only profile. Completed motion diagnostics reproduce the discrepancy without any stress updates; the active controller-only control contributes immutable inputs only. Physical collider-load errors remain pending.')

states={j:state(j,'COMPLETED|0:0') for j in ('728981_0','728990_0')}
p=N/'meta/native-module-replay-terminal-v1.json';p.write_text(json.dumps(states,indent=2)+'\n');paths={p}
d=json.loads((N/'native-module-replay-audit-v1/group0/audit/result.json').read_text());assert d['passed'] and len(d['rows'])==4
for receipt in d['receipts']:assert identity(Path(receipt['path']))==receipt
for name in ('native-module-replay-v2','native-module-replay-audit-v1'):add(paths,name)
for name in ('native-module-replay-v2-plan.json','native-module-replay-audit-v1-preflight.json'):paths.add(N/'meta'/name)
for campaign,module in [('native-module-replay-v2','native_production_controls.py'),('native-module-replay-audit-v1','native_module_replay_audit.py')]:paths.add(N/campaign/'analysis-source/research/rigid_stress/qipc_comparison'/module)
publish('native-module-replay-v1',paths,'Four admitted numerical B8 module replay modes, complete1200-frame histories,24 resident input buffers, actual stress reset schedule and three warm timing repeats on one GPU. Original scene recording, archival and independent validation are outside timing. Module time includes input restoration and native recovery/acceptance, not dynamics. Physical errors and production motion admission are separate unresolved claims.')
