"""Bind every consumer to the three independently placed, periodically saved continuations."""
import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
N=Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
W=Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
M=N/'meta'
read=lambda p:json.loads(p.read_text())
launch=read(M/'canonical125-separated-resume-launch-v1.json');physical=launch['jobs']
new_runs={p['role']:str(N/p['campaign']/'group0/reference') for p in launch['plans']}
old_runs={'final':str(N/'solid-pad-stvk-canonical125-reference-v1/group1/reference'),'contact':str(N/'solid-pad-stvk-canonical125-time-contact-v1/group0/reference'),'halfdt':str(N/'solid-pad-stvk-canonical125-time-contact-v1/group1/reference')}
report=read(N/'solid-pad-periodic-state-reinspection-v1/group0/admission/result.json')
assert report['passed'] and len(report['checks'])==99 and all(report['checks'].values())
assert '730990_0|COMPLETED|0:0' in subprocess.check_output(['sacct','-j','730990','-P','--noheader','--format=JobID,State,ExitCode'],text=True)

def write(p,d):
    assert not p.exists(),p
    p.write_text(json.dumps(d,indent=2)+'\n')
def state(j):return subprocess.check_output(['scontrol','show','job',j,'-o'],text=True)
def event(d):
    with (M/'canonical125-resilient-dag-events-v1.jsonl').open('a') as f:f.write(json.dumps({'at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),**d})+'\n')
def trans(v):
    if isinstance(v,str):
        for role,old in old_runs.items():v=v.replace(old,new_runs[role])
        return v
    if isinstance(v,list):return [trans(x) for x in v]
    if isinstance(v,dict):return {k:trans(x) for k,x in v.items()}
    return v

def original(c):return trans(read(N/c/'tasks.json'))
def option(t,k,v):
    a=t['args']
    if k in a:a[a.index(k)+1]=str(v)
    else:a.extend([k,str(v)])

names={'spatial_guard':'canonical125-resilient-spatial-parent-admission-v1','spatial_compute':'stvk-canonical125-resilient-spatial-compute-v1','spatial_view':'stvk-canonical125-resilient-fields-v1','full_guard':'canonical125-resilient-parent-admission-v1','pair_guard':'contact125-resilient-pair-parent-admission-v1','controls':'stvk-canonical125-resilient-time-contact-fields-v1','qualification':'canonical-friction-resilient-qualification-v1','methods':'canonical-stvk-resilient-method-fields-v1','data':'canonical-friction-resilient-plot-data-v1','ranking':'canonical-stvk-resilient-method-ranking-v1','figures':'canonical-stvk-resilient-figures-v1','cost':'pad-friction-resilient-cost-accuracy-v1'}
recipes={'spatial_guard':'canonical125-spatial-parent-admission-v1','spatial_compute':'stvk-canonical125-fields-v2','full_guard':'canonical125-parent-admission-v1','pair_guard':'contact125-pair-parent-admission-v1','controls':'stvk-canonical125-time-contact-fields-v2','qualification':'canonical-friction-reference-qualification-v1','methods':'canonical-stvk-method-fields-v2','data':'canonical-friction-plot-data-v1','ranking':'canonical-stvk-method-ranking-v1','figures':'canonical-stvk-figures-v1','cost':'pad-friction-cost-accuracy-v1'}
plans={role:original(c) for role,c in recipes.items()};plans['spatial_compute']=plans['spatial_compute'][2:]
override=M/'canonical125-resilient-parent-overrides-v1.json'
write(override,{key+'125':str(Path(new_runs[role]).relative_to(N)) for role,key in [('final','final'),('contact','contact'),('halfdt','halfdt')]})
q=plans['qualification'][0][1]
for k,v in [('--parent-admission',N/names['qualification']/'group0/parents/result.json'),('--spatial-campaign',names['spatial_view']),('--control-campaign',names['controls']),('--parent-overrides',override)]:option(q,k,v)
r=plans['ranking'][0][0]
option(r,'--reference-qualification',N/names['qualification']/'group0/qualification/result.json');option(r,'--fields-campaign',names['methods'])
f=plans['figures'][0][0]
for k,v in [('--reference-qualification',N/names['qualification']/'group0/qualification/result.json'),('--ranking',N/names['ranking']/'group0/ranking/result.json'),('--force-data',N/names['data']/'group0/data/result.json'),('--fields-campaign',names['methods'])]:option(f,k,v)
option(plans['cost'][0][0],'--ranking',N/names['ranking']/'group0/ranking/result.json')
for role,tasks in plans.items():write(M/(names[role]+'-tasks.json'),tasks)
old=read(M/'canonical125-recovered-dag-launch-v1.json')['jobs']
roles=['cost','figures','ranking','data','methods','qualification','controls','spatial_compute','pair_guard','full_guard','spatial_guard']
before={role:state(old[role]) for role in roles}
assert all((v:=re.findall(r'(?:^| )JobState=(\S+)',s)) and set(v)=={'PENDING'} for s in before.values())
contract={'created_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'physical_jobs':physical,'new_parent_runs':new_runs,'parent_overrides':str(override),'new_campaign_names':names,'new_task_files':{role:str(M/(names[role]+'-tasks.json')) for role in plans},'cancel_only_never_started':{role:{'job':old[role],'state':before[role]} for role in roles},'periodic_control_original_failed_wrapper_preserved':'730971_0','independent_scientific_control_reinspection_passed':{'job':'730990','checks':99},'reuse_completed_surface_groups':2,'new_spatial_groups':6,'required_comparisons':36,'required_reference_envelopes':18,'required_method_error_rows':54,'required_metric_ranks':18,'physical_targets_unchanged':True,'whole_goal_complete':False}
write(M/'canonical125-resilient-dag-contract-v1.json',contract)
for stage in [roles[:6],roles[6:]]:
    command=['scancel',*[old[role] for role in stage]]
    p=subprocess.run(command,text=True,capture_output=True)
    event({'operation':'cancel_obsolete_pending_stage','argv':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr});assert p.returncode==0
view=N/names['spatial_view'];view.mkdir(exist_ok=False);links=[]
for i in range(8):
    target=N/'stvk-canonical125-fields-v2'/f'group{i}' if i<2 else N/names['spatial_compute']/f'group{i-2}'
    link=view/f'group{i}';relative=os.path.relpath(target,link.parent);link.symlink_to(relative,target_is_directory=True)
    links.append({'group':i,'link':str(link),'relative_target':relative,'target':str(target),'reused_completed_group':i<2})
write(view/'assembly.json',{'links':links,'scope':'Explicit data-side relative links preserve completed surface2 comparisons and map the six new wall/surface3/joint groups. No original output or tracked path is moved or replaced.'})
jobs={}
def submit(role,dependency):
    c=recipes[role];s=read(N/c/'submission.json');values={x.split('=',1)[0]:x.split('=',1)[1] for x in s['command'] if '=' in x}
    python=next(shlex.split(line)[0] for line in (N/c/'workload-0.sh').read_text().splitlines() if ' -m research.' in line and line.lstrip().startswith('/mnt/data/'))
    concurrency=values['--array'].split('%')[1] if '%' in values['--array'] else '1'
    command=[sys.executable,'-m','research.rigid_stress.shell_comparison.submit','--source',str(W.parent/'native-witness-consistency'),'--output',str(N/names[role]),'--tasks',str(M/(names[role]+'-tasks.json')),'--python',python,'--minutes',values['--time'],'--concurrency',concurrency,'--cpus',values['--cpus-per-task'],'--memory-gb',values['--mem'].removesuffix('G'),'--node-local-cache','--cpu-only','--exclude-nodes','rtx-209-067','--dependency',dependency]
    p=subprocess.run(command,cwd=W,text=True,capture_output=True);event({'operation':'submit','role':role,'argv':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr});assert p.returncode==0,(role,p.stderr)
    jobs[role]=read(N/names[role]/'submission.json')['job'];print(json.dumps({'role':role,'job':jobs[role],'dependency':dependency}),flush=True)
submit('spatial_guard','afterok:'+physical['final']);submit('spatial_compute','afterok:'+jobs['spatial_guard'])
submit('full_guard','afterok:'+':'.join(physical.values()));submit('pair_guard','afterok:'+physical['final']+':'+physical['contact'])
submit('controls','afterok:'+jobs['full_guard'])
for i in (0,1):
    command=['scontrol','update','JobId='+jobs['controls']+f'_{i}','Dependency=afterok:'+jobs['pair_guard']]
    p=subprocess.run(command,text=True,capture_output=True);event({'operation':'contact_pair_readiness','argv':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr});assert p.returncode==0
submit('qualification','afterany:'+jobs['spatial_compute']+':'+jobs['controls']);submit('methods','afterok:'+jobs['qualification']);submit('data','afterok:'+jobs['qualification']);submit('ranking','afterany:'+jobs['methods']);submit('figures','afterany:'+jobs['ranking']+':'+jobs['data']);submit('cost','afterany:'+jobs['ranking'])
write(M/'canonical125-resilient-dag-launch-v1.json',{'jobs':jobs,'physical_jobs':physical,'campaign_names':names,'assembly':str(view/'assembly.json'),'all_required_scientific_gates_preserved':True,'fine_reference_qualified':False,'whole_goal_complete':False})
print(json.dumps({'complete_launch':True,'jobs':jobs,'physical_jobs':physical}),flush=True)
