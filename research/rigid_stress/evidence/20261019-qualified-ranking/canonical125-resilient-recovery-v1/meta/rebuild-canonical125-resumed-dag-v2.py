"""Replace only never-started consumers after actual final-parent preemption."""
import ast
import copy
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
ENGINE=W.parent/'native-witness-consistency'
OLD=str(N/'solid-pad-stvk-canonical125-reference-v1/group1/reference')
NEW=str(N/'solid-pad-stvk-canonical125-final-resume-v2/group0/reference')
META=N/'meta'
EVENTS=META/'canonical125-recovered-dag-events-v1.jsonl'
assert not EVENTS.exists()

def write(path,data):
    if path.exists():
        assert json.loads(path.read_text())==data,path
        return
    path.write_text(json.dumps(data,indent=2)+'\n')

def event(data):
    with EVENTS.open('a') as f:f.write(json.dumps({'at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),**data})+'\n')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def state(job):return subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
def transform(value):
    if isinstance(value,str):return value.replace(OLD,NEW)
    if isinstance(value,list):return [transform(v) for v in value]
    if isinstance(value,dict):return {k:transform(v) for k,v in value.items()}
    return value

def original(name):return transform(read(N/name/'tasks.json'))

def set_option(task,key,value):
    a=task['args']
    if key in a:a[a.index(key)+1]=str(value)
    else:a.extend([key,str(value)])

controls=read(N/'solid-pad-resume-lineage-controls-v1/group0/controls/result.json')
assert controls['passed'] and len(controls['checks'])==26 and all(controls['checks'].values())
accounting=subprocess.check_output(['sacct','-j','730913','-P','--noheader','--format=JobID,State,ExitCode'],text=True)
assert '730913_0|COMPLETED|0:0' in accounting
for name in ['qualify_friction_reference.py','solid_pad_resume_admission.py','check_solid_pad_resume_admission.py']:
    assert (W/'research/rigid_stress/qipc_comparison'/name).read_bytes()==(N/'solid-pad-resume-lineage-controls-v1/analysis-source/research/rigid_stress/qipc_comparison'/name).read_bytes(),name
for name in ['plate_solid_pad.py','solid_pad_checkpoint.py']:
    assert (W/'research/rigid_stress/qipc_comparison'/name).read_bytes()==(N/'solid-pad-stvk-canonical125-final-resume-v2/analysis-source/research/rigid_stress/qipc_comparison'/name).read_bytes(),name
for name in ['friction_reference_gates.py','stvk_figure_render.py','rank_stvk_pad_methods.py','pad_cost_accuracy.py','pad_candidate_admission.py','contact_fem_exact_columns.py']:
    relative='research/rigid_stress/qipc_comparison/'+name
    assert (W/relative).read_bytes()==subprocess.check_output(['git','show','HEAD:'+relative],cwd=W),name
assert 'JobState=RUNNING' in state('730903_0')
assert all('JobState=RUNNING' in state(j) for j in ['729314_0','729314_1'])
proof=read(N/'solid-pad-stvk-canonical125-final-resume-preparation-v1/result.json')
assert proof['passed'] and proof['terminal_scheduler_state']=='PREEMPTED' and sha(Path(proof['journal']['path']))==proof['journal']['sha256']
for group in [0,1]:
    assert (N/'stvk-canonical125-fields-v2'/f'group{group}/returncode.txt').read_text().strip()=='0'

names={
 'spatial_guard':'canonical125-recovered-spatial-parent-admission-v1',
 'spatial_compute':'stvk-canonical125-recovered-spatial-compute-v1',
 'spatial_view':'stvk-canonical125-recovered-fields-v1',
 'full_guard':'canonical125-recovered-parent-admission-v1',
 'pair_guard':'contact125-recovered-pair-parent-admission-v1',
 'controls':'stvk-canonical125-recovered-time-contact-fields-v1',
 'qualification':'canonical-friction-recovered-qualification-v1',
 'methods':'canonical-stvk-recovered-method-fields-v1',
 'data':'canonical-friction-recovered-plot-data-v1',
 'ranking':'canonical-stvk-recovered-method-ranking-v1',
 'figures':'canonical-stvk-recovered-figures-v1',
 'cost':'pad-friction-recovered-cost-accuracy-v1',
 'figure_negative':'recovered-figure-admission-negative-v1',
}
override=META/'canonical125-recovered-parent-overrides-v1.json'
write(override,{'final125':str(Path(NEW).relative_to(N))})
plans={}
plans['spatial_guard']=original('canonical125-spatial-parent-admission-v1')
plans['spatial_compute']=original('stvk-canonical125-fields-v2')[2:]
plans['full_guard']=original('canonical125-parent-admission-v1')
plans['pair_guard']=original('contact125-pair-parent-admission-v1')
plans['controls']=original('stvk-canonical125-time-contact-fields-v2')
plans['qualification']=original('canonical-friction-reference-qualification-v1')
q=plans['qualification'][0][1]
set_option(q,'--parent-admission',N/names['qualification']/'group0/parents/result.json')
set_option(q,'--spatial-campaign',names['spatial_view'])
set_option(q,'--control-campaign',names['controls'])
set_option(q,'--parent-overrides',override)
plans['methods']=original('canonical-stvk-method-fields-v2')
plans['data']=original('canonical-friction-plot-data-v1')
plans['ranking']=original('canonical-stvk-method-ranking-v1')
set_option(plans['ranking'][0][0],'--reference-qualification',N/names['qualification']/'group0/qualification/result.json')
set_option(plans['ranking'][0][0],'--fields-campaign',names['methods'])
plans['figures']=original('canonical-stvk-figures-v1')
f=plans['figures'][0][0]
set_option(f,'--reference-qualification',N/names['qualification']/'group0/qualification/result.json')
set_option(f,'--ranking',N/names['ranking']/'group0/ranking/result.json')
set_option(f,'--force-data',N/names['data']/'group0/data/result.json')
set_option(f,'--fields-campaign',names['methods'])
plans['cost']=original('pad-friction-cost-accuracy-v1')
set_option(plans['cost'][0][0],'--ranking',N/names['ranking']/'group0/ranking/result.json')
plans['figure_negative']=copy.deepcopy(plans['figures'])
set_option(plans['figure_negative'][0][0],'--reference-qualification',N/'pad-normal-qualification-v2/group0/qualification/result.json')
for role,groups in plans.items():write(META/(names[role]+'-tasks.json'),groups)

old_pending=['729487','729478','730699',*[f'729479_{i}' for i in range(2,8)],'729480','729513','729539','729519','729835','729938','730062']
before={j:state(j) for j in old_pending}
assert all((v:=re.findall(r'(?:^| )JobState=(\S+)',s)) and set(v)=={'PENDING'} for s in before.values())
original_hashes={str(N/c/x):sha(N/c/x) for c in ['stvk-canonical125-fields-v2','stvk-canonical125-time-contact-fields-v2','canonical-friction-reference-qualification-v1','canonical-stvk-method-fields-v2'] for x in ['tasks.json','analysis-files.json','source.json','source.patch','job.sh']}
contract={'created_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Actual729312_1 terminal PREEMPTED; restore saved200-step normal state in730903 then require a complete400-step own history/momentum admission and explicit continued-parent lineage. Reuse the two completed surface2 field groups, replace never-started affected consumers only.','terminal_parent':subprocess.check_output(['sacct','-j','729312_1','-P','--noheader','--format=JobID,State,ExitCode,Start,End'],text=True),'new_reference':NEW,'resume_job':'730903','unchanged_other_live_parents':{j:state(j) for j in ['729314_0','729314_1']},'cancel_only_never_started':before,'new_campaign_names':names,'new_task_plans':{role:str(META/(names[role]+'-tasks.json')) for role in plans},'parent_overrides':str(override),'original_capture_hashes':original_hashes,'physical_comparison_targets_unchanged':{'spatial_fields':.02,'spatial_peaks':.05,'controls_fields':.005,'controls_peaks':.01,'comparisons':36,'envelopes':18,'method_error_rows':54,'metric_ranks':18},'lineage_controls':{'job':'730913','passed':True,'checks':26},'completed_surface_comparisons_reused':True,'whole_goal_complete':False}
write(META/'canonical125-recovered-dag-contract-v1.json',contract)
command=['scancel',*old_pending]
r=subprocess.run(command,text=True,capture_output=True)
event({'operation':'cancel_obsolete_pending_consumers','argv':command,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
assert r.returncode==0
assert all(sha(Path(path))==value for path,value in original_hashes.items())
view=N/names['spatial_view'];view.mkdir(exist_ok=False)
links=[]
for group in range(8):
    target=N/'stvk-canonical125-fields-v2'/f'group{group}' if group<2 else N/names['spatial_compute']/f'group{group-2}'
    link=view/f'group{group}';relative=os.path.relpath(target,link.parent);link.symlink_to(relative,target_is_directory=True)
    links.append({'group':group,'link':str(link),'relative_target':relative,'target':str(target),'reused_completed_group':group<2})
write(view/'assembly.json',{'links':links,'scope':'Explicit relative data-side view of unchanged completed surface2 groups and new affected spatial groups; original captures and outputs remain intact. Each field raw input identity is checked independently by qualification.'})

jobs={}
def submit(role,dependency,original_campaign):
    saved=read(N/original_campaign/'submission.json')
    values={a.split('=',1)[0]:a.split('=',1)[1] for a in saved['command'] if '=' in a}
    lines=(N/original_campaign/'workload-0.sh').read_text().splitlines()
    python=next(shlex.split(line)[0] for line in lines if ' -m research.' in line and line.lstrip().startswith('/mnt/data/'))
    concurrency=values['--array'].split('%')[1] if '%' in values['--array'] else '1'
    command=[sys.executable,'-m','research.rigid_stress.shell_comparison.submit','--source',str(ENGINE),'--output',str(N/names[role]),'--tasks',str(META/(names[role]+'-tasks.json')),'--python',python,'--minutes',values['--time'],'--concurrency',concurrency,'--cpus',values['--cpus-per-task'],'--memory-gb',values['--mem'].removesuffix('G'),'--node-local-cache','--cpu-only']
    if dependency:command+=['--dependency',dependency]
    result=subprocess.run(command,cwd=W,text=True,capture_output=True)
    event({'operation':'submit','role':role,'argv':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    assert result.returncode==0,(role,result.stderr)
    jobs[role]=read(N/names[role]/'submission.json')['job']
    print(json.dumps({'role':role,'job':jobs[role],'dependency':dependency}),flush=True)

submit('spatial_guard','afterok:730903','canonical125-spatial-parent-admission-v1')
submit('spatial_compute','afterok:'+jobs['spatial_guard'],'stvk-canonical125-fields-v2')
submit('full_guard','afterok:730903:729314','canonical125-parent-admission-v1')
submit('pair_guard','afterok:730903:729314_0','contact125-pair-parent-admission-v1')
submit('controls','afterok:'+jobs['full_guard'],'stvk-canonical125-time-contact-fields-v2')
for i in (0,1):
    command=['scontrol','update',f'JobId={jobs["controls"]}_{i}','Dependency=afterok:'+jobs['pair_guard']]
    result=subprocess.run(command,text=True,capture_output=True)
    event({'operation':'set_independent_contact_readiness','argv':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    assert result.returncode==0
submit('qualification','afterany:'+jobs['spatial_compute']+':'+jobs['controls'],'canonical-friction-reference-qualification-v1')
submit('methods','afterok:'+jobs['qualification'],'canonical-stvk-method-fields-v2')
submit('data','afterok:'+jobs['qualification'],'canonical-friction-plot-data-v1')
submit('ranking','afterany:'+jobs['methods'],'canonical-stvk-method-ranking-v1')
submit('figures','afterany:'+jobs['ranking']+':'+jobs['data'],'canonical-stvk-figures-v1')
submit('cost','afterany:'+jobs['ranking'],'pad-friction-cost-accuracy-v1')
submit('figure_negative',None,'canonical-stvk-figures-v1')
assert all(sha(Path(path))==value for path,value in original_hashes.items())
current={role:state(job+'_0') for role,job in jobs.items()}
for i in (0,1):assert 'afterok:'+jobs['pair_guard']+'_' in state(jobs['controls']+f'_{i}')
for i in (2,3):assert 'afterok:'+jobs['full_guard']+'_' in state(jobs['controls']+f'_{i}')
write(META/'canonical125-recovered-dag-launch-v1.json',{'jobs':jobs,'states':current,'assembly':str(view/'assembly.json'),'reused_completed_groups':2,'new_spatial_groups':6,'original_captures_unchanged':True,'all_final_qualification_requirements_preserved':True,'whole_goal_complete':False})
print(json.dumps({'complete_launch':True,'jobs':jobs,'new_spatial_tasks':6,'reused_surface_groups':2,'full_goal_complete':False}),flush=True)
