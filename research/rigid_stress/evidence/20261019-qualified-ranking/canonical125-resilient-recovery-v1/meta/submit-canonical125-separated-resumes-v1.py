"""Queue three source-pinned continuations behind the actual periodic-output controls."""
import datetime
import json
import subprocess
import sys
from pathlib import Path

N=Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
W=Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
node_output=subprocess.check_output(['sinfo','-N','-p','rtx-mid','-h','-o','%N|%t|%C|%m|%e'],text=True)
nodes=sorted({line.split('|')[0] for line in node_output.splitlines()}-{'rtx-209-067'})
pools=[nodes[i::3] for i in range(3)];assert all(pools) and not(set(pools[0])&set(pools[1]) or set(pools[0])&set(pools[2]) or set(pools[1])&set(pools[2]))
cases=[('final','solid-pad-stvk-canonical125-final-resume-v3','solid-pad-stvk-canonical125-reference-v1',1,'solid-pad-stvk-canonical125-final-resume-preparation-v1',200),('contact','solid-pad-stvk-canonical125-contact-resume-v1','solid-pad-stvk-canonical125-time-contact-v1',0,'solid-pad-stvk-canonical125-contact-resume-preparation-v1',200),('halfdt','solid-pad-stvk-canonical125-halfdt-resume-v1','solid-pad-stvk-canonical125-time-contact-v1',1,'solid-pad-stvk-canonical125-halfdt-resume-preparation-v1',400)]
plans=[]
for index,(role,campaign,original,group,preparation,count) in enumerate(cases):
    proof=json.loads((N/preparation/'result.json').read_text());assert proof['passed'] and proof['prefix_steps']==count and proof['terminal_scheduler_state']=='PREEMPTED'
    tasks=json.loads((N/original/'tasks.json').read_text())[group]
    reference=next(t for t in tasks if t['name']=='reference')
    reference['args']+=['--resume-checkpoint',str(N/preparation/'result.json'),'--checkpoint-stride','25']
    new=N/campaign
    taskfile=N/'meta'/(campaign+'-tasks.json');assert not taskfile.exists()
    taskfile.write_text(json.dumps([[reference,{'name':'moment','module':'research.rigid_stress.qipc_comparison.audit_solid_pad','args':['--run',str(new/'group0/reference')]}]],indent=2)+'\n')
    exclude=sorted(set(nodes+['rtx-209-067'])-set(pools[index]))
    plans.append({'role':role,'campaign':campaign,'tasks':str(taskfile),'proof':str(N/preparation/'result.json'),'inherited_steps':count,'newly_required_steps':count,'allowed_node_pool':pools[index],'excluded_nodes':exclude})
contract={'created_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_sinfo_snapshot':node_output,'control_job':'730971','control_requirement':'Every array element must succeed: coarse full-history/full-array/common-field equivalence and periodic state packing, three actual fine input controls and continued-parent metadata controls.','plans':plans,'numerical_parameters_unchanged':True,'checkpoint_stride':25,'job_scoped_placement_only':True,'whole_goal_complete':False}
p=N/'meta/canonical125-separated-resume-contract-v1.json';assert not p.exists();p.write_text(json.dumps(contract,indent=2)+'\n')
jobs={}
for plan in plans:
    command=[sys.executable,'-m','research.rigid_stress.shell_comparison.submit','--source',str(W.parent/'native-witness-consistency'),'--output',str(N/plan['campaign']),'--tasks',plan['tasks'],'--python','/mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python','--minutes','900','--concurrency','1','--cpus','4','--memory-gb','128','--node-local-cache','--cpu-only','--exclude-nodes',','.join(plan['excluded_nodes']),'--dependency','afterok:730971']
    r=subprocess.run(command,cwd=W,text=True,capture_output=True)
    with (N/'meta/canonical125-separated-resume-events-v1.jsonl').open('a') as f:f.write(json.dumps({'role':plan['role'],'argv':command,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})+'\n')
    assert r.returncode==0,(plan['role'],r.stderr)
    jobs[plan['role']]=json.loads((N/plan['campaign']/'submission.json').read_text())['job']
    print(json.dumps({'role':plan['role'],'job':jobs[plan['role']],'allowed_pool':plan['allowed_node_pool']}),flush=True)
p=N/'meta/canonical125-separated-resume-launch-v1.json';assert not p.exists();p.write_text(json.dumps({'jobs':jobs,'conditional_on_control_job':'730971','plans':plans,'whole_goal_complete':False},indent=2)+'\n')
print(json.dumps({'all_three_queued':True,'jobs':jobs,'physics_changed':False}),flush=True)
