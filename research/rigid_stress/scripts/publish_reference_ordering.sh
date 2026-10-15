#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/reference-ordering-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={'egg-paired-cad-level3-generic-contact-v1':'716729_0',
 'egg-cad-level3-reference-ordering-qualification-v1':'716810_0',
 'egg-cad-level3-reference-ordering-qualification-audit-v1':'716826_0'}
ids=','.join(job.split('_')[0] for job in runs.values())
jobs=subprocess.check_output(['sacct','-n','-P','-X','-j',ids,'--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList'],text=True)
(meta/'jobs.txt').write_text(jobs)
(meta/'job-commands.txt').write_text(subprocess.check_output(['sacct','-n','-P','-X','-j',ids,'--format=JobID%40,SubmitLine%18000'],text=True))
states={line.split('|')[0].strip():line.split('|')[1].strip() for line in jobs.splitlines() if line.strip()}
identities=[];records=[]
for name,job in runs.items():
 run=p/name;assert states[job]=='COMPLETED' and int((run/'returncode.txt').read_text())==0
 for kind in ['source','input-source']:
  f=run/(kind+'.sha256');assert f.read_bytes()==(run/(kind+'-after.sha256')).read_bytes()
  for line in f.read_text().splitlines():
   sha,path=line.split('  ',1);assert hashlib.sha256((run/kind/path).read_bytes()).hexdigest()==sha
   identities.append({'run':name,'identity':kind,'file':path,'sha256':sha})
 records.append({'run':name,'job':job,'state':states[job],'returncode':0})
candidate=p/'egg-cad-level3-reference-ordering-qualification-v1'
s=json.loads((candidate/'results/summary.json').read_text())
assert s['all_original_arrays_and_complete_fields_qualified'] and s['states_checked']==len(s['rows'])==5
assert all(max(r['all_corner_tensor_relative_change'],r['all_quadrature_tensor_relative_change'],r['raw_global_peak_relative_change'])<=1e-6 for r in s['rows'])
assert all(r['complete_body_dynamic_residual_N']<=1e-9 and r['complete_mapped_dynamic_residual_N']<=1e-9 for r in s['rows'])
audit=json.loads((p/'egg-cad-level3-reference-ordering-qualification-audit-v1/results/summary.json').read_text())
assert audit['states_checked']==len(audit['rows'])==5
assert all(r['complete_body_dynamic_residual_N']<=1e-9 and r['complete_mapped_dynamic_residual_N']<=1e-9 for r in audit['rows'])
timing=[]
for name in ['egg-paired-cad-level3-generic-contact-v1','egg-cad-level3-reference-ordering-qualification-v1']:
 run=p/name;linear=json.loads((run/'linear_solver_stats.json').read_text())['linear_solves']
 assert len(linear)==15 and sum(r['fresh_factorization'] for r in linear)==1
 timing.append({'run':name,'linear_solves':len(linear),'fresh_factors':1,'first_linear_s':linear[0]['elapsed_s'],'total_linear_s':sum(r['elapsed_s'] for r in linear),'node':(run/'node.txt').read_text().strip()})
same_node=timing[0]['node']==timing[1]['node']
summary={'scope':'Complete five-state paired CAD reference numerical qualification and independent audit after sparse column ordering changes',
 'runs':records,'source_identity_records':identities,'all_original_arrays_and_fields':s,'independent_all_state_audit':audit,
 'observed_single_run_reference_linear_timings':timing,'same_node':same_node,'descriptive_elapsed_ratio_not_controlled_speedup':timing[0]['total_linear_s']/timing[1]['total_linear_s'],
 'production_engine_changed':False,'goal_completion':False,
 'limitations':'Only a sparse column permutation changes. Original geometry/material/mass, initializer, controls/history, force budgets and internal tolerances remain fixed. One7680-tet five-state baseline has complete array/corner/quadrature/raw-peak comparison and a separate independent force/global-field audit. Timings are single reference runs on different nodes; their descriptive ratio is not a controlled ordering speedup, warmed production rate or physical accuracy. Fine30720-tet baseline/candidate and adaptive references remain live and excluded; no cancellation/replacement is claimed. Full physical/contact/geometry/history/model/feedback/verified-cost acceptance remains open.'}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
extras=[meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt',p/'meta/reference-ordering-qualification-commands-v1.json',p/'meta/ordered-cad-level4-commands-v1.json']
for job in runs.values():extras.extend(p.glob('*'+job+'.log'))
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
(meta/'extras.txt').write_text(''.join(str(f)+'\n' for f in extras))
print(json.dumps({'closed_runs':len(runs),'source_identity_records':len(identities),'qualified_states':5,'linear_s_ratio':summary['descriptive_elapsed_ratio_not_controlled_speedup']}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/reference-ordering-qualification-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'All five original7680-tet CAD states retain every saved array, complete corner/quadrature stress and raw peak within unchanged numerical gates after a sparse column-ordering change. A separate independent all-state audit passes original force gates. Fifteen linear solves and one fresh factor per path give descriptive reference timings on different nodes, not a controlled speedup, warmed native cost or physical convergence. Production remains unchanged; all live fine/adaptive producers are excluded.' \
    > "$task_meta/publication.log" 2>&1
python3 research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_meta/verification.json" \
    > "$task_meta/verification.log" 2>&1
