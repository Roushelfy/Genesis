#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/original-budget-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={
 'franka-original-physical-budget-short-v1':'716371_0',
 'franka-original-physical-budget-short-audit-v1':'716372_0',
 'original-early-budget-common-state-equivalence-v1':'716377_0',
 'franka-original-stage-checked-full-history-v1':'716357_0',
 'franka-original-stage-checked-full-audit-v1':'716358_0',
}
ids=','.join(job.split('_')[0] for job in runs.values())
jobs=subprocess.check_output(['sacct','-n','-P','-X','-j',ids,
 '--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList'],text=True)
(meta/'jobs.txt').write_text(jobs)
(meta/'job-commands.txt').write_text(subprocess.check_output(['sacct','-n','-P','-X','-j',ids,
 '--format=JobID%40,SubmitLine%16000'],text=True))
states={line.split('|')[0].strip():line.split('|')[1].strip() for line in jobs.splitlines() if line.strip()}
closures=[];records=[];extras=set()
for name,job in runs.items():
 run=p/name;cancelled=job=='716357_0';code=int((run/'returncode.txt').read_text())
 assert states[job].startswith('CANCELLED') if cancelled else states[job]=='COMPLETED'
 assert code==0
 for kind in ['source','input-source']:
  before=run/(kind+'.sha256')
  if not before.exists():continue
  assert before.read_bytes()==(run/(kind+'-after.sha256')).read_bytes()
  for line in before.read_text().splitlines():
   sha,path=line.split('  ',1);assert hashlib.sha256((run/kind/path).read_bytes()).hexdigest()==sha
   closures.append({'run':name,'identity':kind,'file':path,'sha256':sha})
 records.append({'run':name,'job':job,'scheduler_state':states[job],'returncode':code,
                 'producer_EXIT_records_present':True,'scheduler_success_inferred_from_wrapper_code':False})
 extras.update(p.glob('*'+job+'.log'))
def report(name):return json.loads((p/name/'results/summary.json').read_text())
audit=report('franka-original-physical-budget-short-audit-v1')
assert audit['steps_checked']==1944 and audit['complete_requested_history'] and audit['requested_steps']==1944
assert all(row['body_dynamic_residual_N']<=1e-9 and row['actual_robot_step_effort_residual']<=1e-9 for row in audit['rows'])
equivalence=report('original-early-budget-common-state-equivalence-v1')
assert [row['step'] for row in equivalence['rows']]==list(range(1938,1945))
for row in equivalence['rows']:
 assert max(row['all_corner_tensor_relative_change'],row['all_quadrature_tensor_relative_change'],row['raw_global_corner_peak_relative_change'])<=1e-6
baseline=report('franka-original-stage-checked-full-audit-v1')
assert baseline['steps_checked']==1964 and not baseline['complete_requested_history']
assert baseline['producer_returncode']==0 and baseline['producer_report_status']=='missing'
observation=p/'original-early-budget-common-state-observation-v1'
observed=json.loads((observation/'observation.json').read_text())
assert observed['all1937_copied_prefix_files_byte_identical']
for row in observed['files']:
 assert hashlib.sha256(pathlib.Path(row['observed']).read_bytes()).hexdigest()==row['sha256']
extras.update(path for path in observation.rglob('*') if path.is_file())
extras.update([p/'meta/original-physical-budget-full-pipeline-commands-v1.json',
 p/'meta/original-physical-budget-replacement-decision-v1.json'])
summary={'scope':'Independent1944-state short original history, complete1938–1944 field equivalence, cancelled1964-state baseline',
 'runs':records,'source_hash_records':closures,'independent1944_short_audit':audit,
 'seven_common_state_array_and_full_tensor_equivalence':equivalence,
 'cancelled1964_prefix_independent_audit':baseline,
 'all1937_original_prefix_files_byte_identical':True,
 'full330_macro_replacement_job':'716378_0','live_replacement_results_excluded':True,
 'goal_completion':False}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt'])
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'closed_runs':len(runs),'source_hash_records':len(closures),'qualified_short_states':1944}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extras < "$task_meta/extras.txt"
task_destination="research/rigid_stress/evidence/20261014-physical/original-budget-qualification-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_destination" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extras[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'The completed243-macro short history has1944 independently audited physical states, not the full330-macro application. All1937 copied source states are byte-identical; complete1938–1944 arrays/corner/quadrature tensors pass existing numerical budgets. The redundant baseline isCANCELLED with actual wrapper0, missing report and1964 independently audited states. Wrapper zero is not producer completion. Live replacement results are excluded. All physical convergence, model/feedback and verified accuracy/cost requirements remain open.' \
    > "$task_meta/publication.log" 2>&1
python3 research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_destination/manifest.json" --originals --output "$task_meta/verification.json" \
    > "$task_meta/verification.log" 2>&1
cp "$task_meta/verification.json" "$task_destination/verification.json"
python3 - "$task_root/runs" "$task_base/publication-staging" "$task_destination" "$task_meta" <<'PY'
import hashlib,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path('research/rigid_stress/scripts').resolve()))
from verify_nonconvex_publication import digest
root,staging,destination,meta=map(pathlib.Path,sys.argv[1:])
manifest_path=destination/'manifest.json';manifest=json.loads(manifest_path.read_text())
assert not manifest['summary']['sharded_original_files']
stage=staging/destination.name
for row in manifest['files']:
 original=pathlib.Path(row['original'])
 actual=root/original.relative_to(stage) if original.is_relative_to(stage) else original
 assert digest(actual)==(row['sha256'],row['bytes']),actual
result={'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
 'true_runtime_original_entries_verified':len(manifest['files']),
 'source_hash_records_verified':len(manifest['summary']['source_hash_records']),
 'all_stored_and_decompressed_bytes_verified':'verification.json','passed':True,'goal_completion':False}
for output in [meta/'true-runtime-verification.json',destination/'true-runtime-verification.json']:
 output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
PY
