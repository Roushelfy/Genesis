#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/original-history-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={
 'franka-original-residual-globalized-history-v1':'715815_0',
 'franka-original-residual-history-audit-v1':'715817_0',
 'franka-original-history-fields-v1':'715856_0',
 'franka-original-history-native-v1':'715878_0',
 'franka-original-history-bounds-v1':'715879_0',
 'franka-original-1784-direct-candidate-v1':'716034_0',
 'franka-original-1784-direct-candidate-v2':'716039_0',
 'franka-original-1784-direct-audit-v1':'716040_0',
}
ids=','.join(sorted({job.split('_')[0] for job in runs.values()}))
jobs=subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,State,ExitCode,Elapsed,End,NodeList','-n','-P'],text=True)
(meta/'jobs.txt').write_text(jobs)
states={row.split('|')[0].strip():row.split('|')[1].strip() for row in jobs.splitlines() if row.strip()}
(meta/'job-commands.txt').write_text(subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,SubmitLine%12000','-n','-P'],text=True))
closures=[];extras=set()
for name,job in runs.items():
 run=p/name;code=int((run/'returncode.txt').read_text())
 assert states[job]==('FAILED' if job=='715815_0' else 'COMPLETED')
 assert (code!=0)==(job=='715815_0')
 for prefix in ['source','input-source']:
  before=run/(prefix+'.sha256')
  if not before.exists():continue
  assert before.read_bytes()==(run/(prefix+'-after.sha256')).read_bytes()
  for line in before.read_text().splitlines():
   sha,relative=line.split('  ',1)
   assert hashlib.sha256((run/prefix/relative).read_bytes()).hexdigest()==sha,(name,prefix,relative)
   closures.append({'run':name,'identity':prefix,'file':relative,'sha256':sha})
 extras.update(p.glob('*'+job+'.log'))
def report(name):return json.loads((p/name/'results/summary.json').read_text())
prefix=report('franka-original-residual-history-audit-v1')
recovered=report('franka-original-1784-direct-audit-v1')
assert prefix['steps_checked']==1783 and recovered['steps_checked']==1784
assert recovered['independently_verified_recovered_step']==1784 and not recovered['complete_requested_history']
for row in recovered['rows']:
 assert row['body_dynamic_residual_N']<=1e-9 and row['actual_robot_step_effort_residual']<=1e-9
candidate=report('franka-original-1784-direct-candidate-v2')
assert hashlib.sha256(pathlib.Path(candidate['source_candidate']).read_bytes()).hexdigest()==candidate['source_candidate_sha256']
assert candidate['no_newton_or_continuation_steps_taken'] and candidate['final_fraction']==1
native=report('franka-original-history-native-v1')
assert len(native['rows'])==12 and not native['complete_original_330_macro_history']
summary={
 'scope':'Original1783-state force/global-field audits and12 native replays, independently recovered1784th direct candidate at unchanged physical budgets',
 'runs':[{'run':name,'job':job,'state':states[job],'returncode':int((p/name/'returncode.txt').read_text())} for name,job in runs.items()],
 'source_closures':closures,
 'original1783_force_audit':prefix,
 'original1783_global_fields':report('franka-original-history-fields-v1'),
 'actual_native_replays':native,
 'interior_corner_certificate_checks':report('franka-original-history-bounds-v1'),
 'unchanged_saved_direct_candidate':candidate,
 'independent1784_force_state_control_audit':recovered,
 'observables':json.loads((p/'meta/original-history-1783-observables-v1.json').read_text()),
 'live_continuations_excluded':['716041_0','716061_0'],
 'goal_completion':False,
}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt',
 p/'meta/original-history-followup-commands-v1.json',p/'meta/original-history-1783-observables-v1.json'])
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'terminal_runs':len(runs),'source_hash_records':len(closures),'independently_checked_states':1784}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/original-history-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'The original1783-state producer fails at1784 and is not a completed3.3s history. All1783 original controls/forces/states and raw global fields are independently audited;12 actual native replays and7518240 interior/corner certificate samples qualify their declared scopes. The saved direct1784 candidate already satisfies the unchanged original1e-9 physical budgets despite missing the stricter1e-10 Newton target. Re-evaluation takes no Newton/artificial-continuation step; the independent audit checks all1784 physical states. Complete original history, physical contact/geometry/convergence, model/feedback and verified warmed accuracy/cost remain required. Live continuations are excluded.' \
    > "$task_meta/publication.log" 2>&1
python3 research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_meta/verification.json" \
    > "$task_meta/verification.log" 2>&1
cp "$task_meta/verification.json" "$task_dest/verification.json"
python3 - "$task_dest/manifest.json" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);o=json.loads(p.read_text())
print(json.dumps({'files':len(o['files']),'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
PY
