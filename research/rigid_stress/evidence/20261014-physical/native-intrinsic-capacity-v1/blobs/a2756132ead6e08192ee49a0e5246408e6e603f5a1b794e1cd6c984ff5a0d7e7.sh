#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/captured-motion-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={
 'franka-captured-motion-fem-pilot-v1':'715129_0',
 'franka-captured-motion-rigid-pilot-v1':'715129_1',
 'franka-captured-motion-fem-pilot-v2':'715142_0',
 'franka-captured-motion-rigid-pilot-v2':'715142_1',
 'franka-com-velocity-capture-v1':'715153_0',
 'franka-captured-motion-pilot-audit-v1':'715154_0',
 'franka-captured-motion-fem-v1':'715175_0',
 'franka-captured-motion-rigid-v1':'715175_1',
 'franka-captured-motion-zero-control-audit-v1':'715182_0',
 'franka-tip-coupled-12-v1':'715083_1',
 'franka-tip-first-tractions-v1':'715124_0',
 'franka-invalid-com-motion-audit-v1':'715186_0',
 'franka-motion-force-precision-v1':'715187_0',
 'franka-captured-motion-dt-coarse-v1':'715177_0',
}
ids=','.join(sorted({job.split('_')[0] for job in runs.values()} | {'715178','715180','715181','715184'}))
jobs=subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,State,ExitCode,Elapsed,End,NodeList','-n','-P'],text=True)
(meta/'jobs.txt').write_text(jobs)
commands=subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,SubmitLine%12000','-n','-P'],text=True)
(meta/'job-commands.txt').write_text(commands)
states={row.split('|')[0].strip():row.split('|')[1].strip() for row in jobs.splitlines() if row.strip()}
closures=[];extras=set()
for name,job in runs.items():
 run=p/name
 code=int((run/'returncode.txt').read_text())
 assert states[job]==('COMPLETED' if code==0 else 'FAILED'),(name,states[job],code)
 for prefix in ['source','input-source']:
  before=run/(prefix+'.sha256')
  if not before.exists():continue
  assert before.read_bytes()==(run/(prefix+'-after.sha256')).read_bytes()
  for line in before.read_text().splitlines():
   expected,relative=line.split('  ',1)
   source=run/prefix/relative
   assert hashlib.sha256(source.read_bytes()).hexdigest()==expected,source
   closures.append({'run':name,'identity':prefix,'file':relative,'sha256':expected})
 extras.update(p.glob('*'+job+'.log'))
def report(name):return json.loads((p/name/'results/summary.json').read_text())
dependencies=[]
for name in ['contact-limit-checkpoint-v1','feedback-time-checkpoint-v1','articulated-u-checkpoint-v1']:
 manifest=pathlib.Path('research/rigid_stress/evidence/20261014-physical')/name/'manifest.json'
 dependencies.append({'manifest':str(manifest),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()})
summary={
 'scope':'Correct captured COM velocity, qualify moving pilots, diagnose a full-hold arithmetic failure and retain unresolved concentrated contact',
 'runs':[{'run':name,'job':job,'state':states[job],'returncode':int((p/name/'returncode.txt').read_text())} for name,job in runs.items()],
 'source_closures':closures,'dependencies':dependencies,
 'observer_equivalence':json.loads((p/'franka-com-velocity-capture-v1/observer-equivalence.json').read_text()),
 'motion_pilot_audit':report('franka-captured-motion-pilot-audit-v1'),
 'zero_motion_audit':report('franka-captured-motion-zero-control-audit-v1'),
 'same_rejected_state_precision':report('franka-motion-force-precision-v1'),
 'moving_full_hold_failure':'FEM producer715175_0 completes48 accepted states and naturally rejects step49 at1.00159372e-9N above the original1e-9N gate. Rigid producer715175_1 completes80. No complete80-step moving feedback result is established.',
 'moving_coarse_hold_failure':'The2.5ms FEM producer715177_0 completes22 accepted states and rejects step23 at1.01281709e-9N. It does not provide a complete temporal comparison.',
 'tip_tractions':report('franka-tip-first-tractions-v1'),
 'invalid_input_diagnostics':['franka-captured-motion-fem-pilot-v1','franka-captured-motion-rigid-pilot-v1'],
 'invalid_input_reason':'These first pilots interpret get_vel(relative=False), measured at the link origin, as COM velocity. Solver completion does not validate that initial physical input. Corrected v2 pilots and full holds transport origin velocity using the captured world COM and angular velocity.',
 'cancelled_dependency_journal':json.loads((p/'meta/captured-motion-dependency-journal.json').read_text()),
 'excluded_unfinished_jobs':['715083_2','715083_3','715100_0','715102_0','715177_1','715190_0','715192_0','715021_0','714462_1','714503_0','714840_0'],
 'goal_completion':False,
}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt',p/'meta/captured-motion-full-v1-array.sh',p/'meta/captured-motion-temporal-v1-array.sh',p/'meta/captured-motion-commands.json',p/'meta/captured-motion-dependency-journal.json'])
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'terminal_runs':len(runs),'source_identities':len(closures)}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/captured-motion-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'Captured velocities are independently verified at their correct physical reference points. Initial-velocity-only holds start from solved closures and do not reproduce original approach/lift/support history. Contact stress/patch convergence remains unresolved. First v1 motion pilots contain an explicitly invalid COM interpretation and are retained as diagnostics. Only terminal runs are published; finer and temporal studies remain unverified. The complete original elasticity/model/contact/geometry/history/feedback/accuracy-cost goal remains active.' \
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
