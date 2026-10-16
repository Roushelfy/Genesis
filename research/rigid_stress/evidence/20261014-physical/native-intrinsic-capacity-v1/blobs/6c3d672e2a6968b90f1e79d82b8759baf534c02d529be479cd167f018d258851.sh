#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/moving-reference-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={
 'franka-wide-state-middle-v1':'715396_0',
 'franka-wide-state-coarse-v1':'715396_1',
 'franka-wide-state-fine-v1':'715396_2',
 'franka-captured-motion-fem-extended-v1':'715190_0',
 'franka-wide-motion-middle-v1':'715411_1',
 'franka-predicted-motion-coarse-v1':'715506_0',
 'franka-predicted-motion-middle-v1':'715506_1',
 'franka-predicted-motion-fine-v1':'715506_2',
 'franka-predicted-temporal-audit-v1':'715592_0',
 'franka-predicted-feedback-audit-v1':'715593_0',
 'franka-predicted-feedback-audit-v2':'715619_0',
 'franka-predicted-original-equivalence-v1':'715594_0',
 'franka-predicted-middle-equivalence-v1':'715623_0',
 'franka-predicted-native-v1':'715621_0',
 'franka-predicted-initial-motion-v1':'715622_0',
 'franka-predicted-initial-motion-v2':'715630_0',
 'franka-original-protocol-v1':'715620_0',
 'franka-tip-coupled-15-v1':'715083_2',
 'franka-original-ground-prefix-v1':'715639_0',
 'franka-original-robot-state-v1':'715642_0',
 'franka-original-robot-state-v2':'715643_0',
 'franka-original-ground-prefix-v2':'715645_0',
 'franka-original-ground-prefix-audit-v1':'715654_0',
}
optional={
 'franka-wide-motion-coarse-v1':'715411_0',
 'franka-wide-motion-fine-v1':'715411_2',
 'franka-wide-feedback-audit-v1':'715425_0',
 'franka-wide-tip-state-v1':'715582_0',
}
other={'715192','715324','715325','715424','715465','715624','715656','715100','715102'}
ids=','.join(sorted({job.split('_')[0] for job in [*runs.values(),*optional.values()]}|other))
jobs=subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,State,ExitCode,Elapsed,End,NodeList','-n','-P'],text=True)
(meta/'jobs.txt').write_text(jobs)
states={row.split('|')[0].strip():row.split('|')[1].strip() for row in jobs.splitlines() if row.strip()}
commands=subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,SubmitLine%12000','-n','-P'],text=True)
(meta/'job-commands.txt').write_text(commands)
for name,job in optional.items():
 if states.get(job) in {'COMPLETED','FAILED'}:runs[name]=job
closures=[];extras=set()
for name,job in runs.items():
 run=p/name;code=int((run/'returncode.txt').read_text())
 assert states[job]==('COMPLETED' if code==0 else 'FAILED'),(name,states[job],code)
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
dependencies=[]
for name in ['captured-motion-checkpoint-v1','motion-failure-supplement-v1','contact-limit-checkpoint-v1']:
 manifest=pathlib.Path('research/rigid_stress/evidence/20261014-physical')/name/'manifest.json'
 dependencies.append({'manifest':str(manifest),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()})
summary={
 'scope':'Moving time-step/feedback references, qualified Newton initial guesses and original ground-supported input/prefix mechanics',
 'runs':[{'run':name,'job':job,'state':states[job],'returncode':int((p/name/'returncode.txt').read_text())} for name,job in runs.items()],
 'source_closures':closures,'dependencies':dependencies,
 'temporal_audit':report('franka-predicted-temporal-audit-v1'),
 'moving_feedback_audit':report('franka-predicted-feedback-audit-v2'),
 'all_original_states':report('franka-predicted-original-equivalence-v1'),
 'complete_middle_equivalence':report('franka-predicted-middle-equivalence-v1'),
 'initial_motion_sensitivity':report('franka-predicted-initial-motion-v2'),
 'native_replay':report('franka-predicted-native-v1'),
 'original_observer_equivalence':json.loads((p/'franka-original-protocol-v1/observer-equivalence.json').read_text()),
 'original_prefix_audit':report('franka-original-ground-prefix-audit-v1'),
 'original_robot_state_recovery':report('franka-original-robot-state-v2'),
 'dependency_corrections':json.loads((p/'meta/extended-motion-dependency-journal.json').read_text()),
 'excluded_unfinished_jobs':[job for job,state in states.items() if state in {'RUNNING','PENDING'}],
 'retained_failures':['Extended-force-only run rejects step59 after58 accepted states.',
 'Level15 spatial run rejects step26 above its unchanged1e-9N budget.',
 'Feedback v1 fails JSON serialization after physical checks; v2 converts report scalars and retains raw NPZ fields.',
 'Initial-motion v1 uses incompatible linear/nonlinear quadrature weights; v2 uses the pinned nonlinear quadrature.',
 'Original prefix v1 fails its actual robot-step gate at step3; actual-response Newton v2 completes the declared80-step prefix.',
 'Original robot-state v1 startup rejects an undeclared copied MJCF argument; v2 pins and verifies the copied MJCF.'],
 'goal_completion':False,
}
if 'franka-wide-tip-state-v1' in runs:
 summary['spatial_same_state_diagnostic']=report('franka-wide-tip-state-v1')
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt',p/'meta/moving-reference-commands.json',
              p/'meta/wide-motion-state-v1-array.sh',p/'meta/wide-motion-full-v1-array.sh',
              p/'meta/predicted-motion-full-v1-array.sh',p/'meta/extended-motion-dependency-journal.json'])
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'terminal_runs':len(runs),'source_hash_records':len(closures)}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/moving-reference-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'Moving0.1s holds and the original0.1s ground-supported prefix pass scoped independent gates. The original3.3s approach/grasp/lift remains separate unverified work. Physical spatial/contact/geometry tails, complete history, remaining load-model sensitivity, feedback operating ranges and verified warmed accuracy/cost remain incomplete. Exact scientific and startup failures are retained; no force gate is relaxed. Newton-count observations are reference-only and are not native warmed cost claims. Only terminal cases are published.' \
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
