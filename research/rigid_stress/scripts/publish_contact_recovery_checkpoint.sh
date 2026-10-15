#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/contact-recovery-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={
 'franka-original-contact-continuation-v1':'715762_0',
 'franka-original-recovered-audit-v1':'715772_0',
 'franka-original-approach-lift-resumed-v1':'715775_0',
 'u-cached-newton-spatial-4-v1':'715021_0',
 'u-four-level-tail-audit-v1':'715776_0',
 'franka-cached-three-level-tractions-v1':'715758_0',
 'u-nonlinear-cached-spatial-2-v1':'715781_1',
 'u-nonlinear-cached-spatial-3-v1':'715781_2',
}
ids=','.join(sorted({job.split('_')[0] for job in runs.values()}))
jobs=subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,State,ExitCode,Elapsed,End,NodeList','-n','-P'],text=True)
(meta/'jobs.txt').write_text(jobs)
states={row.split('|')[0].strip():row.split('|')[1].strip() for row in jobs.splitlines() if row.strip()}
(meta/'job-commands.txt').write_text(subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,SubmitLine%12000','-n','-P'],text=True))
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
summary={
 'scope':'Recover original step866 at its unchanged final equation, audit all866 physical states and retain fourth-level U tails and later natural failures',
 'runs':[{'run':name,'job':job,'state':states[job],'returncode':int((p/name/'returncode.txt').read_text())} for name,job in runs.items()],
 'source_closures':closures,
 'original_step_recovery':report('franka-original-contact-continuation-v1'),
 'independent_original_866_states':report('franka-original-recovered-audit-v1'),
 'four_level_u':report('u-four-level-tail-audit-v1'),
 'three_level_unsmoothed_traction':report('franka-cached-three-level-tractions-v1'),
 'later_history_failure':'Resumed original history accepts867 and868, then rejects869 at unchanged body/joint gates. Globalized full-history work is separate and live.',
 'nonlinear_startup_failures':'Finer pad grids7/9 reject an inverted initial Newton iterate during the contact-free boundary-only approach. No converged nonlinear contact result is claimed for those attempts.',
 'live_work_excluded':['715757_0','715800_0','715781_0','715792_0','715794_0','715802_0','715802_1','715802_2','715083_3','714503_0','714462_1'],
 'goal_completion':False,
}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt',p/'meta/contact-recovery-next-commands-v1.json',
 p/'meta/nonlinear-u-cached-spatial-v1-array.sh',p/'meta/nonlinear-u-rigid-init-v1-array.sh',
 p/'fine-matrix-mesh-u-v1/meshes/h0.000875.npz',p/'fine-matrix-mesh-u-v1/meshes/h0.000875.json'])
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'terminal_runs':len(runs),'source_hash_records':len(closures)}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/contact-recovery-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'Original step866 is independently verified at the original dt/physics/force gates. Pseudo-timesteps are solver iterates, never physical-history states. Direct continuation rejects869; complete original approach/grasp/lift remains unfinished. Four-level U tensors still change7.38%; raw peak stability does not establish full-field convergence. Three egg meshes retain two active nodes and unresolved concentration. Finer nonlinear startup failures are retained. Live producers are excluded. Complete A/B/C, geometry/contact/model/feedback and verified warmed accuracy/cost remain required.' \
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
