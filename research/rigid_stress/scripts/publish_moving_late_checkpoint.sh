#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/moving-reference-late-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={
 'franka-wide-tip-state-v1':'715582_0',
 'franka-spatial-force-modes-v1':'715674_0',
 'franka-spatial-force-modes-v2':'715682_0',
 'franka-original-approach-lift-v1':'715656_0',
 'franka-original-approach-prefix-audit-v1':'715690_0',
 'franka-original-grasp-state-v1':'715689_0',
}
optional={'franka-original-grasp-probes-v1':'715702_0'}
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
for name in ['moving-reference-checkpoint-v1','moving-reference-equivalence-checkpoint-v1']:
 manifest=pathlib.Path('research/rigid_stress/evidence/20261014-physical')/name/'manifest.json'
 dependencies.append({'manifest':str(manifest),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()})
summary={
 'scope':'Retain original-history failure and865 independently audited accepted states; quantify spatial force precision without accepting failed fields',
 'runs':[{'run':name,'job':job,'state':states[job],'returncode':int((p/name/'returncode.txt').read_text())} for name,job in runs.items()],
 'source_closures':closures,'dependencies':dependencies,
 'original_accepted_prefix_audit':report('franka-original-approach-prefix-audit-v1'),
 'spatial_modes':report('franka-spatial-force-modes-v1'),
 'spatial_stencil_precision':report('franka-spatial-force-modes-v2'),
 'original_grasp_before_retry':json.loads((p/'franka-original-grasp-state-v1/results/before_recovery.json').read_text()),
 'full_original_history_failure':'Step866 at1.0825s rejects large body/joint residuals after865 accepted states; complete3.3s history remains unfinished.',
 'spatial_retry_failure':'Wide coordinates do not complete the mapped spatial force gate; failure serialization also rejects a longdouble report. Saved failed arrays and logs are retained.',
 'unverified_separate_work':['715083_3','715021_0','714462_1','714503_0','714840_0'],
 'goal_completion':False,
}
if 'franka-original-grasp-probes-v1' in runs:
 summary['original_grasp_small_step_probes']=report('franka-original-grasp-probes-v1')
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt',p/'meta/moving-reference-late-commands.json',
              p/'meta/wide-motion-state-v1-array.sh',p/'meta/wide-motion-full-v1-array.sh',
              p/'meta/predicted-motion-full-v1-array.sh',p/'meta/extended-motion-dependency-journal.json'])
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'terminal_runs':len(runs),'source_hash_records':len(closures)}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/moving-reference-late-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'The original3.3s attempt rejects step866 after865 accepted states, all independently audited. The same-state retry also fails. Spatial diagnostics quantify strong contact-arithmetic sensitivity but do not establish its sole cause or a converged contact patch. Complete original history, physical spatial/contact/geometry tails, remaining model/feedback operating ranges and verified accuracy/cost remain incomplete. No force budget or physical parameter is relaxed. Only terminal cases are published.' \
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
