#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/stencil-precision-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={
 'franka-local-contact-precision-v1':'715836_0',
 'franka-precise-stencil-recovery-v1':'715849_0',
 'franka-precise-stencil-state-audit-v1':'715866_0',
 'franka-original-history-fields-pilot-v1':'715842_0',
 'franka-original-history-fields-pilot-v2':'715853_0',
 'franka-original-history-native-pilot-v1':'715855_0',
 'franka-original-history-bounds-pilot-v1':'715858_0',
 'franka-original-history-bounds-pilot-v2':'715862_0',
 'franka-original-history-bounds-pilot-v3':'715886_0',
 'franka-sparse-spatial-2-v1':'714462_1',
}
ids=','.join(sorted({job.split('_')[0] for job in runs.values()}))
jobs=subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,State,ExitCode,Elapsed,End,NodeList','-n','-P'],text=True)
(meta/'jobs.txt').write_text(jobs)
states={row.split('|')[0].strip():row.split('|')[1].strip() for row in jobs.splitlines() if row.strip()}
(meta/'job-commands.txt').write_text(subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,SubmitLine%12000','-n','-P'],text=True))
closures=[];extras=set()
for name,job in runs.items():
 run=p/name;code=int((run/'returncode.txt').read_text())
 expected='TIMEOUT' if job=='714462_1' else ('COMPLETED' if code==0 else 'FAILED')
 assert states[job]==expected,(name,states[job],code)
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
 'scope':'Independently recover8433-tet original step26 and qualify original ground-supported startup stress fields/bounds/native replays',
 'runs':[{'run':name,'job':job,'state':states[job],'returncode':int((p/name/'returncode.txt').read_text())} for name,job in runs.items()],
 'source_closures':closures,
 'stencil_arithmetic_probe':report('franka-local-contact-precision-v1'),
 'recovered_spatial_step':report('franka-precise-stencil-recovery-v1'),
 'independent_spatial_step_audit':report('franka-precise-stencil-state-audit-v1'),
 'original80_field_audit':report('franka-original-history-fields-pilot-v2'),
 'actual_native_cuda_replays':report('franka-original-history-native-pilot-v1'),
 'independent_interior_checks':report('franka-original-history-bounds-pilot-v3'),
 'retained_failed_reports':['715842_0','715858_0'],
 'uncached_timeout':'Slurm714462_1 isTIMEOUT despite wrapper returncode0. It completes49 held targets and saves no dynamic states or complete summary. No successful fine original/cached trajectory equivalence is claimed.',
 'live_excluded':['715757_0','715815_0','715817_0','715856_0','715878_0','715879_0','715870_0','715877_0','715781_0','715792_0','715794_0','715802_0','715802_1','715802_2','715818_0','715083_3','714503_0'],
 'goal_completion':False,
}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt',p/'meta/stencil-precision-commands-v1.json'])
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'terminal_runs':len(runs),'source_hash_records':len(closures)}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/stencil-precision-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'The precision-aware8433-tet step26 satisfies independent original1e-9N complete force gates. Full80-step re-solve remains live. Original startup80-state field bounds,397440 independent interior/corner checks and four actual native CUDA nodal replays qualify one coarse physical setting. Failed report attempts and a natural uncached TIMEOUT are retained. Original full3.3s history, physical spatial/contact/geometry tails, remaining model/feedback ranges and verified warmed accuracy/cost remain incomplete. Live jobs are excluded.' \
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
