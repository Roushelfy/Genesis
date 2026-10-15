#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/reference-methods-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={
 'u-nonlinear-cached-spatial-1-v1':'715781_0',
 'u-nonlinear-cached-coarse-equivalence-v1':'715792_0',
 'u-nonlinear-rigid-init-spatial-1-v1':'715802_0',
 'u-nonlinear-rigid-init-coarse-equivalence-v1':'715818_0',
 'egg-init-continuation-v3':'714503_0',
}
ids=','.join(sorted({job.split('_')[0] for job in runs.values()}))
jobs=subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,State,ExitCode,Elapsed,End,NodeList','-n','-P'],text=True)
(meta/'jobs.txt').write_text(jobs)
states={row.split('|')[0].strip():row.split('|')[1].strip() for row in jobs.splitlines() if row.strip()}
(meta/'job-commands.txt').write_text(subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,SubmitLine%12000','-n','-P'],text=True))
closures=[];extras=set()
for name,job in runs.items():
 run=p/name;code=int((run/'returncode.txt').read_text())
 expected='TIMEOUT' if job=='714503_0' else ('COMPLETED' if code==0 else 'FAILED')
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
 'scope':'Qualify nonlinear-U reference cache/initial-iterate changes and retain the naturally timed-out generic full-egg attempt',
 'runs':[{'run':name,'job':job,'state':states[job],'returncode':int((p/name/'returncode.txt').read_text())} for name,job in runs.items()],
 'source_closures':closures,
 'nonlinear_cache_equivalence':report('u-nonlinear-cached-coarse-equivalence-v1'),
 'rigid_contact_free_initializer_equivalence':report('u-nonlinear-rigid-init-coarse-equivalence-v1'),
 'egg_initializer_guard':json.loads((p/'egg-init-continuation-v3/solution/initialization_resume_guard.json').read_text()),
 'egg_timeout':'Slurm714503_0 isTIMEOUT despite wrapper returncode0; complete initialization/contact acceptance is not established.',
 'extension_request':json.loads((p/'meta/egg-init-time-extension-v1.json').read_text()),
 'identical_original_inputs_for_new_cached_attempt':json.loads((p/'meta/egg-cached-retry-input-audit-v1.json').read_text()),
 'live_excluded':['715941_0','715802_1','715802_2','715794_0','715815_0','715870_0','715083_3'],
 'goal_completion':False,
}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt',
 p/'meta/matched-spatial-followup-commands-v1.json',p/'meta/egg-init-time-extension-v1.json',
 p/'meta/egg-cached-retry-input-audit-v1.json'])
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'terminal_runs':len(runs),'source_hash_records':len(closures)}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/reference-methods-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'Final nonlinear-U full fields and all744 retained accepted-step force-gate records agree under the reference cache and contact-free rigid initial-iterate changes. Intermediate full fields were not saved, so complete trajectory-field equivalence is not claimed. The generic full-egg attempt ends in actual SlurmTIMEOUT despite wrapper returncode0. Its extension request was denied. The new cached retry and finer nonlinear-U/physical references remain live and unverified. Original full A/B/C, physical tails, load models/feedback and verified warmed costs remain required.' \
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
