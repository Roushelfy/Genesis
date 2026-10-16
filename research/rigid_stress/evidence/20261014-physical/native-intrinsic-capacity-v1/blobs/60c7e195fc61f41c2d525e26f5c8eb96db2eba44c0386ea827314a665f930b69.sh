#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/matched-spatial-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={
 'franka-cached-newton-spatial-2-v1':'714840_0',
 'franka-cached-three-level-fields-v1':'715757_0',
 'franka-matched-recovered-spatial-v1':'715908_0',
 'franka-matched-recovered-spatial-v2':'715913_0',
 'franka-matched-recovered-spatial-v3':'715933_0',
 'franka-matched-spatial-figure-v1':'715910_0',
 'franka-matched-spatial-figure-v2':'715916_0',
 'franka-matched-spatial-figure-v3':'715919_0',
 'franka-precise-prefix-equivalence-v1':'715926_0',
}
ids=','.join(sorted({job.split('_')[0] for job in runs.values()}))
jobs=subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,State,ExitCode,Elapsed,End,NodeList','-n','-P'],text=True)
(meta/'jobs.txt').write_text(jobs)
states={row.split('|')[0].strip():row.split('|')[1].strip() for row in jobs.splitlines() if row.strip()}
(meta/'job-commands.txt').write_text(subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,SubmitLine%12000','-n','-P'],text=True))
closures=[];extras=set()
for name,job in runs.items():
 run=p/name;code=int((run/'returncode.txt').read_text())
 expected='COMPLETED'
 assert code==0
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
observation=p/'precise-spatial-prefix-observation-v1'
observed=json.loads((observation/'observation.json').read_text())
assert observed['complete_producer_history'] is False
for row in observed['files']:
 assert hashlib.sha256(pathlib.Path(row['observed']).read_bytes()).hexdigest()==row['sha256']
 assert hashlib.sha256(pathlib.Path(row['source']).read_bytes()).hexdigest()==row['sha256']
summary={
 'scope':'Retain three-level uniform nonconvergence, independently balanced matched32.5ms narrow-contact fields and25-state precision equivalence with reviewed editable maps',
 'runs':[{'run':name,'job':job,'state':states[job],'returncode':int((p/name/'returncode.txt').read_text())} for name,job in runs.items()],
 'source_closures':closures,
 'uniform_three_level_fields':report('franka-cached-three-level-fields-v1'),
 'matched_single_time_fields':report('franka-matched-recovered-spatial-v3'),
 'original25_saved_states_numerical_equivalence':report('franka-precise-prefix-equivalence-v1'),
 'read_only_live_prefix_observation':observed,
 'figure_rendered_qa':json.loads((p/'meta/matched-spatial-figure-qa-v1.json').read_text()),
 'egg_extension_request':json.loads((p/'meta/egg-init-time-extension-v1.json').read_text()),
 'live_producers_excluded':['715815_0','715817_0','715856_0','715878_0','715879_0','715870_0','715877_0','715781_0','715792_0','715794_0','715802_0','715802_1','715802_2','715818_0','715083_3','714503_0'],
 'goal_completion':False,
}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt',p/'meta/matched-spatial-commands-v1.json',
 p/'meta/matched-spatial-figure-qa-v1.json',p/'meta/egg-init-time-extension-v1.json'])
extras.update(path for path in observation.rglob('*') if path.is_file())
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'terminal_runs':len(runs),'source_hash_records':len(closures)}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/matched-spatial-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'Uniform480/3840/30720-tet fields retain49.32% final tensor change despite0.00866% contact-force change. Matched32.5ms narrow-contact fields retain23.94% tensor/49.66% energy changes and2.044GPa raw peak with18 active nodes. All three single states satisfy independently reconstructed original1e-9N body/mapped force gates. First25 original saved states agree to4.86e-11 in full corner tensors, but the complete80-state producer remains live; the immutable observation is not a completed producer. Editable matched maps pass rendered QA. Complete physical geometry/contact/time/model/feedback and verified warmed cost remain open. Live producers are excluded.' \
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
