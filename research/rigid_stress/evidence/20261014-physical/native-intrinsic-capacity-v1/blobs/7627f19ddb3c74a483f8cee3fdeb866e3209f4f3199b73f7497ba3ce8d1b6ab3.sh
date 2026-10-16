#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/egg-paired-contact-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={'egg-paired-polyhedron-level2-contact-v2':'716487_0',
      'egg-paired-polyhedron-level2-contact-v2-audit':'716488_0',
      'egg-paired-cad-level2-contact-v2':'716489_0',
      'egg-paired-cad-level2-contact-v2-audit':'716490_0'}
ids=','.join(job.split('_')[0] for job in runs.values())
jobs=subprocess.check_output(['sacct','-n','-P','-X','-j',ids,'--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList'],text=True)
(meta/'jobs.txt').write_text(jobs)
(meta/'job-commands.txt').write_text(subprocess.check_output(['sacct','-n','-P','-X','-j',ids,'--format=JobID%40,SubmitLine%16000'],text=True))
states={line.split('|')[0].strip():line.split('|')[1].strip() for line in jobs.splitlines() if line.strip()}
records=[];identities=[];results={}
for name,job in runs.items():
 run=p/name;assert states[job]=='COMPLETED' and int((run/'returncode.txt').read_text())==0
 for kind in ['source','input-source']:
  before=run/(kind+'.sha256')
  assert before.read_bytes()==(run/(kind+'-after.sha256')).read_bytes()
  for line in before.read_text().splitlines():
   sha,path=line.split('  ',1)
   assert hashlib.sha256((run/kind/path).read_bytes()).hexdigest()==sha
   identities.append({'run':name,'identity':kind,'file':path,'sha256':sha})
 result=json.loads((run/'results/summary.json').read_text());results[name]=result
 if name.endswith('-audit'):
  assert result['states_checked']==len(result['rows'])==5
  assert all(row['complete_body_dynamic_residual_N']<=1e-9 and row['complete_mapped_dynamic_residual_N']<=1e-9 and row['force_pullback_max_difference_N']==0 for row in result['rows'])
 else:
  assert len(result['closure'])==64 and len(result['steps'])==5
  assert all(row['complete_held_residual_N']<=1e-9 for row in result['closure'])
 records.append({'run':name,'job':job,'scheduler_state':states[job],'returncode':0})
summary={'scope':'Two1920-tet actual original-control contact references on matched-connectivity CAD/polyhedron geometry with all five dynamic states independently audited',
 'runs':records,'source_hash_records':identities,'results':results,'goal_completion':False,
 'limitations':'Material/density, actual fingers, physical initializer points/components, friction, dhat, jaw actuation and dt are fixed. Geometry and represented mass differ explicitly; this comparison does not isolate pure geometry stress error. All64 held closure residuals are producer records, not independent audits of every closure state. Five independently reconstructed dynamic states do not establish a full original history, a contact/geometry tail, remaining model/feedback accuracy or verified warmed cost. Finer live producers are excluded.'}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
extras=[meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt',p/'meta/egg-paired-fine-contact-commands-v1.json']
for job in runs.values():extras.extend(p.glob('*'+job+'.log'))
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in extras))
print(json.dumps({'completed_runs':len(runs),'source_identity_records':len(identities),'independently_audited_dynamic_states':10}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/egg-paired-contact-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'Two declared1920-tet geometry families,64 freshly solved held closure targets each and10 independently reconstructed original dynamic states. Raw peaks, full-volume bounds and contact diagnostics remain explicit. Fixed density causes different represented mass on CAD/polyhedral geometry. These short references do not establish geometry/contact convergence, complete application/history, models/feedback or verified accuracy/cost. Finer live runs are excluded.' \
    > "$task_meta/publication.log" 2>&1
python3 research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_meta/verification.json" \
    > "$task_meta/verification.log" 2>&1
