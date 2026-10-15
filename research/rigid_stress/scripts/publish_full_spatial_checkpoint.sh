#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/full-spatial-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={
 'franka-precise-spatial-history-v1':'715870_0',
 'franka-precise-spatial-history-audit-v1':'715877_0',
 'franka-precise-spatial-fields-export-v1':'715965_0',
 'franka-precise-spatial-full-tail-v1':'715966_0',
 'franka-resolved-linearization-v1':'715999_0',
 'franka-resolved-linearization-native-0-v1':'716010_0',
 'franka-resolved-linearization-native-1-v1':'716010_1',
 **{f'franka-resolved-linearization-native-{i}-v2':f'716014_{i}' for i in range(3)},
}
ids=','.join(sorted({job.split('_')[0] for job in runs.values()}))
jobs=subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,State%50,ExitCode,Elapsed,End,NodeList','-n','-P'],text=True)
(meta/'jobs.txt').write_text(jobs)
states={row.split('|')[0].strip():row.split('|')[1].strip().split()[0] for row in jobs.splitlines() if row.strip()}
(meta/'job-commands.txt').write_text(subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,SubmitLine%12000','-n','-P'],text=True))
closures=[];extras=set();records=[]
for name,job in runs.items():
 run=p/name;code=int((run/'returncode.txt').read_text())
 if job=='716010_0':
  assert states[job]=='FAILED' and code!=0
 elif job=='716010_1':
  assert states[job]=='CANCELLED'
 elif job.startswith('716014_'):
  assert states[job] in {'COMPLETED','FAILED'}
  assert (states[job]=='COMPLETED')==(code==0)
 else:
  assert states[job]=='COMPLETED' and code==0
 for prefix in ['source','input-source']:
  before=run/(prefix+'.sha256')
  if not before.exists():continue
  assert before.read_bytes()==(run/(prefix+'-after.sha256')).read_bytes()
  for line in before.read_text().splitlines():
   sha,relative=line.split('  ',1)
   assert hashlib.sha256((run/prefix/relative).read_bytes()).hexdigest()==sha,(name,prefix,relative)
   closures.append({'run':name,'identity':prefix,'file':relative,'sha256':sha})
 records.append({'run':name,'job':job,'state':states[job],'returncode':code})
 extras.update(p.glob('*'+job+'.log'))
def report(name):return json.loads((p/name/'results/summary.json').read_text())
native=[]
for i in range(3):
 run=p/f'franka-resolved-linearization-native-{i}-v2'
 if states[f'716014_{i}']=='COMPLETED':
  result=report(run.name)
  for path,sha in result['input_sha256'].items():assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==sha
  assert len(result['rows'])==2
  assert all(row['native_valid_including_six_gauge_rows'] and row['independent_complete_budget_fraction']<=1 for row in result['rows'])
  native.append(result)
 else:
  assert not (run/'results/summary.json').exists()
  native.append({'case':i,'qualified':False,'progress':json.loads((run/'results/progress.json').read_text())})
history=report('franka-precise-spatial-history-audit-v1')
assert history['states_checked']==len(history['rows'])==80
assert all(row['complete_body_dynamic_residual_N']<=1e-9 and row['complete_mapped_dynamic_residual_N']<=1e-9 for row in history['rows'])
journal=json.loads((p/'meta/full-spatial-followup-commands-v1.json').read_text())
for path,sha in journal['classification_source_sha256'].items():assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==sha
summary={
 'scope':'Complete independently audited80-state fine spatial history, three-level full-history nonconvergence and resolved-load model/numerical-error separation',
 'runs':records,
 'source_closures':closures,
 'fine_history_independent_audit':history,
 'three_level_complete_history_fields':report('franka-precise-spatial-full-tail-v1'),
 'resolved_linearization':report('franka-resolved-linearization-v1'),
 'native_resolved_load_replays':native,
 'cancelled_pending_task_without_runtime':{'job':'716010_2','state':states['716010_2'],'runtime_created':False},
 'history_completeness_metadata_qualification':json.loads((p/'original-history-status-qualification-v1/summary.json').read_text()),
 'failed_comparison':'716010_0 uses inconsistent comparison quadrature;716010_1/2 were cancelled after diagnosis. The corrected716014 replays use identical order-six reference points. Failed/cancelled attempts are retained and are not accepted results.',
 'goal_completion':False,
}
assert states['716010_2']=='CANCELLED' and not (p/'franka-resolved-linearization-native-2-v1').exists()
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt',
 p/'meta/full-spatial-followup-commands-v1.json',p/'meta/current-reference-submit-lines-v1.txt',
 p/'original-history-status-qualification-v1/summary.json',p/'meta/original-history-status-qualification-v1.log'])
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'terminal_runs':len(runs),'source_hash_records':len(closures),'native_qualified_cases':sum('rows' in row for row in native)}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/full-spatial-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'All80 precision-aware8433-tet states pass independently reconstructed original force gates. Complete480/3577/8433 histories retain23.94% tensor/49.66% energy change and2.023GPa final raw peak. Independent resolved-load linear FEM differs5.17% in volume-weighted tensor and3.00% in peak at32.5ms. Native replay outcomes retain explicit original complete residuals and independently reconstructed fields, including failed/cancelled attempts. These do not establish physical contact/geometry convergence, full original history, remaining model/feedback comparisons or warmed cost at verified physical accuracy. Live producers are excluded.' \
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
