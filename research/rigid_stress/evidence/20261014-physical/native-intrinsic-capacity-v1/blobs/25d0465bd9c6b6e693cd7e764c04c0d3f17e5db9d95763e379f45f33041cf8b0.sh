#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/egg-spatial-geometry-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import datetime,hashlib,json,pathlib,shutil,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={
 'franka-four-level-step5-fields-v1':'716451_0',
 'franka-four-level-step5-linearization-v1':'716452_0',
 **{f'franka-four-level-step5-native-{i}-v1':f'{716453+i}_0' for i in range(4)},
 'franka-four-level-step5-native-2-diagnostic-v2':'716476_0',
 'franka-four-level-step5-native-3-diagnostic-v2':'716477_0',
 'egg-continuum-geometry-v1':'716475_0',
 'egg-paired-cad-polyhedron-geometry-v1':'716481_0',
 'egg-paired-polyhedron-level2-contact-v1':'716482_0',
 'egg-paired-cad-level2-contact-v1':'716484_0',
 'egg-geometry-spatial-figure-v1':'716486_0',
 'egg-geometry-spatial-figure-v2':'716492_0',
}
failed={'716455_0','716456_0','716476_0','716477_0','716482_0','716484_0'}
ids=','.join(job.split('_')[0] for job in runs.values())
jobs=subprocess.check_output(['sacct','-n','-P','-X','-j',ids,'--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList'],text=True)
(meta/'jobs.txt').write_text(jobs)
(meta/'job-commands.txt').write_text(subprocess.check_output(['sacct','-n','-P','-X','-j',ids,'--format=JobID%40,SubmitLine%16000'],text=True))
states={line.split('|')[0].strip():line.split('|')[1].strip() for line in jobs.splitlines() if line.strip()}
closures=[];records=[];extras=set();external=[]
observation=meta/'external-input-observations'
for name,job in runs.items():
 run=p/name;code=int((run/'returncode.txt').read_text())
 assert states[job]==('FAILED' if job in failed else 'COMPLETED'),(name,states[job])
 assert (code!=0)==(job in failed)
 for kind in ['source','input-source']:
  before=run/(kind+'.sha256')
  if not before.exists():continue
  assert before.read_bytes()==(run/(kind+'-after.sha256')).read_bytes(),(name,kind)
  for line in before.read_text().splitlines():
   sha,relative=line.split('  ',1);rel=pathlib.Path(relative)
   snapshot=run/kind/(rel.name if rel.is_absolute() else rel)
   assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==sha,(name,kind,relative)
   closures.append({'run':name,'identity':kind,'file':relative,'sha256':sha})
 records.append({'run':name,'job':job,'scheduler_state':states[job],'returncode':code,'natural_EXIT_records_present':True})
 extras.update(p.glob('*'+job+'.log'))
def report(name,folder='results'):
 return json.loads((p/name/folder/'summary.json').read_text())
def add_input(path,sha=None):
 path=pathlib.Path(path)
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 if sha is not None:assert digest==sha,path
 if path.is_relative_to(p.parent):extras.add(path);return
 observation.mkdir(exist_ok=True)
 target=observation/(digest+path.suffix)
 if not target.exists():shutil.copy2(path,target)
 assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
 external.append({'original':str(path),'observation':str(target),'sha256':digest,'observed_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Post-termination byte-identical input observation, not a producer source capture'})
 extras.add(target)
comparison=report('franka-four-level-step5-fields-v1')
linear=report('franka-four-level-step5-linearization-v1')
continuum=report('egg-continuum-geometry-v1')
paired=report('egg-paired-cad-polyhedron-geometry-v1')
assert [r['tetrahedra'] for r in comparison['rows']]==[480,3577,8433,30298]
assert all(r['time_s']==0.00625 and r['complete_body_dynamic_residual_N']<=1e-9 and r['complete_mapped_dynamic_residual_N']<=1e-9 for r in comparison['rows'])
assert len(linear['rows'])==4 and len(paired['rows'])==8
assert all(r['original_initializer_all_anchors_retained'] for r in paired['rows'])
native=[]
for i in range(4):
 name=f'franka-four-level-step5-native-{i}-v1'
 path=p/name/'results/progress.json';row={'run':name,'accepted':i<2,'progress':json.loads(path.read_text())}
 if i<2:row['summary']=report(name)
 native.append(row)
diagnostics=[report(f'franka-four-level-step5-native-{i}-diagnostic-v2') for i in [2,3]]
assert all(not s['all_existing_gates_passed'] and len(s['rows'])==2 for s in diagnostics)
assert all(r['complete_equilibrium_passed'] and r['independent_tensor_passed'] for s in diagnostics for r in s['rows'])
for summary in [comparison,linear,continuum,paired,*diagnostics]:
 for path,sha in summary['input_sha256'].items():add_input(path,sha)
for row in comparison['rows']:
 source=pathlib.Path(row['run'])
 for name in ['step0004.npz','step0005.npz','configuration.json','closure_state.npz','reference_geometry.npz','contact_geometry.npz','contact_node_map.npz']:
  add_input(source/'results'/name)
 for path in (source/'source').rglob('*'):
  if path.is_file() and '__pycache__' not in path.parts:extras.add(path)
 for name in ['source.sha256','source-after.sha256','returncode.txt']:extras.add(source/name)
figure=report('egg-geometry-spatial-figure-v2','figure')
qa=p/'meta/egg-geometry-spatial-figure-v2-rendered-QA.json'
assert json.loads(qa.read_text())['visual_QA_passed']
extras.update([qa,p/'meta/egg-spatial-geometry-commands-v1.json',p/'meta/four-level-step5-commands-v1.json',p/'meta/egg-continuum-geometry-spec-v1.json'])
summary={'scope':'Four matched6.25ms spatial fields, independent linear-model discrepancy, exact continuum geometry qualification and paired CAD/polyhedron meshes',
 'runs':records,'source_hash_records':closures,'spatial':comparison,'linearization':linear,'continuum_geometry':continuum,'paired_mesh_geometry':paired,
 'native_initial':native,'native_increased_iteration_diagnostics':diagnostics,
 'external_input_observations':external,'figure':figure,
 'failed_reference_startup':'Initial drivers incorrectly required successful completion of a historical source used only for configuration. The preserved attempts fail before solving; corrected fresh references are live and excluded.',
 'goal_completion':False}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt'])
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'terminal_runs':len(runs),'source_hash_records':len(closures),'external_inputs':len(external)}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/egg-spatial-geometry-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'One matched6.25ms state on four fixed-polyhedron meshes retains15.52% tensor/9.35% energy changes,3.79–5.94GPa raw peak bounds and11.22% resolved-load linear tensor discrepancy at30298 tetrahedra. Exact geometry checks separate the12.79% original mass deficit from CAD approximation, which decreases to0.219% on30720-tet paired geometry assets. Increased native iterations recover original complete force gates but retain failures of the unchanged strict cached-corner field comparison. Failed startup attempts remain visible. New actual CAD/polyhedron contact references are live and excluded. Original histories, credible physical/contact/geometry tails, models/feedback and warmed costs at verified accuracy remain open.' \
    > "$task_meta/publication.log" 2>&1
python3 research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_meta/verification.json" \
    > "$task_meta/verification.log" 2>&1
