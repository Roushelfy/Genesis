#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/thickness-original-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={
 'franka-original-audited-prefix-full-history-v1':'716302_0',
 'franka-original-audited-prefix-full-history-audit-v1':'716303_0',
 'franka-original-audited-prefix-full-history-fields-v1':'716319_0',
 'franka-original-audited-prefix-full-history-native-v1':'716320_0',
 'franka-original-audited-prefix-full-history-bounds-v1':'716321_0',
 'franka-original-1937-direct-candidate-v1':'716350_0',
 'franka-original-1937-continued-stage-budget-v1':'716352_0',
 'franka-original-1937-continued-stage-audit-v1':'716354_0',
 'franka-original-869-stage-budget-qualification-v1':'716355_0',
 'original-869-stage-method-equivalence-v1':'716356_0',
 'egg-fixed-surface-geometry-audit-v1':'716370_0',
}
ids=','.join(job.split('_')[0] for job in runs.values())
jobs=subprocess.check_output(['sacct','-n','-P','-X','-j',ids,
 '--format=JobID%40,State,Elapsed,ExitCode,Start,End,NodeList'],text=True)
(meta/'jobs.txt').write_text(jobs)
(meta/'job-commands.txt').write_text(subprocess.check_output(['sacct','-n','-P','-X','-j',ids,
 '--format=JobID%40,SubmitLine%16000'],text=True))
states={line.split('|')[0].strip():line.split('|')[1].strip() for line in jobs.splitlines() if line.strip()}
closures=[];records=[];extras=set()
for name,job in runs.items():
 run=p/name;failed=job in {'716302_0','716350_0'};code=int((run/'returncode.txt').read_text())
 assert states[job]==('FAILED' if failed else 'COMPLETED')
 assert (code!=0)==failed
 for kind in ['source','input-source']:
  before=run/(kind+'.sha256')
  if not before.exists():continue
  assert before.read_bytes()==(run/(kind+'-after.sha256')).read_bytes(),(name,kind)
  for line in before.read_text().splitlines():
   sha,path=line.split('  ',1)
   assert hashlib.sha256((run/kind/path).read_bytes()).hexdigest()==sha
   closures.append({'run':name,'identity':kind,'file':path,'sha256':sha})
 records.append({'run':name,'job':job,'scheduler_state':states[job],'returncode':code,
                 'closure_kind':'natural_producer_exit'})
 extras.update(p.glob('*'+job+'.log'))
def report(name):return json.loads((p/name/'results/summary.json').read_text())
prefix=report('franka-original-audited-prefix-full-history-audit-v1')
recovered=report('franka-original-1937-continued-stage-audit-v1')
assert prefix['steps_checked']==1936 and not prefix['complete_requested_history']
assert recovered['steps_checked']==1937 and recovered['independently_verified_recovered_step']==1937
assert not recovered['complete_requested_history']
assert all(row['body_dynamic_residual_N']<=1e-9 and row['actual_robot_step_effort_residual']<=1e-9
           for row in recovered['rows'])
fields=report('franka-original-audited-prefix-full-history-fields-v1')
native=report('franka-original-audited-prefix-full-history-native-v1')
bounds=report('franka-original-audited-prefix-full-history-bounds-v1')
assert fields['states_checked']==1936 and len(fields['rows'])==1936
assert len(native['rows'])==12 and all(row['native_complete_budget_fraction']<=1 for row in native['rows'])
assert bounds['total_sampled_points']==8147520
equivalence=report('original-869-stage-method-equivalence-v1')
assert len(equivalence['rows'])==1 and equivalence['rows'][0]['step']==869
assert equivalence['rows'][0]['all_corner_tensor_relative_change']==0
assert equivalence['rows'][0]['all_quadrature_tensor_relative_change']==0
assert all(row['absolute_L2']==0 for row in equivalence['rows'][0]['all_saved_array_changes'].values())
geometry=report('egg-fixed-surface-geometry-audit-v1')
assert [row['layers'] for row in geometry['rows']]==[2,3,4]
assert len({row['oriented_surface_flux_volume_m3'] for row in geometry['rows']})==1
for path,sha in geometry['unchanged_input_sha256'].items():
 assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==sha
 extras.add(pathlib.Path(path))
observation=p/'original-869-stage-method-observation-v1'
observed=json.loads((observation/'observation.json').read_text())
assert observed['observed_steps']==[869] and not observed['complete_producer_history']
for row in observed['files']:
 path=pathlib.Path(row['observed']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
extras.update(path for path in observation.rglob('*') if path.is_file())
extras.update([p/'meta/thickness-original-followup-commands-v1.json',
 p/'meta/thickness-original-followup-commands-v2.json'])
summary={
 'scope':'Independent fixed-boundary egg geometry axis;1936 original global fields and1937 physical-state recovery',
 'runs':records,'source_hash_records':closures,
 'independent1936_prefix_audit':prefix,'all1936_global_fields':fields,
 'actual_native_replays':native,'interior_corner_certificate_checks':bounds,
 'original1937_residual_recovery':report('franka-original-1937-continued-stage-budget-v1'),
 'independent1937_control_force_audit':recovered,'known869_complete_state_field_equivalence':equivalence,
 'independent_mesh_only_geometry_audit':geometry,
 'live_FEM_producer_mesh_inputs_only':geometry['data_producers_naturally_closed'],
 'live_FEM_fields_and_original_continuations_excluded':True,
 'goal_turn_classification':'progress: physical1937 recovery, exact known-state qualification and independent fixed-boundary geometry axis',
 'goal_completion':False,
}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt'])
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'closed_runs':len(runs),'source_hash_records':len(closures),'independent_physical_states':1937}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extras < "$task_meta/extras.txt"
task_destination="research/rigid_stress/evidence/20261014-physical/thickness-original-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_destination" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extras[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'The original1936-state producer fails and its saved direct1937 candidate fails the original physical gates. Only the fraction-one unmodified residual-continuation state is independently accepted, with all1937 states audited. Artificial stages are not physical history. Known869 arrays and full tensors are unchanged. Fixed-surface2/3/4-layer meshes are independently audited geometry inputs; their live FEM fields and all live continuations are excluded. Full original history, physical convergence, model/feedback and verified accuracy/cost remain incomplete.' \
    > "$task_meta/publication.log" 2>&1
python3 research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_destination/manifest.json" --originals --output "$task_meta/verification.json" \
    > "$task_meta/verification.log" 2>&1
cp "$task_meta/verification.json" "$task_destination/verification.json"
python3 - "$task_root/runs" "$task_base/publication-staging" "$task_destination" "$task_meta" <<'PY'
import hashlib,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path('research/rigid_stress/scripts').resolve()))
from verify_nonconvex_publication import digest
root,staging,destination,meta=map(pathlib.Path,sys.argv[1:])
manifest_path=destination/'manifest.json';manifest=json.loads(manifest_path.read_text())
assert not manifest['summary']['sharded_original_files']
stage=staging/destination.name
for row in manifest['files']:
 original=pathlib.Path(row['original'])
 actual=root/original.relative_to(stage) if original.is_relative_to(stage) else original
 assert digest(actual)==(row['sha256'],row['bytes']),actual
result={'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
 'true_runtime_original_entries_verified':len(manifest['files']),
 'source_hash_records_verified':len(manifest['summary']['source_hash_records']),
 'all_stored_and_decompressed_bytes_verified':'verification.json','passed':True,'goal_completion':False}
for output in [meta/'true-runtime-verification.json',destination/'true-runtime-verification.json']:
 output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
PY
