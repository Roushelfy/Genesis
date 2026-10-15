#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/generic-stencil-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={
 'egg-paired-polyhedron-level3-contact-v1':'716498_0',
 'egg-paired-polyhedron-level3-contact-v1-audit':'716499_0',
 'egg-paired-cad-level3-contact-v1':'716502_0',
 'egg-paired-cad-level4-contact-v1':'716504_0',
 'generic-stencil-api-v1':'716612_0',
 'generic-stencil-fixture-qualification-v1':'716616_0',
 'generic-stencil-actual-observation-v2':'716626_0',
 'generic-stencil-original-polyhedron-level2-audit-v1':'716727_0',
 'generic-stencil-original-cad-level2-audit-v1':'716728_0',
 'egg-paired-cad-level3-generic-contact-v1':'716729_0',
 'egg-paired-cad-level3-generic-contact-v1-audit':'716730_0',
 'egg-paired-polyhedron-three-level-fields-v1':'716732_0',
 'generic-stencil-cad-level3-resolved-observation-v1':'716734_0',
 'egg-paired-polyhedron-three-level-fields-v2':'716735_0',
 'egg-adaptive-cad-polyhedron-geometry-v1':'716746_0',
 'egg-adaptive-cad-polyhedron-geometry-v2':'716784_0',
 'egg-adaptive-cad-polyhedron-geometry-v3':'716792_0',
}
failed={'716502_0','716504_0','716732_0'}
cancelled={'716746_0'}
ids=','.join(job.split('_')[0] for job in runs.values())
jobs=subprocess.check_output(['sacct','-n','-P','-X','-j',ids,'--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList'],text=True)
(meta/'jobs.txt').write_text(jobs)
(meta/'job-commands.txt').write_text(subprocess.check_output(['sacct','-n','-P','-X','-j',ids,'--format=JobID%40,SubmitLine%18000'],text=True))
states={line.split('|')[0].strip():line.split('|')[1].strip() for line in jobs.splitlines() if line.strip()}
records=[];identities=[];extras=set()
for name,job in runs.items():
 run=p/name;code=int((run/'returncode.txt').read_text())
 expected='CANCELLED' if job in cancelled else 'FAILED' if job in failed else 'COMPLETED'
 assert states[job].startswith(expected),(name,states[job])
 assert (code!=0)==(job in failed),(name,code)
 for kind in ['source','input-source']:
  before=run/(kind+'.sha256')
  assert before.read_bytes()==(run/(kind+'-after.sha256')).read_bytes(),(name,kind)
  for line in before.read_text().splitlines():
   sha,path=line.split('  ',1)
   assert hashlib.sha256((run/kind/path).read_bytes()).hexdigest()==sha,(name,kind,path)
   identities.append({'run':name,'identity':kind,'file':path,'sha256':sha})
 records.append({'run':name,'job':job,'scheduler_state':states[job],'actual_returncode':code,'actual_EXIT_identities_present':True,'completed_physics_inferred_from_wrapper_code':False})
 extras.update(p.glob('*'+job+'.log'))
def report(name):return json.loads((p/name/'results/summary.json').read_text())
fixture=report('generic-stencil-fixture-qualification-v1')
assert fixture['qualified'] and len(fixture['rows'])==4
assert all(test['relative_derivative_difference']<=1e-6 for row in fixture['rows'] for test in row['directional_derivatives'])
audits={name:report(name) for name in ['egg-paired-polyhedron-level3-contact-v1-audit','generic-stencil-original-polyhedron-level2-audit-v1','generic-stencil-original-cad-level2-audit-v1','egg-paired-cad-level3-generic-contact-v1-audit']}
assert all(s['states_checked']==len(s['rows'])==5 for s in audits.values())
assert all(row['complete_body_dynamic_residual_N']<=1e-9 and row['complete_mapped_dynamic_residual_N']<=1e-9 for s in audits.values() for row in s['rows'])
comparison=report('egg-paired-polyhedron-three-level-fields-v2')
assert [r['tetrahedra'] for r in comparison['rows']]==[480,1920,7680]
assert comparison['common_physical_probes']==2400
assert all(r['maximum_boundary_vertex_and_facet_centroid_distance_to_supported_seed_facet_m']<=1e-12 for r in comparison['rows'])
adaptive=report('egg-adaptive-cad-polyhedron-geometry-v3')
assert [r['tetrahedra'] for r in adaptive['rows']]==[17340,17340,22152,22152,32340,32340,61812,61812]
assert all(r['maximum_local_CAD_surface_edge_length_m']<=r['target_CAD_surface_edge_length_m'] and r['original_initializer_all_anchors_retained'] for r in adaptive['rows'])
for s in [comparison,report('egg-adaptive-cad-polyhedron-geometry-v2'),adaptive]:
 for path,sha in s['input_sha256'].items():
  path=pathlib.Path(path);assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
  extras.add(path)
observation=report('generic-stencil-actual-observation-v2')
resolved=report('generic-stencil-cad-level3-resolved-observation-v1')
for s in [observation,resolved]:
 for state in s['actual_state_observations']:
  path=pathlib.Path(state['state']);assert hashlib.sha256(path.read_bytes()).hexdigest()==state['state_sha256'];extras.add(path)
assert {r['type'] for r in observation['actual_state_observations'][0]['stencils']}=={'EdgeEdgeNormalCollision'}
for name in ['generic-stencil-progress-commands-v1.json','adaptive-cad-contact-commands-v1.json','adaptive-initial-termination-observation-v1.json']:
 extras.add(p/'meta'/name)
summary={'scope':'Generic point/edge/face reference force qualification, recovered five-state7680-tet CAD contact, independently audited old states and three fixed-polyhedron spatial fields',
 'runs':records,'source_hash_records':identities,'fixture_qualification':fixture,'independent_audits':audits,'three_level_polyhedron_fields':comparison,'actual_stencils_before_recovery':observation,'actual_stencils_after_recovery':resolved,'adaptive_geometry_assets':adaptive,
 'cancelled_adaptive_build':json.loads((p/'meta/adaptive-initial-termination-observation-v1.json').read_text()),
 'live_level4_and_adaptive_contact_results_excluded':True,'goal_completion':False,
 'limitations':'Generic centered collision reconstruction retains IPC1.6 unmollified friction magnitude and all original force budgets. Independent old-state audits cover ten unchanged states; the recovered CAD case covers five new dynamic states with byte-identical original initializer, not independently audited closure history. The fixed polyhedron still has47.22% tensor/21.54% energy tail changes. Adaptive meshes qualify geometry/edge targets only, not contact accuracy. Cancelled wrapper0/missing summary remains incomplete; failed type/mismatched-arithmetic attempts remain visible. Full application, physical/contact/geometry/regularization tails, models/feedback and verified warmed costs remain required.'}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt'])
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'terminal_runs':len(runs),'source_identity_records':len(identities),'independently_checked_dynamic_states':20}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/generic-stencil-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'Four qualified reference stencils,20 independently reconstructed dynamic states, recovered7680-tet CAD edge-edge contact, fixed-polyhedron three-level nonconvergence and conforming paired adaptive geometry assets. Successes, failures and a cancelled wrapper0 build are preserved with actual scheduler/EXIT identities. This does not establish full original motion/history, converged contact/geometry/regularization, remaining models/feedback or physically verified warmed cost. All live producers are excluded.' \
    > "$task_meta/publication.log" 2>&1
python3 research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_meta/verification.json" \
    > "$task_meta/verification.log" 2>&1
