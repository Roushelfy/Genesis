#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/egg-thickness-fields-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={'egg-fixed-surface-layers3-v1':'716343_0','egg-fixed-surface-layers4-v1':'716344_0',
      'egg-fixed-surface-thickness-audit-v1':'716353_0',
      'egg-thickness-probe-trace-audit-v1':'716828_0','egg-thickness-probe-trace-audit-v2':'716834_0'}
ids=','.join(job.split('_')[0] for job in runs.values())
jobs=subprocess.check_output(['sacct','-n','-P','-X','-j',ids,'--format=JobID%40,State%40,Elapsed,ExitCode,Start,End,NodeList'],text=True)
(meta/'jobs.txt').write_text(jobs)
(meta/'job-commands.txt').write_text(subprocess.check_output(['sacct','-n','-P','-X','-j',ids,'--format=JobID%40,SubmitLine%18000'],text=True))
states={line.split('|')[0].strip():line.split('|')[1].strip() for line in jobs.splitlines() if line.strip()}
identities=[];records=[];extras=set()
for name,job in runs.items():
 run=p/name;assert states[job]=='COMPLETED' and int((run/'returncode.txt').read_text())==0
 scopes=[run,*[run/folder for folder in ['fem','probes'] if (run/folder/'source.sha256').exists()]]
 for scope in scopes:
  assert int((scope/'returncode.txt').read_text())==0
  for kind in ['source','input-source']:
   f=scope/(kind+'.sha256')
   if not f.exists():continue
   assert f.read_bytes()==(scope/(kind+'-after.sha256')).read_bytes()
   for line in f.read_text().splitlines():
    sha,path=line.split('  ',1);assert hashlib.sha256((scope/kind/path).read_bytes()).hexdigest()==sha
    identities.append({'run':str(scope),'identity':kind,'file':path,'sha256':sha})
 records.append({'run':name,'job':job,'state':states[job],'returncode':0})
 extras.update(p.glob('*'+job+'.log'))
audit=json.loads((p/'egg-fixed-surface-thickness-audit-v1/results/summary.json').read_text())
assert audit['fields_verified']==len(audit['rows'])==15 and not audit['physical_accuracy_accepted']
assert len(audit['adjacent_changes'])==10
assert all(not row['all_five_changes_below_initial3percent'] for row in audit['adjacent_changes'])
for row in audit['rows']:
 assert row['recorded_complete_FEM_budget_fraction']<=1
traces=json.loads((p/'egg-thickness-probe-trace-audit-v2/results/summary.json').read_text())
assert len(traces['rows'])==15 and all(row['off_minimum_barycentric_coordinate']>1e-8 for row in traces['rows'])
for path,sha in traces['input_sha256'].items():
 path=pathlib.Path(path);assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;extras.add(path)
for path,sha in audit['independent_boundary_and_surface_flux_audit']['unchanged_input_sha256'].items():
 path=pathlib.Path(path);assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;extras.add(path)
baseline=p/'fine-ref-egg-l2-v1';probes=p/'fine-ref-egg-l2-v1-probes'
for run in [baseline,probes]:
 for folder in ['source','results']:
  extras.update(path for path in (run/folder).rglob('*') if path.is_file() and '__pycache__' not in path.parts)
 for name in ['source.sha256','source-after.sha256','returncode.txt']:
  extras.add(run/name)
large=[path for path in extras if path.stat().st_size>=50_000_000]
extras.difference_update(large)
# The complete baseline directories put large fields through the same byte-sharding path
archive_runs=[*(p/name for name in runs),baseline,probes]
summary={'scope':'Fixed-surface two/three/four-layer FEM fields, exact traction/geometry checks and original-point stress-trace diagnostics',
 'runs':records,'source_identity_records':identities,'independent_full_axis_audit':audit,'probe_trace_diagnostics':traces,
 'baseline_originals_preserved_in_complete_run_directories':True,
 'limitations':'All15 original FEM fields and full raw peaks remain intact. Stress still changes3.62–4.93% on the3-to4-layer tail; independent additional interior probes retain3.77–4.17% changes, so exact-interface traces do not explain the discrepancy. More layers are scheduled, not accepted. Full contact/geometry/model/feedback/application and verified-cost scope remains open.',
 'goal_completion':False}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt',p/'meta/thickness-five-six-commands-v1.json',
 p/'publication-binary-sharding-qualification-v1/before-result.json',p/'publication-binary-sharding-qualification-v1/after-result.json',p/'publication-binary-sharding-qualification-v1/actual-runtime-verification.json'])
(meta/'runs.txt').write_text(''.join(str(run)+'\n' for run in archive_runs))
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'closed_primary_runs':len(runs),'source_identity_records':len(identities),'verified_original_FEM_fields':15,'large_fields_sharded':len(large)}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/egg-thickness-fields-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'Full original two/three/four-layer fields, unchanged actual boundaries/geometry/mass and all15 boundary-traction/field audits are retained. Raw peak/energy stability does not override remaining full-tensor changes. Large NPZ files use exact ordered byte chunks; no arrays or records are reduced. Additional probes and unsmoothed one-sided neighborhoods diagnose the existing discrepancy without replacing original observables. Finer layers and full original physical/contact/model/feedback/verified-cost gates remain open.' \
    > "$task_meta/publication.log" 2>&1
python3 research/rigid_stress/scripts/verify_runtime_publication.py \
    --manifest "$task_dest/manifest.json" --runtime-root "$task_root/runs" \
    --staging-root "$task_base/publication-staging/egg-thickness-fields-$task_version" \
    --output "$task_meta/runtime-verification.json" > "$task_meta/runtime-verification.log" 2>&1
