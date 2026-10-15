#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/residual-recovery-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p,meta=map(pathlib.Path,sys.argv[1:])
runs={'franka-original-globalized-history-v1':'715800_0',
      'franka-original-residual-continuation-v1':'715811_0',
      'franka-original-869-recovered-audit-v1':'715813_0'}
ids=','.join(job.split('_')[0] for job in runs.values())
jobs=subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,State,ExitCode,Elapsed,End,NodeList','-n','-P'],text=True)
(meta/'jobs.txt').write_text(jobs)
states={row.split('|')[0].strip():row.split('|')[1].strip() for row in jobs.splitlines() if row.strip()}
(meta/'job-commands.txt').write_text(subprocess.check_output(['sacct','-X','-j',ids,'-o','JobID%40,SubmitLine%12000','-n','-P'],text=True))
closures=[];extras=set()
for name,job in runs.items():
 run=p/name;code=int((run/'returncode.txt').read_text())
 assert states[job]==('COMPLETED' if code==0 else 'FAILED'),(name,states[job],code)
 for identity in ['source','input-source']:
  before=run/(identity+'.sha256')
  assert before.read_bytes()==(run/(identity+'-after.sha256')).read_bytes()
  for line in before.read_text().splitlines():
   sha,relative=line.split('  ',1)
   assert hashlib.sha256((run/identity/relative).read_bytes()).hexdigest()==sha
   closures.append({'run':name,'identity':identity,'file':relative,'sha256':sha})
 extras.update(p.glob('*'+job+'.log'))
def report(name):return json.loads((p/name/'results/summary.json').read_text())
summary={'scope':'Retain failed pseudo-dt globalization and independently verified fixed-dt recovery of original869',
 'runs':[{'run':name,'job':job,'state':states[job]} for name,job in runs.items()],
 'source_closures':closures,
 'pseudo_dt_failure':json.loads((p/'franka-original-globalized-history-v1/globalization/original00869/summary.json').read_text()),
 'fixed_dt_residual_recovery':report('franka-original-residual-continuation-v1'),
 'all869_original_states_independent_audit':report('franka-original-869-recovered-audit-v1'),
 'live_history_and_audit_excluded':['715815_0','715817_0'], 'goal_completion':False}
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(meta/'runs.txt').write_text(''.join(str(p/name)+'\n' for name in runs))
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt',p/'meta/residual-recovery-commands-v1.json',
              p/'meta/contact-recovery-runtime-originals-verification.json'])
(meta/'extras.txt').write_text(''.join(str(path)+'\n' for path in sorted(extras)))
print(json.dumps({'terminal_runs':len(runs),'source_hash_records':len(closures)}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/residual-recovery-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'Fixed-dt residual continuation reaches the exact unmodified original869 equation and fresh robot-step budgets. Independent reconstruction checks every one of869 original physical states. Artificial defect stages are numerical iterates, never physical history; failed pseudo-dt attempts remain visible. Complete3.3s history and physical contact/geometry/model/feedback/verified accuracy-cost acceptance remain unfinished. Live producers are excluded.' \
    > "$task_meta/publication.log" 2>&1
python3 research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_meta/verification.json" \
    > "$task_meta/verification.log" 2>&1
cp "$task_meta/verification.json" "$task_dest/verification.json"
python3 - "$task_dest/manifest.json" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1:][0]);o=json.loads(p.read_text())
print(json.dumps({'files':len(o['files']),'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
PY
