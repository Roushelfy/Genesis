#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/stencil-patch-publication-$task_version"
task_run="$task_base/franka-precise-stencil-state-audit-v2"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_run" "$task_meta" <<'PY'
import hashlib,json,pathlib,subprocess,sys
run,meta=map(pathlib.Path,sys.argv[1:])
jobs=subprocess.check_output(['sacct','-X','-n','-P','-j','715888','--format=JobID,State,Elapsed,ExitCode'],text=True)
assert '715888_0|COMPLETED|' in jobs and int((run/'returncode.txt').read_text())==0
(meta/'jobs.txt').write_text(jobs)
for identity in ['source','input-source']:
 assert (run/(identity+'.sha256')).read_bytes()==(run/(identity+'-after.sha256')).read_bytes()
 for line in (run/(identity+'.sha256')).read_text().splitlines():
  sha,path=line.split('  ',1);assert hashlib.sha256((run/identity/path).read_bytes()).hexdigest()==sha
report=json.loads((run/'results/summary.json').read_text())
(meta/'summary.json').write_text(json.dumps({'scope':'Independent18-node traction diagnostic for the recovered8433-tet state26',
 'report':report,'goal_completion':False},indent=2)+'\n')
PY
task_dest="research/rigid_stress/evidence/20261014-physical/stencil-patch-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" --staging-root "$task_base/publication-staging" \
    --run "$task_run" --extra "$task_meta/summary.json" "$task_meta/jobs.txt" \
    --summary "$task_meta/summary.json" \
    --limitations 'Recovered8433-tet state26 has18 active body nodes,0.00283025mm² lumped support and1.45993GPa unsmoothed normal force/area density. This is one recovered dynamic state, not a physical contact boundary, completed80-state trajectory or spatial/contact tail. All original requirements remain open.' \
    > "$task_meta/publication.log" 2>&1
python3 research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_meta/verification.json" \
    > "$task_meta/verification.log" 2>&1
cp "$task_meta/verification.json" "$task_dest/verification.json"
python3 - "$task_dest/manifest.json" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);print(json.dumps({'files':len(json.loads(p.read_text())['files']),
 'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
PY
