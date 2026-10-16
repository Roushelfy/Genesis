#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_python=${PUBLICATION_PYTHON:-python3}
task_epoch="$RIGID_STRESS_DATA_ROOT/runs/20261014-physical-validation"
task_version=${1:-v1}
task_tools="$task_epoch/franka-fixture-publication-$task_version"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{audit_franka_fixture,audit_franka_fixture_axes,audit_franka_release,franka_rigid_mode_coordinates}.py
    research/rigid_stress/scripts/{audit_franka_fixture.sh,audit_franka_fixture_axes.sh,audit_franka_release.sh,publish_franka_fixture_checkpoint.sh,publish_nonconvex_checkpoint.py,verify_nonconvex_publication.py})
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
task_names=(franka-ref-release-modes-v1 franka-release-modes-audit-v1
    franka-rigid-fixture-controlled-v1 franka-rigid-fixture-audit-v1 franka-rigid-fixture-audit-v2)
task_axis_limits="Temporal axes are excluded."
if [[ "$task_version" == v2 ]]; then
    task_names+=(franka-rigid-fixture-s2-v1 franka-rigid-fixture-s8-v1 franka-rigid-fixture-time-audit-v1)
    task_axis_limits="Completed2/4/8-substep histories are compared at identical physical times without interpolation. Temporal agreement is not contact-regularization or model/deformation-feedback accuracy."
fi
TASK_FIXTURE_EPOCH="$task_epoch" TASK_FIXTURE_NAMES="${task_names[*]}" "$task_python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
from pathlib import Path
root = Path(os.environ['TASK_FIXTURE_EPOCH'])
closures = []
for name in os.environ['TASK_FIXTURE_NAMES'].split():
    run = root/name
    before = (run/'source.sha256').read_bytes()
    assert before == (run/'source-after.sha256').read_bytes(), name
    for line in before.decode().splitlines():
        expected,relative=line.split('  ',1)
        assert hashlib.sha256((run/'source'/relative).read_bytes()).hexdigest() == expected, (name,relative)
    closures.append({'run':name,'returncode':int((run/'returncode.txt').read_text())})
rigid=json.loads((root/'franka-rigid-fixture-controlled-v1/results/summary.json').read_text())
assert len(rigid['captures']) == 576
assert len({(r['phase'],r['tick'],r['physical_substep']) for r in rigid['captures']}) == 576
audit=json.loads((root/'franka-rigid-fixture-audit-v2/results/summary.json').read_text())
report={'scope':'Complete corresponding rigid fixed-hand history and retained failed nonlinear free-body coordinate experiment',
        'source_closures':closures,'rigid_fixture_audit':audit,
        'rigid_mode_release_diagnostics':json.loads((root/'franka-release-modes-audit-v1/results/summary.json').read_text()),
        'live_jobs_excluded':['714003_0','714003_3','714277_0','714277_1'], 'goal_completion':False}
if 'franka-rigid-fixture-time-audit-v1' in os.environ['TASK_FIXTURE_NAMES'].split():
    report['temporal_axis']=json.loads((root/'franka-rigid-fixture-time-audit-v1/results/summary.json').read_text())
    report['live_jobs_excluded']=['714003_0','714003_3']
print(json.dumps(report,indent=2))
PY
task_jobs=(714252 714253 714262 714276 714279)
if [[ "$task_version" == v2 ]]; then task_jobs+=(714277 714294); fi
task_ids=$(IFS=,; printf '%s' "${task_jobs[*]}")
sacct -X -j "$task_ids" -o JobIDRaw,State,ExitCode,Elapsed,End,NodeList -n -P > "$task_tools/jobs.txt"
sacct -X -j "$task_ids" -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
task_runs=()
for task_name in "${task_names[@]}"; do task_runs+=("$task_epoch/$task_name"); done
task_extra=("$task_tools/source.sha256" "$task_tools/jobs.txt" "$task_tools/job-commands.txt" "$task_tools/summary.json")
while IFS= read -r -d '' task_source; do task_extra+=("$task_source"); done < <(find "$task_tools/source" -type f -print0)
for task_job in "${task_jobs[@]}"; do
    for task_path in "$task_epoch"/*"$task_job"*.log "$task_epoch"/*"$task_job"*.err; do
        if test -f "$task_path"; then task_extra+=("$task_path"); fi
    done
done
task_dest="research/rigid_stress/evidence/20261014-physical/franka-fixture-checkpoint-$task_version"
"$task_python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$RIGID_STRESS_DATA_ROOT/runs" --destination "$task_dest" \
    --staging-root "$task_epoch/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_tools/summary.json" \
    --limitations "Complete rigid fixed-hand protocol and unique same-substep force/pose frames are retained. The body still rotates after20s released hold; small translational speed and nearly stable normal force do not prove a static reference. The temporary compliant weld differs from the independent held-row initializer. Full arm PD/soft finger equality are omitted. Rigid-mode free-body reference still fails its original solver gate, and its global forces/moments/stresses remain unaccepted. $task_axis_limits Live U/full-egg are excluded. Corresponding solved deformable/dynamic contact, full convergence/model/feedback and physical accuracy/cost remain required."
"$task_python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
