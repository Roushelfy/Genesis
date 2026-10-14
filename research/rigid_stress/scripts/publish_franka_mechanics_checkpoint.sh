#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_python=${PUBLICATION_PYTHON:-python3}
task_epoch="$RIGID_STRESS_DATA_ROOT/runs/20261014-physical-validation"
task_version=${1:-v1}
task_tools="$task_epoch/franka-mechanics-publication-$task_version"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{franka_contact_reference,franka_rigid_mode_coordinates,stable_neohookean_reference,qualify_franka_contact,franka_fixture_geometry,capture_franka_fixture,audit_franka_coordinate_frame,audit_franka_release}.py
    research/rigid_stress/scripts/{franka_contact_reference.sh,capture_franka_fixture.sh,audit_franka_coordinate_frame.sh,audit_franka_release.sh,publish_franka_mechanics_checkpoint.sh,publish_nonconvex_checkpoint.py,verify_nonconvex_publication.py})
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
task_names=(franka-ref-qualification-v1 franka-ref-controlled-v1 franka-stable-qualification-v1
    franka-centred-qualification-v1 franka-coordinate-audit-v1 franka-ref-centred-v1
    franka-ref-release-lsq-v1 franka-release-audit-v1 franka-release-lsq-audit-v1
    franka-rigid-fixture-geometry-v1 franka-rigid-fixture-geometry-v2 franka-centred-qualification-v2)
TASK_FRANKA_EPOCH="$task_epoch" TASK_FRANKA_NAMES="${task_names[*]}" "$task_python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ['TASK_FRANKA_EPOCH'])
closures = []
for name in os.environ['TASK_FRANKA_NAMES'].split():
    run = root/name
    before = (run/'source.sha256').read_bytes()
    assert before == (run/'source-after.sha256').read_bytes(), name
    for line in before.decode().splitlines():
        expected, relative = line.split('  ',1)
        assert hashlib.sha256((run/'source'/relative).read_bytes()).hexdigest() == expected, (name,relative)
    closures.append({'run':name,'returncode':int((run/'returncode.txt').read_text())})
selected = json.loads((root/'franka-centred-qualification-v2/results/summary.json').read_text())
state = selected['active_contact_failure_requalification']
assert state['unchanged_tolerance_N'] == 1e-9 and state['stable_solved_residual_N'] <= 1e-9
assert state['original_held_rows'] == [25,31,75,76,84,473]
geometry = json.loads((root/'franka-rigid-fixture-geometry-v2/results/geometry-audit.json').read_text())
assert len(geometry['geometry']) == 12
assert max(row['world_vertex_hausdorff_error_m'] for row in geometry['geometry']) < 1e-10
report = {'scope':'Actual compound Panda kinematics, same-held-row contact requalification and explicit incomplete free-body release diagnostics',
          'source_closures':closures, 'selected_contact_qualification':selected,
          'strict_rigid_fixture_geometry':geometry,
          'direct_release_diagnostics':json.loads((root/'franka-release-audit-v1/results/summary.json').read_text()),
          'least_squares_release_diagnostics':json.loads((root/'franka-release-lsq-audit-v1/results/summary.json').read_text()),
          'excluded_same_state_claim':'Centred qualification v1 selected different held rows after QR pivot changes. Its numeric outputs are retained exactly but do not qualify an identical fixture. Selected v2 and the coordinate audit preserve all original held rows and reproduce the actual original failure residual.',
          'live_jobs_excluded':['714003_0','714003_3','714252_0','714253_0'], 'goal_completion':False}
print(json.dumps(report,indent=2))
PY
task_jobs=(714213 714216 714227 714229 714230 714233 714236 714237 714238 714247 714249 714250)
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
task_dest="research/rigid_stress/evidence/20261014-physical/franka-mechanics-checkpoint-$task_version"
"$task_python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$RIGID_STRESS_DATA_ROOT/runs" --destination "$task_dest" \
    --staging-root "$task_epoch/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_tools/summary.json" \
    --limitations "Actual12-geom rigid Panda fingers and mass/PD kinematics are qualified. The source-coordinate numerical floor is requalified using identical held rows and unchanged1e-9N tolerance. The completed held closure still has support reactions; both original free-body release solvers fail and are unaccepted. A nonlinear peak interval or small deformation is not equilibrium evidence. Fixed-hand fixture omits full-arm PD and soft finger equality; its time history and omitted-feedback comparison remain required. Original v1 fixture/source-coordinate and wrong-held-row diagnostic attempts are preserved exactly. Live rigid fixture/rigid-mode solver and U/full-egg are excluded. Full physical/contact/model/cost acceptance remains incomplete."
"$task_python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
