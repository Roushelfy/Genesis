#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_python=${PUBLICATION_PYTHON:-python3}
task_epoch="$RIGID_STRESS_DATA_ROOT/runs/20261014-physical-validation"
task_version=${1:-v1}
task_tools="$task_epoch/objective-franka-publication-$task_version"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{audit_objective_frame,audit_contact_axes,audit_franka_reference,capture_franka_reference,plot_franka_reference,replay_objective_contact,replay_resolved_grasp}.py
    research/rigid_stress/scripts/{publish_objective_franka_checkpoint.sh,publish_nonconvex_checkpoint.py,verify_nonconvex_publication.py})
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
task_names=(grasp-u-pad-axis-l1-p7-v1 grasp-u-body-axis-l2-p5-v1
    grasp-u-neohookean-pads-y-l1-v1 grasp-u-neohookean-all-y-l1-v1
    grasp-u-neohookean-pads-replay-v1 grasp-u-objective-replay-v1 grasp-u-objective-replay-v2 grasp-u-objective-replay-v3
    objective-frame-v1 objective-frame-v2 contact-separated-axes-v1
    franka-input-v1 franka-input-s2-v1 franka-input-s4-v1 franka-input-s4-v2
    franka-input-complete-s1-v1 franka-input-complete-s2-v1 franka-input-complete-s4-v1
    franka-input-audit-v1 franka-input-audit-v2 franka-input-audit-v3 franka-input-audit-v4
    franka-input-plots-v1)
TASK_OBJECTIVE_EPOCH="$task_epoch" TASK_OBJECTIVE_NAMES="${task_names[*]}" "$task_python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ['TASK_OBJECTIVE_EPOCH'])
closures = []
for name in os.environ['TASK_OBJECTIVE_NAMES'].split():
    run = root / name
    before = (run / 'source.sha256').read_bytes()
    assert before == (run / 'source-after.sha256').read_bytes(), name
    for line in before.decode().splitlines():
        expected, relative = line.split('  ', 1)
        assert hashlib.sha256((run / 'source' / relative).read_bytes()).hexdigest() == expected, (name, relative)
    closures.append({'run': name, 'returncode': int((run / 'returncode.txt').read_text())})
for name in ('grasp-u-neohookean-pads-y-l1-v1','grasp-u-neohookean-all-y-l1-v1'):
    run = root / name
    assert (run/'input-source.sha256').read_bytes() == (run/'input-source-after.sha256').read_bytes()
current = ['franka-input-complete-s1-v1','franka-input-complete-s2-v1','franka-input-complete-s4-v1']
for name in current:
    run = root/name
    report = json.loads((run/'results/summary.json').read_text())
    assert len(report['captures']) == len({row['geometry_snapshot'] for row in report['captures']})
    assert (run/'source/examples/rigid/assets/hollow_egg/level1/elastic.npz').exists()
audit = json.loads((root/'franka-input-audit-v4/results/summary.json').read_text())
assert all(run['body_asset_contemporaneously_snapshotted'] for run in audit['runs'])
assert all(g['unchanged_geometry_guard_passed'] for run in audit['runs'] for g in run['geometry_checks'])
summary = {'scope':'Solved nonlinear U, objective native replay, separated spatial axes and complete actual Panda inputs with exact original failures',
           'source_closures':closures,
           'objective_replay':json.loads((root/'grasp-u-objective-replay-v3/results/summary.json').read_text()),
           'franka_strict_input_audit':audit,
           'contact_axes':json.loads((root/'contact-separated-axes-v1/results/summary.json').read_text()),
           'earlier_capture_limitations':'Original rg-unavailable captures have incomplete engine/asset source lists; earlier2/4-substep geometry filenames overwrite within a macro step. Diagnostic v3 retains its failed geometry guards. These original attempts are preserved but are not accepted as complete matched inputs.',
           'live_jobs_excluded':['714003_0','714003_3'], 'goal_completion':False}
print(json.dumps(summary,indent=2))
PY
task_jobs=(714003 714067 714106 714113 714140 714141 714144 714145 714146 714161 714186 714188 714189 714194 714195 714196 714197 714198)
task_ids=$(IFS=,; printf '%s' "${task_jobs[*]}")
sacct -X -j "$task_ids" -o JobIDRaw,State,ExitCode,Elapsed,End,NodeList -n -P > "$task_tools/jobs.txt"
sacct -X -j "$task_ids" -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
task_runs=()
for task_name in "${task_names[@]}"; do task_runs+=("$task_epoch/$task_name"); done
task_extra=("$task_tools/source.sha256" "$task_tools/jobs.txt" "$task_tools/job-commands.txt"
    "$task_epoch/grasp-neohookean-v1.sbatch" "$task_epoch/grasp-spatial-next-v1.sbatch"
    "$task_epoch/franka-input-temporal-v1.sbatch" "$task_epoch/franka-input-complete-v1.sbatch"
    "$task_epoch/grasp-u-plots-v3/style/nature_mpl_style.py")
while IFS= read -r -d '' task_source; do task_extra+=("$task_source"); done < <(find "$task_tools/source" -type f -print0)
for task_job in 714067 714106 714113 714140 714141 714144 714145 714146 714161 714186 714188 714189 714194 714195 714196 714197 714198; do
    for task_path in "$task_epoch"/*"$task_job"*.log "$task_epoch"/*"$task_job"*.err; do
        if test -f "$task_path"; then task_extra+=("$task_path"); fi
    done
done
for task_index in 1 2; do
    task_extra+=("$task_epoch/grasp-spatial-next-v1-714003_$task_index.log" "$task_epoch/grasp-spatial-next-v1-714003_$task_index.err")
done
task_dest="research/rigid_stress/evidence/20261014-physical/objective-franka-checkpoint-$task_version"
"$task_python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$RIGID_STRESS_DATA_ROOT/runs" --destination "$task_dest" \
    --staging-root "$task_epoch/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_tools/summary.json" \
    --limitations "Objective low-strain nonlinear U pilot and strict same-load native recovery are qualified at one spatial level. Separated body/contact axes remain unresolved. Complete actual Panda mass/COM/geometry/actuation/friction/substep snapshots pass original geometry gates but retain angular motion and rank5 point support. Original missing-source/callback-index/method-name failures remain exact and excluded from complete matched evidence. Live third-level U/full egg are excluded. Actual matched deformable Panda, full physical/contact tails, feedback/model separation and representative accuracy/cost remain required."
"$task_python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
