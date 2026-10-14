#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_python=${PUBLICATION_PYTHON:-python3}
task_epoch="$RIGID_STRESS_DATA_ROOT/runs/20261014-physical-validation"
task_version=${1:-v1}
task_tools="$task_epoch/resolved-model-publication-$task_version"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{map_resolved_grasp,summarize_resolved_refinement,plot_resolved_models}.py
    research/rigid_stress/scripts/{publish_resolved_models_checkpoint.sh,publish_nonconvex_checkpoint.py,verify_nonconvex_publication.py})
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
task_names=(grasp-u-spatial-y-l2-v1 grasp-u-replay-y-l2-v1 grasp-u-cache-only-v1 grasp-u-centred-cache-v1
    resolved-refinement-v1 resolved-load-models-y-v1 resolved-load-models-zero-v1
    resolved-load-models-y-v2 resolved-load-models-zero-v2 resolved-load-models-zero-v3
    resolved-load-models-y-v4 resolved-load-models-zero-v4
    resolved-load-models-y-v5 resolved-load-models-zero-v5 resolved-load-models-y-l2-v1
    resolved-model-plots-v1 resolved-model-plots-v2 resolved-model-plots-v3 resolved-model-plots-v4)
TASK_MODEL_EPOCH="$task_epoch" TASK_MODEL_NAMES="${task_names[*]}" "$task_python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ['TASK_MODEL_EPOCH'])
closures = []
for name in os.environ['TASK_MODEL_NAMES'].split():
    run = root / name
    before = (run / 'source.sha256').read_bytes()
    assert before == (run / 'source-after.sha256').read_bytes(), name
    for line in before.decode().splitlines():
        expected, relative = line.split('  ', 1)
        assert hashlib.sha256((run / 'source' / relative).read_bytes()).hexdigest() == expected
    closures.append({'name': name, 'returncode': int((run / 'returncode.txt').read_text())})
egg = json.loads((root / 'grasp-egg-spatial-l0-v1/external-termination.json').read_text())
assert egg['terminal_state'] == 'TIMEOUT'
assert not egg['producer_returncode_available'] and not egg['producer_source_after_available']
report = {
    'scope': 'Resolved U source-wrench model replay, independent contact refinement, qualified reference cache/gauge and exact original failures',
    'source_closures': closures,
    'external_egg_termination': egg,
    'refinement': json.loads((root / 'resolved-refinement-v1/results/summary.json').read_text()),
    'mapping': {name: json.loads((root / name / 'results/summary.json').read_text()) for name in
                ('resolved-load-models-y-v5', 'resolved-load-models-zero-v5', 'resolved-load-models-y-l2-v1')},
    'excluded_physical_observations': 'Earlier unguarded/repeated traction captures, rejected body-source-prior loads and original figure failures remain exact but are not physical acceptance evidence.',
    'live_jobs_excluded': ['714003_0', '714003_1', '714003_2', '714003_3'],
    'goal_completion': False,
}
print(json.dumps(report, indent=2))
PY
task_jobs=(713653 713701 713931 713942 713944 713949 713962 713989 713992 713999 714000 714017 714018 714030 714037)
task_ids=$(IFS=,; printf '%s' "${task_jobs[*]}")
sacct -X -j "$task_ids" -o JobIDRaw,State,ExitCode,Elapsed,End,NodeList -n -P > "$task_tools/jobs.txt"
sacct -X -j "$task_ids" -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
task_runs=()
for task_name in "${task_names[@]}"; do task_runs+=("$task_epoch/$task_name"); done
task_runs+=("$task_epoch/grasp-egg-spatial-l0-v1")
task_extra=("$task_tools/summary.json" "$task_tools/source.sha256" "$task_tools/jobs.txt" "$task_tools/job-commands.txt"
    "$task_epoch/matrix-mesh-u-v1/meshes/h0.0025.npz" "$task_epoch/matrix-mesh-u-v1/meshes/h0.00175.npz"
    "$task_epoch/matrix-mesh-egg-v1/meshes/h0.15.npz" "$task_epoch/grasp-u-plots-v3/style/nature_mpl_style.py"
    "$task_epoch/grasp-reference-acceleration-v1.sbatch" "$task_epoch/resolved-load-models-v1.sbatch"
    "$task_epoch/resolved-load-models-v2.sbatch" "$task_epoch/resolved-load-models-v4.sbatch"
    "$task_epoch/resolved-load-models-v5.sbatch" "$task_epoch/grasp-spatial-next-v1.sbatch")
while IFS= read -r -d '' task_source; do task_extra+=("$task_source"); done < <(find "$task_tools/source" -type f -print0)
for task_job in "${task_jobs[@]}"; do
    for task_path in "$task_epoch"/*"$task_job"*.log "$task_epoch"/*"$task_job"*.err; do
        if test -f "$task_path"; then task_extra+=("$task_path"); fi
    done
done
task_dest="research/rigid_stress/evidence/20261014-physical/resolved-model-checkpoint-$task_version"
"$task_python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$RIGID_STRESS_DATA_ROOT/runs" --destination "$task_dest" \
    --staging-root "$task_epoch/publication-staging" --run "${task_runs[@]}" \
    --extra "${task_extra[@]}" --summary "$task_tools/summary.json" \
    --limitations "Two useful resolved-U levels have large tensor/energy changes and up to7.5% pad strain. Original force/moment inputs, local traction and full native fields are audited in guarded replay. Near-zero aggregate body-prior rejection/acceptance at FP64 cancellation scale and original observer/figure failures remain visible. Egg wall-time termination has external metadata only, not fabricated producer closure. Third-level/separated-axis/egg retries are live and excluded. Nonlinear pads, contact regularization/history tails, actual Franka matching, rigid feedback and physically qualified accuracy/cost remain required."
"$task_python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
