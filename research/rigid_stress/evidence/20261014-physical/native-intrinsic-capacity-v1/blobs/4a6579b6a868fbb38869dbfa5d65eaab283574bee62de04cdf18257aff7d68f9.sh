#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_python=${PUBLICATION_PYTHON:-python3}
task_epoch="$RIGID_STRESS_DATA_ROOT/runs/20261014-physical-validation"
task_version=${1:-v1}
task_tools="$task_epoch/neohookean-publication-$task_version"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{neohookean_reference,qualify_neohookean}.py
    research/rigid_stress/neohookean_grasp_driver.patch
    research/rigid_stress/scripts/{qualify_neohookean.sh,run_neohookean_grasp.sh,publish_neohookean_checkpoint.sh,publish_nonconvex_checkpoint.py,verify_nonconvex_publication.py})
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
TASK_NEO_EPOCH="$task_epoch" "$task_python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ['TASK_NEO_EPOCH'])
closures = []
for name in ('neohookean-qualification-v1', 'neohookean-qualification-v2'):
    run = root / name
    before = (run / 'source.sha256').read_bytes()
    assert before == (run / 'source-after.sha256').read_bytes()
    for line in before.decode().splitlines():
        expected, relative = line.split('  ', 1)
        assert hashlib.sha256((run / 'source' / relative).read_bytes()).hexdigest() == expected
    assert int((run / 'returncode.txt').read_text()) == 0
    closures.append({'run': name, 'returncode': 0})
report = {'scope': 'Independent nonlinear P2 assembly, derivative/objectivity/compression checks, whole-volume stress bounds and saved-position diagnostics',
          'source_closures': closures,
          'qualification': json.loads((root / 'neohookean-qualification-v2/results/summary.json').read_text()),
          'live_jobs_excluded': ['714003_0','714003_1','714003_2','714003_3','714067_0','714067_1'],
          'goal_completion': False}
print(json.dumps(report, indent=2))
PY
sacct -X -j 714051,714063,714067 -o JobIDRaw,State,ExitCode,Elapsed,NodeList -n -P > "$task_tools/jobs.txt"
sacct -X -j 714051,714063,714067 -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
task_extra=("$task_tools/source.sha256" "$task_tools/jobs.txt" "$task_tools/job-commands.txt"
    "$task_epoch/grasp-neohookean-v1.sbatch" "$task_epoch/environment-v1/environment.json")
while IFS= read -r -d '' task_source; do task_extra+=("$task_source"); done < <(find "$task_tools/source" -type f -print0)
for task_job in 714051 714063; do
    for task_path in "$task_epoch"/*"$task_job"*.log "$task_epoch"/*"$task_job"*.err; do
        if test -f "$task_path"; then task_extra+=("$task_path"); fi
    done
done
task_dest="research/rigid_stress/evidence/20261014-physical/neohookean-qualification-$task_version"
"$task_python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$RIGID_STRESS_DATA_ROOT/runs" --destination "$task_dest" \
    --staging-root "$task_epoch/publication-staging" \
    --run "$task_epoch/neohookean-qualification-v1" "$task_epoch/neohookean-qualification-v2" \
    --extra "${task_extra[@]}" --summary "$task_tools/summary.json" \
    --limitations "Bulk controls and saved-position audits qualify nonlinear assembly and reveal finite-rotation assumptions; they do not solve nonlinear contact or measure physical stress error. Live pad-only/all-part contact, original spatial axes and full-egg tasks are excluded. Objective pose/native-load replay, contact quadrature/spatial/regularization/history tails, actual Franka matching, rigid feedback and physical accuracy/cost remain required. Executable source-copy runner/patch are retained for reproducibility without altering live canonical references."
"$task_python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
