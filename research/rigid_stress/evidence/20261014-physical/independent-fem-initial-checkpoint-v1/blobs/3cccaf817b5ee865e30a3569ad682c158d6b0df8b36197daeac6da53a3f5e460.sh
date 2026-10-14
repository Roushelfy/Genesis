#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?New runtime directory required}
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source" "$task_run/meshes"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/validation_meshes.py research/rigid_stress/scripts/generate_physical_meshes.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git rev-parse HEAD > "$task_run/base-revision.txt"
git diff HEAD > "$task_run/source.patch"
task_status=0
for task_case in ellipsoid:0.006 egg:0.45 capsule:0.006 u:0.004 thin_neck:0.006 cavity:0.6 narrow_gap:0.004; do
    task_shape=${task_case%:*}
    task_size=${task_case#*:}
    task_case_status=0
    "$task_env/bin/python" -m research.rigid_stress.validation_meshes \
        --shape "$task_shape" --size "$task_size" --shell-layers 2 \
        --output "$task_run/meshes/$task_shape.npz" > "$task_run/$task_shape.log" 2>&1 || task_case_status=$?
    printf '%s\n' "$task_case_status" > "$task_run/$task_shape.returncode"
    if [[ "$task_case_status" != 0 ]]; then task_status=1; fi
done
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
printf '%s\n' "$task_status" > "$task_run/returncode.txt"
exit "$task_status"
