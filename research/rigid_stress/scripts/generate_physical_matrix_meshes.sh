#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?New runtime directory required}
task_shape=${2:?Full geometry name required}
shift 2
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source" "$task_run/meshes"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/validation_meshes.py research/rigid_stress/scripts/generate_physical_matrix_meshes.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
case "$task_shape" in
    egg) task_sizes=(0.15 0.11 0.08) ;;
    cavity) task_sizes=(0.28 0.20 0.15) ;;
    u|narrow_gap) task_sizes=(0.0035 0.0025 0.00175) ;;
    ellipsoid|capsule|thin_neck) task_sizes=(0.005 0.0035 0.0025) ;;
    *) printf 'Unknown geometry: %s\n' "$task_shape" >&2; exit 2 ;;
esac
if test "$#" -gt 0; then task_sizes=("$@"); fi
task_status=0
for task_size in "${task_sizes[@]}"; do
    task_case_status=0
    "$task_env/bin/python" -m research.rigid_stress.validation_meshes \
        --shape "$task_shape" --size "$task_size" --shell-layers 2 \
        --output "$task_run/meshes/h${task_size}.npz" > "$task_run/h${task_size}.log" 2>&1 || task_case_status=$?
    printf '%s\n' "$task_case_status" > "$task_run/h${task_size}.returncode"
    if [[ "$task_case_status" != 0 ]]; then task_status=1; fi
done
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
printf '%s\n' "$task_status" > "$task_run/returncode.txt"
exit "$task_status"
