#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?New runtime directory required}
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{skfem_reference,validation_meshes,spherical_shell_validation}.py
    research/rigid_stress/scripts/run_shell_validation.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git rev-parse HEAD > "$task_run/base-revision.txt"
git diff HEAD > "$task_run/source.patch"
task_status=0
"$task_env/bin/python" -m research.rigid_stress.spherical_shell_validation \
    --output "$task_run/sweep" > "$task_run/validation.log" 2>&1 || task_status=$?
printf '%s\n' "$task_status" > "$task_run/returncode.txt"
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_status"
