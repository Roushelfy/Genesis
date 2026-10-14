#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_size=${2:?Normalized shell mesh size required}
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{validation_meshes,skfem_reference,spherical_shell_validation,qualify_reference_scaling}.py
    research/rigid_stress/scripts/{run_shell_amg_tail,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
task_status=0
"$task_env/bin/python" -m research.rigid_stress.validation_meshes --shape cavity --size "$task_size" \
    --shell-layers 2 --output "$task_run/mesh.npz" > "$task_run/mesh.log" 2>&1 || task_status=$?
if test "$task_status" -eq 0; then
    "$task_env/bin/python" -m research.rigid_stress.qualify_reference_scaling --mesh "$task_run/mesh.npz" \
        --solver amg-cg --output "$task_run/solution" > "$task_run/reference.log" 2>&1 || task_status=$?
fi
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
printf '%s\n' "$task_status" > "$task_run/returncode.txt"
exit "$task_status"
