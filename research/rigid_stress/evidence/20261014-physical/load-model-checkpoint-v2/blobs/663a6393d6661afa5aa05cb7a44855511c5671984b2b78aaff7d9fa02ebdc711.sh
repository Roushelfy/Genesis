#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?New runtime directory required}
shift
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export GS_CACHE_FILE_PATH="$task_run/cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
export NUMBA_CACHE_DIR="$task_run/cache/numba" MPLCONFIGDIR="$task_run/cache/matplotlib"
task_sources=(research/rigid_stress/{skfem_reference,finite_traction_reference,native_load_samples,load_model_validation,qualify_reference_scaling,spherical_shell_validation,validation_meshes}.py
    research/rigid_stress/scripts/run_load_models.sh)
while IFS= read -r task_source; do task_sources+=("$task_source"); done < <(rg --files genesis/engine/solvers/rigid/stress)
task_sources+=(genesis/options/rigid_stress.py)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git rev-parse HEAD > "$task_run/base-revision.txt"
git diff HEAD > "$task_run/source.patch"
hostname > "$task_run/node.txt"
printf '%s\n' "$*" > "$task_run/arguments.txt"
task_status=0
"$task_env/bin/python" -m research.rigid_stress.load_model_validation \
    --output "$task_run/results" "$@" > "$task_run/validation.log" 2>&1 || task_status=$?
printf '%s\n' "$task_status" > "$task_run/returncode.txt"
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_status"
