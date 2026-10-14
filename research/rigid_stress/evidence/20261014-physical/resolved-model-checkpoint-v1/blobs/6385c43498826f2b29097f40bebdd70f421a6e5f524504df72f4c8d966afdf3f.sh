#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
shift
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export GS_CACHE_FILE_PATH="$task_run/cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
export NUMBA_CACHE_DIR="$task_run/cache/numba"
task_sources=(research/rigid_stress/{map_resolved_grasp,native_physical_sequence,finite_traction_reference,load_model_validation,profile_reference_sensitivity,skfem_reference,native_load_samples,qualify_reference_scaling,spherical_shell_validation,validation_meshes,summarize_fine_references}.py
    research/rigid_stress/scripts/{map_resolved_grasp,cluster_env}.sh genesis/options/rigid_stress.py)
while IFS= read -r task_source; do task_sources+=("$task_source"); done < <("$task_env/bin/python" -c 'from pathlib import Path; print("\n".join(str(p) for p in Path("genesis/engine/solvers/rigid/stress").rglob("*.py")))')
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; exit "$task_status"' EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$task_run/hardware.csv"
"$task_env/bin/python" -m research.rigid_stress.map_resolved_grasp --output "$task_run/results" "$@" \
    > "$task_run/mapping.log" 2>&1
