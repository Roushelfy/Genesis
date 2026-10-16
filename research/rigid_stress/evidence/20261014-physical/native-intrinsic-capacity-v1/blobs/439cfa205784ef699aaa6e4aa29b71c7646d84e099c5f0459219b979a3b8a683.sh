#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
shift
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1 MPLCONFIGDIR="$task_run/cache/matplotlib"
task_sources=(research/rigid_stress/{plot_franka_reference,plot_physical_controls,summarize_fine_references,spherical_shell_validation,skfem_reference,validation_meshes}.py
    research/rigid_stress/scripts/{plot_franka_reference,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; exit "$task_status"' EXIT
"$task_env/bin/python" -m research.rigid_stress.plot_franka_reference --output "$task_run/results" "$@" \
    > "$task_run/plot.log" 2>&1
