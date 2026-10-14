#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_style=${2:?Academic style module required}
shift 2
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" MPLCONFIGDIR="$task_run/cache/matplotlib" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{plot_fine_references,plot_physical_controls,spherical_shell_validation,skfem_reference,validation_meshes}.py
    research/rigid_stress/scripts/plot_fine_references.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
cp "$task_style" "$task_run/source/style.py"
sha256sum "$task_style" > "$task_run/style.sha256"
task_status=0
"$task_env/bin/python" -m research.rigid_stress.plot_fine_references --style-module "$task_run/source/style.py" \
    --output "$task_run/results" "$@" > "$task_run/plot.log" 2>&1 || task_status=$?
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
printf '%s\n' "$task_status" > "$task_run/returncode.txt"
exit "$task_status"
