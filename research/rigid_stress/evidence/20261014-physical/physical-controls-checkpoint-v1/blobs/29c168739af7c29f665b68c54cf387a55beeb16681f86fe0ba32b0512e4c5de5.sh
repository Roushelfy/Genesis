#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?New runtime directory required}
task_style=${2:?Academic figure style module required}
shift 2
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1 MPLCONFIGDIR="$task_run/cache/matplotlib"
task_sources=(research/rigid_stress/{plot_physical_controls,spherical_shell_validation,skfem_reference,validation_meshes}.py
    research/rigid_stress/scripts/plot_physical_controls.sh)
sha256sum "${task_sources[@]}" "$task_style" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
cp "$task_style" "$task_run/source/nature_mpl_style.py"
hostname > "$task_run/node.txt"
task_status=0
"$task_env/bin/python" -m research.rigid_stress.plot_physical_controls \
    --style-module "$task_run/source/nature_mpl_style.py" \
    --output "$task_run/figures" "$@" > "$task_run/render.log" 2>&1 || task_status=$?
printf '%s\n' "$task_status" > "$task_run/returncode.txt"
sha256sum "${task_sources[@]}" "$task_style" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_status"
