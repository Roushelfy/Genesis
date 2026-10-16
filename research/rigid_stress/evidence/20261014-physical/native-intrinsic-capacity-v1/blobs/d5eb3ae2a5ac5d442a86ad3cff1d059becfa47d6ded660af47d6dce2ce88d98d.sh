#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_style=${2:?Academic Matplotlib style source required}
shift 2
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp" "$task_run/cache/mpl"
export TMPDIR="$task_run/tmp" MPLCONFIGDIR="$task_run/cache/mpl" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{plot_franka_spatial,skfem_reference,summarize_fine_references}.py
    research/rigid_stress/scripts/{plot_franka_spatial,cluster_env}.sh)
sha256sum "${task_sources[@]}" "$task_style" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
cp "$task_style" "$task_run/source/nature_mpl_style.py"
trap 'task_status=$?; sha256sum "${task_sources[@]}" "$task_style" > "$task_run/source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; exit "$task_status"' EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
"$task_env/bin/python" -m research.rigid_stress.plot_franka_spatial --style "$task_run/source/nature_mpl_style.py" \
    --output "$task_run/figure" "$@" > "$task_run/plot.log" 2>&1
