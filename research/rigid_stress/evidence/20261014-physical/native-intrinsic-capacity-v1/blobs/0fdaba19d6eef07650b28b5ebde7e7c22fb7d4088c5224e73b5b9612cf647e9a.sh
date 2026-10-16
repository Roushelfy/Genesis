#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_style=${2:?Academic Matplotlib style required}
shift 2
test ! -e "$task_run"
mkdir -p "$task_run/input-source" "$task_run/source" "$task_run/tmp" "$task_run/cache/mpl"
export TMPDIR="$task_run/tmp" MPLCONFIGDIR="$task_run/cache/mpl" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{plot_recovered_spatial_state,plot_franka_spatial,skfem_reference,summarize_fine_references}.py
    research/rigid_stress/scripts/{plot_recovered_spatial_state,cluster_env}.sh)
sha256sum "${task_sources[@]}" "$task_style" > "$task_run/input-source.sha256"
cp --parents "${task_sources[@]}" "$task_run/input-source"
cp "$task_style" "$task_run/input-source/nature_mpl_style.py"
cp -a "$task_run/input-source/." "$task_run/source/"
touch "$task_run/source/research/__init__.py" "$task_run/source/research/rigid_stress/__init__.py"
task_actual=("${task_sources[@]}" nature_mpl_style.py research/__init__.py research/rigid_stress/__init__.py)
(cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source.sha256"
trap 'task_status=$?; (cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source-after.sha256"; sha256sum "${task_sources[@]}" "$task_style" > "$task_run/input-source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; cmp "$task_run/input-source.sha256" "$task_run/input-source-after.sha256"; exit "$task_status"' EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
(cd "$task_run/source"
    export PYTHONPATH="$task_run/source"
    "$task_env/bin/python" -m research.rigid_stress.plot_recovered_spatial_state --style "$task_run/source/nature_mpl_style.py" \
        --output "$task_run/figure" "$@" > "$task_run/plot.log" 2>&1)
