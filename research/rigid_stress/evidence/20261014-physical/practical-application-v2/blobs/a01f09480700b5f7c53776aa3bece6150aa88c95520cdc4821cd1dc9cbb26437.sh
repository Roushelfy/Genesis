#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_style=${2:?Path to the academic figure nature_mpl_style.py utility required}
shift 2
test ! -e "$task_run"
mkdir -p "$task_run/source/research/rigid_stress" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg MPLCONFIGDIR="$task_run/cache/matplotlib"
task_sources=(research/rigid_stress/{analyse_practical_application,skfem_reference,summarize_fine_references}.py
    research/rigid_stress/scripts/{analyse_practical_application,cluster_env}.sh)
sha256sum "${task_sources[@]}" "$task_style" > "$task_run/input-source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
cp "$task_style" "$task_run/source/nature_mpl_style.py"
touch "$task_run/source/research/__init__.py" "$task_run/source/research/rigid_stress/__init__.py"
task_actual=("${task_sources[@]}" nature_mpl_style.py research/__init__.py research/rigid_stress/__init__.py)
(cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source.sha256"
task_finish() {
    task_status=$?
    trap - EXIT
    set +e
    printf '%s\n' "$task_status" > "$task_run/program-returncode.txt"
    (cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source-after.sha256"
    cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
    task_check=$?
    sha256sum "${task_sources[@]}" "$task_style" > "$task_run/input-source-after.sha256"
    cmp "$task_run/input-source.sha256" "$task_run/input-source-after.sha256"
    task_input_check=$?
    if test "$task_status" = 0; then
        if test "$task_check" != 0; then task_status=$task_check
        elif test "$task_input_check" != 0; then task_status=$task_input_check
        fi
    fi
    printf '%s\n' "$task_status" > "$task_run/returncode.txt"
    exit "$task_status"
}
trap task_finish EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
(
    cd "$task_run/source"
    export PYTHONPATH="$PWD"
    "$task_env/bin/python" -m research.rigid_stress.analyse_practical_application \
        --output "$task_run/results" "$@" > "$task_run/analysis.log" 2>&1
)
