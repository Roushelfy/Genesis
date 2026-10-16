#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
shift
test ! -e "$task_run"
mkdir -p "$task_run/input-source" "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{compare_native_finite_axes,summarize_fine_references}.py
    research/rigid_stress/scripts/{compare_native_finite_axes,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/input-source.sha256"
cp --parents "${task_sources[@]}" "$task_run/input-source"
cp -a "$task_run/input-source/." "$task_run/source/"
touch "$task_run/source/research/__init__.py" "$task_run/source/research/rigid_stress/__init__.py"
task_actual=("${task_sources[@]}" research/__init__.py research/rigid_stress/__init__.py)
(cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source.sha256"
task_finish() {
    task_status=$?
    trap - EXIT
    set +e
    printf '%s\n' "$task_status" > "$task_run/program-returncode.txt"
    (cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source-after.sha256"
    sha256sum "${task_sources[@]}" > "$task_run/input-source-after.sha256"
    cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
    task_source_status=$?
    cmp "$task_run/input-source.sha256" "$task_run/input-source-after.sha256"
    task_input_status=$?
    if test "$task_status" = 0; then
        if test "$task_source_status" != 0; then task_status=$task_source_status
        elif test "$task_input_status" != 0; then task_status=$task_input_status
        fi
    fi
    printf '%s\n' "$task_status" > "$task_run/returncode.txt"
    exit "$task_status"
}
trap task_finish EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
lscpu > "$task_run/cpu.txt"
"$task_env/bin/python" - <<'PY' > "$task_run/environment.json"
import importlib.metadata,json,platform,sys
print(json.dumps({'python':sys.version,'platform':platform.platform(),
                 'packages':{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-fem']}},indent=2))
PY
(cd "$task_run/source"
    export PYTHONPATH="$task_run/source"
    "$task_env/bin/python" -m research.rigid_stress.compare_native_finite_axes \
        --output "$task_run/results" "$@" > "$task_run/comparison.log" 2>&1)
