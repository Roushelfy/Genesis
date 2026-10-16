#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_native=${2:?Immutable native snapshot required}
shift 2
test ! -e "$task_run"
mkdir -p "$task_run/source/research/rigid_stress" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export GS_CACHE_FILE_PATH="$task_run/cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
export QD_OFFLINE_CACHE=0 NUMBA_CACHE_DIR="$task_run/cache/numba" MPLCONFIGDIR="$task_run/cache/matplotlib"
export CUDA_CACHE_PATH="$task_run/cache/cuda"
task_sources=(research/rigid_stress/practical_reward_proxy.py
    research/rigid_stress/scripts/{run_practical_reward_proxy,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/input-source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
touch "$task_run/source/research/__init__.py" "$task_run/source/research/rigid_stress/__init__.py"
task_actual=("${task_sources[@]}" research/__init__.py research/rigid_stress/__init__.py)
(cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source.sha256"
(cd "$task_native"; sha256sum -c snapshot-files.sha256) > "$task_run/native-snapshot-before.txt"
cp "$task_native/snapshot.json" "$task_native/snapshot-files.sha256" "$task_run/"
task_finish() {
    task_status=$?
    trap - EXIT
    set +e
    printf '%s\n' "$task_status" > "$task_run/program-returncode.txt"
    (cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source-after.sha256"
    cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
    task_check=$?
    sha256sum "${task_sources[@]}" > "$task_run/input-source-after.sha256"
    cmp "$task_run/input-source.sha256" "$task_run/input-source-after.sha256"
    task_input_check=$?
    (cd "$task_native"; sha256sum -c snapshot-files.sha256) > "$task_run/native-snapshot-after.txt"
    task_native_check=$?
    if test "$task_status" = 0; then
        if test "$task_check" != 0; then task_status=$task_check
        elif test "$task_input_check" != 0; then task_status=$task_input_check
        elif test "$task_native_check" != 0; then task_status=$task_native_check
        fi
    fi
    printf '%s\n' "$task_status" > "$task_run/returncode.txt"
    exit "$task_status"
}
trap task_finish EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$task_run/hardware.csv"
(
    cd "$task_run/source"
    export PYTHONPATH="$PWD:$task_native"
    "$task_env/bin/python" - "$task_native" <<'PY' > "$task_run/import-origins.txt"
import sys
from pathlib import Path
import genesis
from examples.rigid import franka_egg_stress
assert Path(genesis.__file__).resolve().is_relative_to(Path(sys.argv[1]))
assert Path(franka_egg_stress.__file__).resolve().is_relative_to(Path(sys.argv[1]))
print(genesis.__file__)
print(franka_egg_stress.__file__)
PY
    "$task_env/bin/python" -m research.rigid_stress.practical_reward_proxy \
        --output "$task_run/results" "$@" > "$task_run/reward-proxy.log" 2>&1
)
