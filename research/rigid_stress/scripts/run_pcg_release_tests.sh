#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_backend=${2:?CPU or GPU pytest backend required}
shift 2
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export GS_CACHE_FILE_PATH="$task_run/cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
export NUMBA_CACHE_DIR="$task_run/cache/numba"
task_sources=(tests/rigid/test_stress.py tests/rigid/test_stress_nonconvex.py tests/conftest.py
    research/rigid_stress/scripts/{run_pcg_release_tests,cluster_env}.sh genesis/options/rigid_stress.py)
while IFS= read -r task_source; do task_sources+=("$task_source"); done < <("$task_env/bin/python" -c 'from pathlib import Path; print("\n".join(str(p) for p in Path("genesis/engine/solvers/rigid/stress").rglob("*.py")))')
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git diff HEAD > "$task_run/source.patch"
hostname > "$task_run/node.txt"
task_status=0
"$task_env/bin/python" -m pytest tests/rigid/test_stress.py --backend "$task_backend" -k pcg_balanced_grip_release \
    -q --color=no --basetemp "$task_run/tmp/pytest" -o "cache_dir=$task_run/cache/pytest" \
    --junitxml "$task_run/results.xml" "$@" > "$task_run/pytest.log" 2>&1 || task_status=$?
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
printf '%s\n' "$task_status" > "$task_run/returncode.txt"
exit "$task_status"
