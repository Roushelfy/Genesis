#!/usr/bin/env bash
# Validate the complete native stress feature from this worktree on an allocation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

task_backend=${1:?Backend required}
task_run=${2:?New runtime directory required}
shift 2
[[ "$task_backend" == cpu || "$task_backend" == cuda ]]
test ! -e "$task_run"
source research/rigid_stress/scripts/cluster_env.sh
mkdir -p "$task_run/tmp" "$task_run/source"
task_run=$(realpath "$task_run")
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
export GS_CACHE_FILE_PATH="$task_run/cache/genesis"
mapfile -t task_sources < <(git ls-files ':(glob)genesis/**/*.py' ':(glob)tests/**/*.py' ':(glob)research/rigid_stress/*.py')
task_sources+=(tests/rigid/assets/*.npz
    examples/rigid/assets/hollow_egg/level1/elastic.npz research/rigid_stress/scripts/check_nonconvex_final.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git rev-parse HEAD > "$task_run/base-revision.txt"
git diff HEAD -- "${task_sources[@]}" > "$task_run/source.patch"
hostname > "$task_run/node.txt"
"$RIGID_STRESS_ENV/bin/python" -c 'import genesis, sys; print(sys.version); print(genesis.__file__)' > "$task_run/import.txt"
if [[ "$task_backend" == cuda ]]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$task_run/hardware.csv"
fi
if [[ "$#" -gt 0 ]]; then
    task_tests=("$@")
else
    task_tests=(tests/rigid/test_stress*.py)
fi
printf '%q ' "$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend "$task_backend" --tb=short -q --junitxml "$task_run/report.xml" -o junit_family=legacy "${task_tests[@]}" > "$task_run/command.txt"
printf '\n' >> "$task_run/command.txt"
task_status=0
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend "$task_backend" --tb=short -q \
    --junitxml "$task_run/report.xml" -o junit_family=legacy "${task_tests[@]}" > "$task_run/tests.log" 2>&1 || task_status=$?
printf '%s\n' "$task_status" > "$task_run/returncode.txt"
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_status"
