#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/shared-edge-diagnostic-v4"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/rigid_stress.py tests/rigid/test_stress_{source_priors,sources}.py tests/rigid/assets/nonconvex_shared_edge_roundoff.npz examples/rigid/assets/hollow_egg/level1/elastic.npz "$task_root/run_shared_edge_diagnostic_v4.sh")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git diff HEAD > "$task_run/source.patch"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend cpu -q -x --tb=short --junitxml "$task_run/report.xml" -o junit_family=legacy tests/rigid/test_stress_source_priors.py tests/rigid/test_stress_sources.py -k roundoff > "$task_run/tests.log" 2>&1 || task_failed=1
echo "$task_failed" > "$task_run/returncode.txt"
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
