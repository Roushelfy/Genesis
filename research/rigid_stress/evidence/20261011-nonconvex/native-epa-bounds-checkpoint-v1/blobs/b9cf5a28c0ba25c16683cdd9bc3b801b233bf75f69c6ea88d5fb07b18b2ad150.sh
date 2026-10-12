#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-refinement
source research/rigid_stress/scripts/cluster_env.sh
export HF_HOME=/mnt/data/zhaofeng/hf_home
task_backend=${1:?Backend required}
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/epa-fp32-baseline-v1/$task_backend"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_sources=(genesis/engine/solvers/rigid/collider/*.py tests/rigid/test_collision.py "$task_root/run_epa_fp32_baseline_v1.sh")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git diff HEAD > "$task_run/source.patch"
hostname > "$task_run/node.txt"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend "$task_backend" --tb=short -q --junitxml "$task_run/report.xml" tests/rigid/test_collision.py::test_convex_collision_across_geom_scales -k True > "$task_run/tests.log" 2>&1 || task_failed=1
echo "$task_failed" > "$task_run/returncode.txt"
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
