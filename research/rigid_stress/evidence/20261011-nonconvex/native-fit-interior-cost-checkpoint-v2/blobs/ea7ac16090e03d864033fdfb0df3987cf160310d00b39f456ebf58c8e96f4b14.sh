#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/native-fit-interior-v1/b2048-623102"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/engine/solvers/rigid/collider/*.py genesis/options/{rigid_stress,morphs}.py genesis/engine/solvers/rigid/rigid_solver.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/speed_benchmark/rigid_stress_metrics.py examples/rigid/assets/hollow_egg/level1/elastic.npz "$task_root/probe_native_fit_interior_v1.py" "$task_root/run_native_fit_interior_v1.sh")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git rev-parse HEAD > "$task_run/base-revision.txt"
git diff HEAD > "$task_run/source.patch"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" "$task_root/probe_native_fit_interior_v1.py" --conditions "$task_root/native-selected-matched-scopes-v1/b2048-623102/conditions.npz" --output "$task_run/result.json" > "$task_run/probe.log" 2>&1 || task_failed=1
echo "$task_failed" > "$task_run/returncode.txt"
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
