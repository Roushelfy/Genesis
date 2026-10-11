#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/square-profiles-v1"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/rigid_stress.py tests/rigid/{test_stress_surface_fan,test_stress_footprint}.py "$task_root/diagnose_square_profiles_v1.py")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
task_failed=0
for task_backend in cpu cuda; do
  export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/$task_backend/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/$task_backend/genesis"
  task_tick=147
  if [[ "$task_backend" == cuda ]]; then task_tick=454; fi
  "$RIGID_STRESS_ENV/bin/python" "$task_root/diagnose_square_profiles_v1.py" --backend "$task_backend" --tick "$task_tick" --output "$task_run/$task_backend" > "$task_run/$task_backend.log" 2>&1 || task_failed=1
done
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
