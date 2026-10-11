#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/surface-fan-stable-v1"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/rigid_stress.py tests/rigid/test_stress_surface_fan.py "$task_root/diagnose_native_surface_fans_v9.py" "$task_root/traction_warp_candidate_v1.py")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/hardware.csv"
task_failed=0
for task_backend in cpu cuda; do
  export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/$task_backend/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/$task_backend/genesis"
  mkdir -p "$task_run/$task_backend"
  "$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend "$task_backend" --tb=short -q tests/rigid/test_stress_surface_fan.py > "$task_run/$task_backend/tests.log" 2>&1 || task_failed=1
  for task_case in v2 v3; do
    mkdir -p "$task_run/$task_backend/$task_case"
    "$RIGID_STRESS_ENV/bin/python" "$task_root/diagnose_native_surface_fans_v9.py" --backend "$task_backend" --run-dir "$task_root/online-surface-fan-gjk-patch-$task_case/results/failure" --anchor-fan --packed --output "$task_run/$task_backend/$task_case/diagnosis.json" > "$task_run/$task_backend/$task_case/diagnosis.log" 2>&1 || task_failed=1
  done
done
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
