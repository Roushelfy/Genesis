#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/native-pair-safety-v2"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/{rigid_stress,morphs}.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py tests/rigid/test_stress_shared_lifecycle.py "$task_root/probe_recovery_rigid_immutability_v3.py")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/hardware.csv"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" -m pytest -q -n 0 --tb=short --backend cuda tests/rigid/test_stress_shared_lifecycle.py > "$task_run/tests.log" 2>&1 || task_failed=1
if [[ "$task_failed" == 0 ]]; then
    "$RIGID_STRESS_ENV/bin/python" "$task_root/probe_recovery_rigid_immutability_v3.py" > "$task_run/immutability.log" 2>&1 || task_failed=1
fi
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
