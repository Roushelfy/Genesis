#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/exact-602-replay-v1"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/rigid_stress.py research/rigid_stress/wrench.py tests/rigid/test_stress_surface_fan.py "$task_root/traction_warp_candidate_v2.py" "$task_root/traction_warp_candidate_v4.py" "$task_root/traction_warp_candidate_v5.py" "$task_root/traction_warp_candidate_v6.py" "$task_root/traction_warp_candidate_v7.py" "$task_root/probe_exact_602_v1.py")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/hardware.csv"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" "$task_root/probe_exact_602_v1.py" > "$task_run/probe.log" 2>&1 || task_failed=1
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
