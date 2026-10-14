#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-refinement
source research/rigid_stress/scripts/cluster_env.sh
task_backend="$1"
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/aggregate-wrench-fields-v1/$task_backend"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/rigid_stress.py tests/rigid/{test_stress_shared_support,test_stress_surface_fan,test_stress_footprint}.py tests/rigid/assets/{nonconvex_mesh64_tick602,nonconvex_policy_tick876}.npz examples/rigid/assets/hollow_egg/level1/elastic.npz research/rigid_stress/{wrench,mechanics,field_cpu}.py "$task_root/probe_aggregate_wrench_fields_v1.py" "$task_root/run_aggregate_wrench_fields_v1.sh")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
if [[ "$task_backend" == cuda ]]; then nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$task_run/hardware.csv"; fi
task_failed=0
"$RIGID_STRESS_ENV/bin/python" "$task_root/probe_aggregate_wrench_fields_v1.py" "$task_backend" "$task_run/results" > "$task_run/diagnostic.log" 2>&1 || task_failed=1
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
