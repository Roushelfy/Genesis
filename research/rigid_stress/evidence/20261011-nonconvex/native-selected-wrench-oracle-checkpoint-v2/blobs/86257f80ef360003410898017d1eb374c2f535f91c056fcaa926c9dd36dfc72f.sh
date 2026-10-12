#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_batch=${1:?Batch required}
task_seed=${2:?Seed required}
task_run="$task_root/native-selected-wrench-oracle-v1/b${task_batch}-${task_seed}"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
export RIGID_STRESS_SOURCE_REVISION="$(git rev-parse HEAD)+selected-wrench-oracle-v1"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/engine/solvers/rigid/collider/*.py genesis/engine/solvers/rigid/rigid_solver.py genesis/options/{rigid_stress,morphs}.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/speed_benchmark/rigid_stress{,_metrics}.py examples/rigid/assets/hollow_egg/level1/elastic.npz "$task_root/run_native_selected_wrench_oracle_v1.sh" "$task_root/probe_native_selected_wrench_oracle_v1.py")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git diff HEAD > "$task_run/source.patch"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$task_run/hardware.csv"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" "$task_root/probe_native_selected_wrench_oracle_v1.py" --envs "$task_batch" --seed "$task_seed" --output "$task_run/result.json" > "$task_run/profile.log" 2>&1 || task_failed=1
echo "$task_failed" > "$task_run/returncode.txt"
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
