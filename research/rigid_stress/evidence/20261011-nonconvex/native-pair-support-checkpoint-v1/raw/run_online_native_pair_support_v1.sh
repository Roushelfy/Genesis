#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/online-native-pair-support-v1"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/{rigid_stress,morphs}.py genesis/engine/entities/rigid_entity/description.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py research/rigid_stress/{probe_nonconvex_online,wrench,mechanics,field_cpu}.py)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/hardware.csv"
for task_spec in '16 623001' '32 623101'; do
    read -r task_batch task_seed <<< "$task_spec"
    task_scope="$task_run/b${task_batch}-seed${task_seed}"
    mkdir -p "$task_scope"
    (export QD_OFFLINE_CACHE_FILE_PATH="$task_scope/cache/quadrants" GS_CACHE_FILE_PATH="$task_scope/cache/genesis"; "$RIGID_STRESS_ENV/bin/python" research/rigid_stress/probe_nonconvex_online.py --envs "$task_batch" --steps 1280 --seed "$task_seed" --contact-block-size 4 --intrinsic-support pair_union --oracle-stride 160 --egg-geometry mesh64 --collision-method gjk --contact-patch on --output "$task_scope/results" > "$task_scope/probe.log" 2>&1) &
    if [[ "$task_batch" == 16 ]]; then task_first_pid=$!; else task_second_pid=$!; fi
done
task_failed=0
wait "$task_first_pid" || task_failed=1
wait "$task_second_pid" || task_failed=1
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
