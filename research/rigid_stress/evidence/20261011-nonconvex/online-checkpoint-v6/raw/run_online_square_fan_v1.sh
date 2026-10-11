#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/online-square-fan-v1"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/{rigid_stress,morphs}.py genesis/engine/entities/rigid_entity/description.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py research/rigid_stress/probe_nonconvex_online.py)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/hardware.csv"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/probe_nonconvex_online.py --envs 32 --steps 1280 --seed 623101 --contact-block-size 4 --oracle-stride 160 --egg-geometry mesh64 --collision-method gjk --contact-patch on --output "$task_run/results" > "$task_run/probe.log" 2>&1 || task_failed=1
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
