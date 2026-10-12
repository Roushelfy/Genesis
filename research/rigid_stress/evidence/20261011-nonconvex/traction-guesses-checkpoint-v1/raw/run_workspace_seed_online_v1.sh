#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-refinement
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_failed=0
for task_envs in 16 32; do
    task_seed=623001
    if [[ "$task_envs" == 32 ]]; then task_seed=623101; fi
    task_run="$task_root/workspace-seed-online-b${task_envs}-v1"
    test ! -e "$task_run"
    mkdir -p "$task_run/tmp" "$task_run/source"
    export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
    export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
    task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/rigid_stress.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/rigid/assets/hollow_egg/level1/elastic.npz research/rigid_stress/{probe_nonconvex_online,wrench,mechanics,field_cpu}.py "$task_root/traction_workspace_seed_candidate_v1.py" "$task_root/probe_workspace_seed_online_v1.py" "$task_root/run_workspace_seed_online_v1.sh")
    sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
    cp --parents "${task_sources[@]}" "$task_run/source"
    hostname > "$task_run/node.txt"
    nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$task_run/hardware.csv"
    "$RIGID_STRESS_ENV/bin/python" "$task_root/probe_workspace_seed_online_v1.py" --envs "$task_envs" --seed "$task_seed" --steps 1280 --oracle-stride 160 --intrinsic-support pair_union --contact-block-size 4 --egg-geometry mesh64 --collision-method gjk --contact-patch on --output "$task_run/results" > "$task_run/probe.log" 2>&1 || task_failed=1
    sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
    cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
done
exit "$task_failed"
