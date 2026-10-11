#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/singular-fallback-warp-v7-v1"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1 QD_KERNEL_PROFILER=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/{rigid_stress,morphs}.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/speed_benchmark/rigid_stress.py research/rigid_stress/{wrench,mechanics,field_cpu}.py tests/rigid/{test_stress_shared_support,test_stress_shared_lifecycle,test_stress_surface_fan,test_stress_traction}.py "$task_root/traction_warp_candidate_v7.py" "$task_root/traction_sample_warp_candidate_v1.py" "$task_root/probe_warp_v7_v1.py")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/hardware.csv"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" "$task_root/probe_warp_v7_v1.py" --test > "$task_run/tests.log" 2>&1 || task_failed=1
if [[ "$task_failed" == 0 ]]; then
    "$RIGID_STRESS_ENV/bin/python" "$task_root/probe_warp_v7_v1.py" --scope profile --envs 32 --warmup 600 --steps 20 --seed 623101 --varied --contact-model intrinsic --intrinsic-support pair_union --contact-block-size 4 --egg-geometry mesh64 --collision-method gjk --contact-patch on --conditions "$task_root/warp-candidate-pair-v2/conditions.npz" --compact-log --source-revision 9db33909+experimental-unregularized-newton --output "$task_run/profile.json" > "$task_run/profile.log" 2>&1 || task_failed=1
fi
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
