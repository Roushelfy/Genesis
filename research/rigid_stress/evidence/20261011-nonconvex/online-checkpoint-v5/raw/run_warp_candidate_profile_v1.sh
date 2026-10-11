#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/warp-candidate-profile-v1"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1 QD_KERNEL_PROFILER=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/{rigid_stress,morphs}.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/speed_benchmark/rigid_stress.py "$task_root/traction_warp_candidate_v1.py" "$task_root/probe_warp_candidate_profile_v1.py")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/hardware.csv"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" examples/speed_benchmark/rigid_stress.py --scope profile --envs 16 --warmup 180 --steps 20 --seed 623001 --varied --contact-model intrinsic --contact-block-size 4 --egg-geometry mesh64 --collision-method gjk --contact-patch on --save-conditions "$task_run/conditions.npz" --compact-log --source-revision 1fe4f804+contact-patch-cli --output "$task_run/scalar.json" > "$task_run/scalar.log" 2>&1 || task_failed=1
if [[ -e "$task_run/scalar.json" ]]; then
    "$RIGID_STRESS_ENV/bin/python" "$task_root/probe_warp_candidate_profile_v1.py" --scope profile --envs 16 --warmup 180 --steps 20 --seed 623001 --varied --contact-model intrinsic --contact-block-size 4 --egg-geometry mesh64 --collision-method gjk --contact-patch on --conditions "$task_run/conditions.npz" --compact-log --source-revision 1fe4f804+contact-patch-cli+experimental-warp --output "$task_run/warp.json" > "$task_run/warp.log" 2>&1 || task_failed=1
fi
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
