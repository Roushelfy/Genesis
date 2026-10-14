#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/copied-native-four-scopes-b1024-v1"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
unset QD_KERNEL_PROFILER
task_conditions="$task_root/original-point-scalar-b32-profile-v1/conditions.npz"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/{rigid_stress,morphs}.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/speed_benchmark/rigid_stress.py examples/rigid/assets/hollow_egg/level1/elastic.npz "$task_conditions" "$task_root/run_copied_native_four_scopes_b1024_v1.sh")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git rev-parse HEAD > "$task_run/revision.txt"
git status --short > "$task_run/working-tree.txt"
test ! -s "$task_run/working-tree.txt"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$task_run/hardware.csv"
task_revision=$(cat "$task_run/revision.txt")
task_failed=0
for task_scope in rigid recovery live policy; do
    task_case="$task_run/$task_scope"
    mkdir -p "$task_case/tmp"
    export TMPDIR="$task_case/tmp"
    export QD_OFFLINE_CACHE_FILE_PATH="$task_case/cache/quadrants" GS_CACHE_FILE_PATH="$task_case/cache/genesis"
    task_returncode=0
    "$RIGID_STRESS_ENV/bin/python" examples/speed_benchmark/rigid_stress.py --scope "$task_scope" --envs 1024 --warmup 600 --steps 1280 --repetitions 3 --minimum-seconds 10 --seed 623101 --varied --contact-model intrinsic --intrinsic-support pair_union --contact-block-size 4 --egg-geometry mesh64 --collision-method gjk --contact-patch on --conditions "$task_conditions" --compact-log --source-revision "$task_revision+copied-bank32" --output "$task_case/result.json" > "$task_case/benchmark.log" 2>&1 || task_returncode=$?
    printf '%s\n' "$task_returncode" > "$task_case/returncode.txt"
    if [[ "$task_returncode" != 0 ]]; then task_failed=1; fi
done
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
