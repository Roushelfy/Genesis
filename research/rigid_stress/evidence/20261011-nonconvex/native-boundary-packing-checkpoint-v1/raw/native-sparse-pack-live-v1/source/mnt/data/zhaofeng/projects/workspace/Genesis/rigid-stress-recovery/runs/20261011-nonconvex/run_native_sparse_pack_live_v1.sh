#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-refinement
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/native-sparse-pack-live-v1"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
unset QD_KERNEL_PROFILER
task_conditions="$task_root/original-point-scalar-b32-profile-v1/conditions.npz"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/{rigid_stress,morphs}.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/speed_benchmark/rigid_stress.py examples/rigid/assets/hollow_egg/level1/elastic.npz "$task_conditions" "$task_root/parallel_contact_clear_candidate_v1.py" "$task_root/probe_parallel_contact_clear_v1.py" "$task_root/run_native_sparse_pack_live_v1.sh")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$task_run/hardware.csv"
task_failed=0
task_revision=$(git rev-parse HEAD)
for task_variant in linear_scan native_hybrid; do
    task_case="$task_run/$task_variant"
    mkdir -p "$task_case/tmp"
    export TMPDIR="$task_case/tmp"
    export QD_OFFLINE_CACHE_FILE_PATH="$task_case/cache/quadrants" GS_CACHE_FILE_PATH="$task_case/cache/genesis"
    task_driver=(examples/speed_benchmark/rigid_stress.py)
    if [[ "$task_variant" == linear_scan ]]; then task_driver=("$task_root/probe_parallel_contact_clear_v1.py" benchmark); fi
    "$RIGID_STRESS_ENV/bin/python" "${task_driver[@]}" --scope live --envs 32 --warmup 600 --steps 1280 --repetitions 3 --minimum-seconds 10 --seed 623101 --varied --contact-model intrinsic --intrinsic-support pair_union --contact-block-size 4 --egg-geometry mesh64 --collision-method gjk --contact-patch on --conditions "$task_conditions" --compact-log --source-revision "$task_revision+boundary-fallback+native-packing-$task_variant" --output "$task_case/result.json" > "$task_case/benchmark.log" 2>&1 || task_failed=1
done
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
