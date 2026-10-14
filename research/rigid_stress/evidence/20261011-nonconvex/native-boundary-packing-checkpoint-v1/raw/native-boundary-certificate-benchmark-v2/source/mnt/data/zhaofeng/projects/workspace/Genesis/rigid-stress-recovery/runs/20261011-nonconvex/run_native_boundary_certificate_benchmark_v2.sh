#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-refinement
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/native-boundary-certificate-benchmark-v2"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_conditions="$task_root/original-point-scalar-b32-profile-v1/conditions.npz"
task_legacy="$task_root/parallel-contact-clear-benchmark-v1/source/genesis/engine/solvers/rigid/stress/compatibility.py"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/{rigid_stress,morphs}.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/speed_benchmark/rigid_stress.py examples/rigid/assets/hollow_egg/level1/elastic.npz "$task_conditions" "$task_legacy" "$task_root/probe_sampled_wrench_search_v2.py" "$task_root/run_native_boundary_certificate_benchmark_v2.sh")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$task_run/hardware.csv"
task_failed=0
task_revision=$(git rev-parse HEAD)
for task_scope in profile live; do
    task_steps=1280
    if [[ "$task_scope" == profile ]]; then task_steps=128; export QD_KERNEL_PROFILER=1; else unset QD_KERNEL_PROFILER; fi
    for task_variant in sampled boundary; do
        task_case="$task_run/$task_scope/$task_variant"
        mkdir -p "$task_case/tmp"
        export TMPDIR="$task_case/tmp"
        export QD_OFFLINE_CACHE_FILE_PATH="$task_case/cache/quadrants" GS_CACHE_FILE_PATH="$task_case/cache/genesis"
        task_driver=(examples/speed_benchmark/rigid_stress.py)
        if [[ "$task_variant" == sampled ]]; then task_driver=("$task_root/probe_sampled_wrench_search_v2.py"); fi
        "$RIGID_STRESS_ENV/bin/python" "${task_driver[@]}" --scope "$task_scope" --envs 32 --warmup 600 --steps "$task_steps" --repetitions 3 --minimum-seconds 10 --seed 623101 --varied --contact-model intrinsic --intrinsic-support pair_union --contact-block-size 4 --egg-geometry mesh64 --collision-method gjk --contact-patch on --conditions "$task_conditions" --compact-log --source-revision "$task_revision+boundary-certificate-$task_variant" --output "$task_case/result.json" > "$task_case/benchmark.log" 2>&1 || task_failed=1
    done
done
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
