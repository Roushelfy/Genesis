#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
for task_tests in "$task_root/native-circle-integration-v2/combined/cpu" "$task_root/native-circle-integration-v2/combined/cuda" "$task_root/cooperative-fallback-regression-v2/combined/cpu" "$task_root/cooperative-fallback-regression-v1/targeted/cuda"; do
    cmp "$task_tests/source.sha256" "$task_tests/source-after.sha256"
    grep -q '52 passed' "$task_tests/tests.log"
done
task_baseline="$task_root/verified-circle-scalar-b32-profile-v1"
cmp "$task_baseline/source.sha256" "$task_baseline/source-after.sha256"
test -f "$task_baseline/profile.json"
task_conditions="$task_root/original-point-scalar-b32-profile-v1/conditions.npz"
task_run="$task_root/verified-circle-matched-b32-v1"
test ! -e "$task_run"
mkdir -p "$task_run/source"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/{rigid_stress,morphs}.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/speed_benchmark/rigid_stress.py examples/rigid/assets/hollow_egg/level1/elastic.npz research/rigid_stress/{wrench,mechanics,field_cpu}.py "$task_root/traction_original_point_candidate_v10.py" "$task_root/traction_scalar_fallback_candidate_v1.py" "$task_root/traction_cooperative_fallback_candidate_v1.py" "$task_root/traction_sample_cooperative_candidate_v2.py" "$task_root/probe_cooperative_fallback_candidate_v1.py" "$task_root/run_verified_circle_matched_scopes_v1.sh" "$task_conditions")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/hardware.csv"
task_failed=0
for task_variant in scalar cooperative-fallback; do
    for task_scope in profile rigid recovery live policy; do
        task_variant_run="$task_run/$task_variant/$task_scope"
        mkdir -p "$task_variant_run/tmp"
        export TMPDIR="$task_variant_run/tmp" PYTHONUNBUFFERED=1
        export QD_OFFLINE_CACHE_FILE_PATH="$task_run/$task_variant/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/$task_variant/cache/genesis"
        task_command=("$RIGID_STRESS_ENV/bin/python")
        if [[ "$task_variant" == scalar ]]; then
            task_command+=(-m examples.speed_benchmark.rigid_stress)
        else
            task_command+=("$task_root/probe_cooperative_fallback_candidate_v1.py")
        fi
        task_steps=1280
        if [[ "$task_scope" == profile ]]; then
            export QD_KERNEL_PROFILER=1
            task_steps=20
        else
            unset QD_KERNEL_PROFILER
        fi
        "${task_command[@]}" --scope "$task_scope" --envs 32 --warmup 600 --steps "$task_steps" --repetitions 3 --minimum-seconds 10 --seed 623101 --varied --contact-model intrinsic --intrinsic-support pair_union --contact-block-size 4 --egg-geometry mesh64 --collision-method gjk --contact-patch on --conditions "$task_conditions" --compact-log --source-revision "$(git rev-parse HEAD)+$task_variant" --output "$task_variant_run/result.json" > "$task_variant_run/benchmark.log" 2>&1 || task_failed=1
    done
done
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
