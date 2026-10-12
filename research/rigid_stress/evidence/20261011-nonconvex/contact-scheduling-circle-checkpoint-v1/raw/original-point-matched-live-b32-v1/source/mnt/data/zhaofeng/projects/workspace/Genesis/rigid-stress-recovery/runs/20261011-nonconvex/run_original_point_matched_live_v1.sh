#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
for task_backend in cpu cuda; do
    task_tests="$task_root/original-point-cooperative-v10-v1/$task_backend"
    cmp "$task_tests/source.sha256" "$task_tests/source-after.sha256"
    grep -q '51 passed' "$task_tests/tests.log"
done
for task_envs in 16 32; do
    task_live="$task_root/original-point-cooperative-online-b${task_envs}-v2"
    test -f "$task_live/results/summary.json"
    cmp "$task_live/source.sha256" "$task_live/source-after.sha256"
done
task_conditions="$task_root/original-point-scalar-b32-profile-v1/conditions.npz"
task_run="$task_root/original-point-matched-live-b32-v1"
test ! -e "$task_run"
mkdir -p "$task_run/source"
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/{rigid_stress,morphs}.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/speed_benchmark/rigid_stress.py research/rigid_stress/{wrench,mechanics,field_cpu}.py "$task_root/traction_original_point_candidate_v10.py" "$task_root/traction_sample_cooperative_candidate_v2.py" "$task_root/probe_original_point_candidate_v10.py" "$task_root/run_original_point_matched_live_v1.sh" "$task_conditions")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$task_run/hardware.csv"
task_failed=0
for task_variant in scalar cooperative; do
    task_variant_run="$task_run/$task_variant"
    mkdir -p "$task_variant_run/tmp"
    export TMPDIR="$task_variant_run/tmp" PYTHONUNBUFFERED=1
    export QD_OFFLINE_CACHE_FILE_PATH="$task_variant_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_variant_run/cache/genesis"
    task_command=("$RIGID_STRESS_ENV/bin/python")
    if [[ "$task_variant" == scalar ]]; then
        task_command+=(-m examples.speed_benchmark.rigid_stress)
    else
        task_command+=("$task_root/probe_original_point_candidate_v10.py")
    fi
    "${task_command[@]}" --scope live --envs 32 --warmup 600 --steps 1280 --repetitions 3 --minimum-seconds 10 --seed 623101 --varied --contact-model intrinsic --intrinsic-support pair_union --contact-block-size 4 --egg-geometry mesh64 --collision-method gjk --contact-patch on --conditions "$task_conditions" --compact-log --source-revision "$(git rev-parse HEAD)+${task_variant}" --output "$task_variant_run/live.json" > "$task_variant_run/live.log" 2>&1 || task_failed=1
done
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
