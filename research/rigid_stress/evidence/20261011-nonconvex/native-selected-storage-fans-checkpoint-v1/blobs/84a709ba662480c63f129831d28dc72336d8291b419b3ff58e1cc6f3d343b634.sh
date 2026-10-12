#!/usr/bin/env bash
set -euo pipefail
task_mode=${1:?Mode required}
task_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex
if [[ "$task_mode" == baseline ]]; then
    cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-profile-growth
else
    test "$task_mode" == candidate
    cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
fi
source research/rigid_stress/scripts/cluster_env.sh
task_run="$task_root/native-compact-cache-fan-selected-live-v1/b2048-${task_mode}-623102"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_conditions=(--save-conditions "$task_run/conditions.npz")
if [[ "$task_mode" == candidate ]]; then
    task_conditions+=(--conditions "$task_root/native-compact-cache-fan-selected-live-v1/b2048-baseline-623102/conditions.npz")
fi
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/engine/solvers/rigid/collider/*.py genesis/options/{rigid_stress,morphs}.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/speed_benchmark/rigid_stress{,_metrics}.py examples/rigid/assets/hollow_egg/level1/elastic.npz "$task_root/run_compact_cache_fan_selected_live_v1.sh")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git diff HEAD > "$task_run/source.patch"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$task_run/hardware.csv"
task_gpu=$("$RIGID_STRESS_ENV/bin/python" -c 'import torch; print("GPU-" + str(torch.cuda.get_device_properties(0).uuid))')
nvidia-smi --id="$task_gpu" --query-gpu=timestamp,uuid,memory.used,power.draw,clocks.sm,utilization.gpu --format=csv -lms 200 > "$task_run/device-telemetry.csv" 2>&1 &
task_monitor=$!
trap 'kill "$task_monitor" 2>/dev/null || true; wait "$task_monitor" 2>/dev/null || true' EXIT
task_failed=0
"$RIGID_STRESS_ENV/bin/python" -m examples.speed_benchmark.rigid_stress --scope live --envs 2048 --warmup 600 --steps 1280 --repetitions 3 --minimum-seconds 0 --seed 623102 --varied --contact-model intrinsic --intrinsic-load-partition body_source_prior --contact-block-size 4 --egg-geometry mesh64 --collision-method gjk --contact-patch on --trajectory-metrics --compact-log "${task_conditions[@]}" --source-revision "$(git rev-parse HEAD)+compact-cache-fan-selected-v1-$task_mode" --output "$task_run/result.json" > "$task_run/benchmark.log" 2>&1 || task_failed=1
echo "$task_failed" > "$task_run/returncode.txt"
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
