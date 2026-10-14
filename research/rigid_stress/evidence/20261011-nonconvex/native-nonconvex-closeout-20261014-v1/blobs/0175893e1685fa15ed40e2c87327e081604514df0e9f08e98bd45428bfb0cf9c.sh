#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_batch=${1:?Batch required}
task_bank="$task_root/native-selected-matched-scopes-v1/b2048-623102/conditions.npz"
test -f "$task_bank"
task_gpu=$("$RIGID_STRESS_ENV/bin/python" -c 'import torch; print("GPU-" + str(torch.cuda.get_device_properties(0).uuid))')
for task_scope in live policy; do
    task_run="$task_root/native-final-copied-edge-v1/b${task_batch}-623102/$task_scope"
    test ! -e "$task_run"
    mkdir -p "$task_run/tmp" "$task_run/source"
    export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
    export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
    task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/engine/solvers/rigid/collider/*.py genesis/options/{rigid_stress,morphs}.py genesis/utils/{array_class,misc}.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/speed_benchmark/rigid_stress{,_metrics}.py examples/rigid/assets/hollow_egg/level1/elastic.npz "$task_root/run_native_final_copied_edge_v1.sh")
    sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
    cp --parents "${task_sources[@]}" "$task_run/source"
    git diff HEAD > "$task_run/source.patch"
    git rev-parse HEAD > "$task_run/base-revision.txt"
    hostname > "$task_run/node.txt"
    nvidia-smi --id="$task_gpu" --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$task_run/hardware.csv"
    nvidia-smi --id="$task_gpu" --query-gpu=timestamp,uuid,memory.used,power.draw,clocks.sm,utilization.gpu --format=csv -lms 200 > "$task_run/device-telemetry.csv" 2>&1 &
    task_monitor=$!
    trap 'kill "$task_monitor" 2>/dev/null || true; wait "$task_monitor" 2>/dev/null || true' EXIT
    sha256sum "$task_bank" > "$task_run/conditions-before.sha256"
    task_failed=0
    "$RIGID_STRESS_ENV/bin/python" -m examples.speed_benchmark.rigid_stress --scope "$task_scope" --envs "$task_batch" --warmup 600 --steps 1280 --repetitions 3 --minimum-seconds 0 --seed 623102 --varied --conditions "$task_bank" --contact-model intrinsic --intrinsic-load-partition body_source_prior --contact-block-size 4 --egg-geometry mesh64 --collision-method gjk --contact-patch on --trajectory-metrics --compact-log --source-revision "$(git rev-parse HEAD)+final-copied-edge-v1" --output "$task_run/result.json" > "$task_run/benchmark.log" 2>&1 || task_failed=1
    echo "$task_failed" > "$task_run/returncode.txt"
    sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
    cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
    sha256sum "$task_bank" > "$task_run/conditions-after.sha256"
    cmp "$task_run/conditions-before.sha256" "$task_run/conditions-after.sha256"
    cp "$task_bank" "$task_run/conditions.npz"
    kill "$task_monitor" 2>/dev/null || true
    wait "$task_monitor" 2>/dev/null || true
    trap - EXIT
    if [[ "$task_failed" != 0 ]]; then exit "$task_failed"; fi
done
