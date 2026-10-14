#!/usr/bin/env bash
# Measure max/full/max on one allocated GPU and an unchanged condition bank.
set -euo pipefail

task_batch=${1:?Batch required}
task_seed=${2:?Seed required}
task_bank=${3:?Condition bank required}
task_root=${4:?New runtime directory required}
test -f "$task_bank"
test ! -e "$task_root"
source research/rigid_stress/scripts/cluster_env.sh
mkdir -p "$task_root"
task_bank=$(realpath "$task_bank")
task_root=$(realpath "$task_root")
task_revision=$(git rev-parse HEAD)
task_gpu=$("$RIGID_STRESS_ENV/bin/python" -c 'import torch; print("GPU-" + str(torch.cuda.get_device_properties(0).uuid))')
mapfile -t task_sources < <(git ls-files 'genesis/*.py' 'genesis/**/*.py' 'examples/rigid/*.py' 'examples/speed_benchmark/*.py')
task_sources+=(examples/rigid/assets/hollow_egg/level1/elastic.npz research/rigid_stress/scripts/measure_nonconvex_output_modes.sh)

for task_scope in live policy; do
    for task_route in max-before full max-after; do
        task_mode=max
        if [[ "$task_route" == full ]]; then task_mode=full; fi
        task_run="$task_root/$task_scope/$task_route"
        mkdir -p "$task_run/tmp" "$task_run/source"
        export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
        export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
        export GS_CACHE_FILE_PATH="$task_run/cache/genesis"
        sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
        cp --parents "${task_sources[@]}" "$task_run/source"
        git diff HEAD > "$task_run/source.patch"
        printf '%s\n' "$task_revision" > "$task_run/base-revision.txt"
        hostname > "$task_run/node.txt"
        nvidia-smi --id="$task_gpu" --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$task_run/hardware.csv"
        nvidia-smi --id="$task_gpu" --query-gpu=timestamp,uuid,memory.used,power.draw,clocks.sm,utilization.gpu --format=csv -lms 200 > "$task_run/device-telemetry.csv" 2>&1 &
        task_monitor=$!
        trap 'kill "$task_monitor" 2>/dev/null || true; wait "$task_monitor" 2>/dev/null || true' EXIT
        sha256sum "$task_bank" > "$task_run/conditions-before.sha256"
        cp "$task_bank" "$task_run/conditions.npz"
        task_args=(--scope "$task_scope" --envs "$task_batch" --seed "$task_seed"
            --conditions "$task_bank" --warmup 600 --steps 1280 --repetitions 3
            --minimum-seconds 10 --varied --contact-model intrinsic
            --intrinsic-load-partition body_source_prior --contact-block-size 4
            --egg-geometry mesh64 --collision-method gjk --contact-patch on
            --trajectory-metrics --compact-log --output-mode "$task_mode"
            --source-revision "$task_revision+nonconvex-output-modes"
            --output "$task_run/result.json")
        # The intrusive trace runs after throughput timing and stays separate.
        if [[ "$task_scope" == live && "$task_route" != max-after ]]; then task_args+=(--trace); fi
        printf '%q ' "$RIGID_STRESS_ENV/bin/python" -m examples.speed_benchmark.rigid_stress "${task_args[@]}" > "$task_run/command.txt"
        printf '\n' >> "$task_run/command.txt"
        task_status=0
        "$RIGID_STRESS_ENV/bin/python" -m examples.speed_benchmark.rigid_stress "${task_args[@]}" > "$task_run/benchmark.log" 2>&1 || task_status=$?
        printf '%s\n' "$task_status" > "$task_run/returncode.txt"
        sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
        cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
        sha256sum "$task_bank" > "$task_run/conditions-after.sha256"
        cmp "$task_run/conditions-before.sha256" "$task_run/conditions-after.sha256"
        kill "$task_monitor" 2>/dev/null || true
        wait "$task_monitor" 2>/dev/null || true
        trap - EXIT
        if [[ "$task_status" != 0 ]]; then exit "$task_status"; fi
    done
done
