#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-path-reuse
source research/rigid_stress/scripts/cluster_env.sh
task_mode=${1:?Port bounds mode required}
task_scope=${2:?Scope required}
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_probe="$task_root/probe_compact_traction_profiles_v1.py"
task_probe_mode="$task_mode"
if [[ "$task_mode" == whole ]]; then
    task_probe="$task_root/probe_native_contact_graph_v2.py"
    task_probe_mode=on
elif [[ "$task_mode" == whole-fast ]]; then
    task_probe="$task_root/probe_native_contact_graph_fast_v1.py"
    task_probe_mode=on
fi
task_run="$task_root/native-compact-traction-profiles-v1/b16-varied-${task_scope}-${task_mode}-623102"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_steps=1280
if [[ "$task_scope" == profile ]]; then task_steps=128; export QD_KERNEL_PROFILER=1; fi
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/engine/solvers/rigid/collider/*.py genesis/options/{rigid_stress,morphs}.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/speed_benchmark/rigid_stress{,_metrics}.py examples/rigid/assets/hollow_egg/level1/elastic.npz "$task_root/run_compact_traction_profiles_v1.sh" "$task_probe")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git diff HEAD > "$task_run/source.patch"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$task_run/hardware.csv"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" "$task_probe" "$task_probe_mode" --scope "$task_scope" --envs 16 --warmup 600 --steps "$task_steps" --repetitions 1 --minimum-seconds 0 --seed 623102 --varied --contact-model intrinsic --intrinsic-load-partition body_source_prior --contact-block-size 4 --egg-geometry mesh64 --collision-method gjk --contact-patch on --trajectory-metrics --compact-log --source-revision "$(git rev-parse HEAD)+native-compact-traction-profiles-v1-$task_mode" --output "$task_run/result.json" > "$task_run/benchmark.log" 2>&1 || task_failed=1
echo "$task_failed" > "$task_run/returncode.txt"
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
