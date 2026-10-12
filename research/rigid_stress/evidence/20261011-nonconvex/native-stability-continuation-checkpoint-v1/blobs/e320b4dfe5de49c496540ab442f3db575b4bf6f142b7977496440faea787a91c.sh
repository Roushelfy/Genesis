#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-refinement
source research/rigid_stress/scripts/cluster_env.sh
task_batch=${1:?Batch required}
task_mode=${2:?Mode required: copied or varied}
task_scope=${3:?Scope required}
task_variant=${4:-frontier}
task_seed=${5:-623101}
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/native-canonical-feature-v1/b${task_batch}-${task_mode}-${task_scope}-${task_variant}-${task_seed}"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_conditions="$task_root/original-point-scalar-b32-profile-v1/conditions.npz"
task_extra=()
if [[ "$task_mode" == copied ]]; then task_extra+=(--conditions "$task_conditions"); fi
if [[ "$task_variant" == dense ]]; then task_extra+=(--dense-surface-fan); fi
task_steps=1280
if [[ "$task_scope" == profile ]]; then task_steps=128; export QD_KERNEL_PROFILER=1; fi
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/options/{rigid_stress,morphs}.py examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/speed_benchmark/rigid_stress.py examples/rigid/assets/hollow_egg/level1/elastic.npz "$task_conditions" "$task_root/run_native_canonical_feature_v1.sh")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
git diff HEAD > "$task_run/source.patch"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$task_run/hardware.csv"
task_failed=0
"$RIGID_STRESS_ENV/bin/python" examples/speed_benchmark/rigid_stress.py "${task_extra[@]}" --scope "$task_scope" --envs "$task_batch" --warmup 600 --steps "$task_steps" --repetitions 3 --minimum-seconds 10 --seed "$task_seed" --varied --contact-model intrinsic --intrinsic-load-partition body_source_prior --contact-block-size 4 --egg-geometry mesh64 --collision-method gjk --contact-patch on --compact-log --source-revision "$(git rev-parse HEAD)+canonical-shared-feature-v1" --output "$task_run/result.json" > "$task_run/benchmark.log" 2>&1 || task_failed=1
echo "$task_failed" > "$task_run/returncode.txt"
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
