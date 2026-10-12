#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-profile-growth
source research/rigid_stress/scripts/cluster_env.sh
task_backend=${1:?Backend required}
task_root="$RIGID_STRESS_DATA_ROOT/runs/20261011-nonconvex"
task_run="$task_root/native-compact-profile-growth-regression-v1/$task_backend"
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1 PYTHONPATH="$task_root:$PYTHONPATH"
export QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants" GS_CACHE_FILE_PATH="$task_run/cache/genesis"
task_test_sources=(tests/rigid/test_stress_{source_priors,sources,nonconvex,footprint}.py)
task_tests=(tests/rigid/test_stress_source_priors.py tests/rigid/test_stress_sources.py tests/rigid/test_stress_nonconvex.py::test_native_surface_paths_do_not_cross_gaps_or_holes tests/rigid/test_stress_footprint.py)
task_sources=(genesis/engine/solvers/rigid/stress/*.py genesis/engine/solvers/rigid/collider/*.py genesis/options/{rigid_stress,morphs}.py "${task_test_sources[@]}" tests/rigid/assets/nonconvex_*.npz examples/rigid/assets/hollow_egg/level1/elastic.npz examples/rigid/{franka_egg_stress,egg_stress_controller}.py examples/speed_benchmark/rigid_stress{,_metrics}.py research/rigid_stress/{wrench,mechanics,field_cpu}.py "$task_root/run_compact_profile_growth_regression_v1.sh")
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
hostname > "$task_run/node.txt"
git rev-parse HEAD > "$task_run/base-revision.txt"
git diff > "$task_run/source.patch"
if [[ "$task_backend" == cuda ]]; then nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv > "$task_run/hardware.csv"; fi
task_failed=0
"$RIGID_STRESS_ENV/bin/python" -m pytest -n 0 --backend "$task_backend" --tb=short -q --junitxml "$task_run/report.xml" -o junit_family=legacy "${task_tests[@]}" > "$task_run/tests.log" 2>&1 || task_failed=1
echo "$task_failed" > "$task_run/returncode.txt"
sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
exit "$task_failed"
