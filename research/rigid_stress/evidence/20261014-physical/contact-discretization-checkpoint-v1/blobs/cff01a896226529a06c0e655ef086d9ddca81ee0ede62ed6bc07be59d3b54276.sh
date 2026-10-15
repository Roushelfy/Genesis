#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_mode=${2:?dynamic or grasp}
shift 2
test ! -e "$task_run"
mkdir -p "$task_run/input-source" "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export REFERENCE_LINEAR_SOLVER_STATS_PATH="$task_run/linear_solver_stats.json"
task_sources=(research/rigid_stress/{cached_reference_newton,dynamic_franka_reference,grasp_contact_reference,finite_traction_reference,refine_franka_reference,franka_contact_reference,franka_rigid_mode_coordinates,audit_franka_reference,contact_reference,neohookean_reference,stable_neohookean_reference,skfem_reference,summarize_fine_references,sparse_franka_inertia}.py
    research/rigid_stress/{cached_reference_newton,sparse_franka_driver}.patch research/rigid_stress/scripts/{run_cached_reference,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/input-source.sha256"
cp --parents "${task_sources[@]}" "$task_run/input-source"
cp -a "$task_run/input-source/." "$task_run/source/"
patch -p1 -d "$task_run/source" < research/rigid_stress/cached_reference_newton.patch > "$task_run/patch.log"
case "$task_mode" in
    dynamic) task_module=dynamic_franka_reference
        patch -p1 -d "$task_run/source" < research/rigid_stress/sparse_franka_driver.patch >> "$task_run/patch.log" ;;
    grasp) task_module=grasp_contact_reference ;;
    *) exit 2 ;;
esac
touch "$task_run/source/research/__init__.py" "$task_run/source/research/rigid_stress/__init__.py"
task_actual=("${task_sources[@]}" research/__init__.py research/rigid_stress/__init__.py)
(cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source.sha256"
trap 'task_status=$?; (cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source-after.sha256"; sha256sum "${task_sources[@]}" > "$task_run/input-source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; cmp "$task_run/input-source.sha256" "$task_run/input-source-after.sha256"; exit "$task_status"' EXIT
printf '%s\n' "$task_mode $*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
lscpu > "$task_run/cpu.txt"
(cd "$task_run/source"
    export PYTHONPATH="$task_run/source"
    "$task_env/bin/python" -c 'from pathlib import Path; from research.rigid_stress import contact_reference as m; assert Path(m.__file__).resolve().is_relative_to(Path.cwd()); print(m.__file__)' > "$task_run/driver-origin.txt"
    "$task_env/bin/python" -m "research.rigid_stress.$task_module" --output "$task_run/results" "$@" \
        > "$task_run/reference.log" 2>&1)
