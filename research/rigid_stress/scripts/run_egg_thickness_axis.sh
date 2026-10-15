#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_seed=${2:?Existing layered full egg mesh required}
task_layers=${3:?Number of thickness layers required}
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{refine_egg_thickness,refine_fixed_geometry}.py
    research/rigid_stress/scripts/{run_egg_thickness_axis,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; exit "$task_status"' EXIT
printf '%s\n' "$task_seed $task_layers" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
"$task_env/bin/python" -m research.rigid_stress.refine_egg_thickness \
    --seed "$task_seed" --layers "$task_layers" --output "$task_run/mesh.npz" \
    > "$task_run/mesh.log" 2>&1
bash research/rigid_stress/scripts/run_load_models.sh "$task_run/fem" \
    --geometry egg --mesh "$task_run/mesh.npz" --reference-only --reference-solver amg-cg
bash research/rigid_stress/scripts/sample_load_fields.sh "$task_run/probes" \
    --run "$task_run/fem/results" --mesh "$task_run/mesh.npz" --geometry egg
