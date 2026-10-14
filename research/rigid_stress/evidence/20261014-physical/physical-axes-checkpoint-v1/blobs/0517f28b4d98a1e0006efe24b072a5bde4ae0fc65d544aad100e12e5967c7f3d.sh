#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${1:?Physical runtime root required}
task_index=${2:-${SLURM_ARRAY_TASK_ID:?Radius index required}}
task_geometries=(u capsule)
task_radii=(0.001 0.002 0.004)
test "$task_index" -ge 0 && test "$task_index" -lt 6
task_geometry=${task_geometries[task_index / 3]}
task_radius=${task_radii[task_index % 3]}
task_size=0.00175
if test "$task_geometry" = capsule; then task_size=0.0025; fi
task_run="$task_root/radius-$task_geometry-r$task_radius-v1"
task_launch="$task_run-launcher"
mkdir -p "$task_launch/source"
sha256sum research/rigid_stress/scripts/run_radius_sensitivity.sh > "$task_launch/source.sha256"
cp --parents research/rigid_stress/scripts/run_radius_sensitivity.sh "$task_launch/source"
task_status=0
bash research/rigid_stress/scripts/run_load_models.sh "$task_run" \
    --mesh "$task_root/matrix-mesh-$task_geometry-v1/meshes/h$task_size.npz" \
    --geometry "$task_geometry" --radius "$task_radius" --backend cuda --method pcg || task_status=$?
sha256sum research/rigid_stress/scripts/run_radius_sensitivity.sh > "$task_launch/source-after.sha256"
cmp "$task_launch/source.sha256" "$task_launch/source-after.sha256"
printf '%s\n' "$task_status" > "$task_launch/returncode.txt"
exit "$task_status"
