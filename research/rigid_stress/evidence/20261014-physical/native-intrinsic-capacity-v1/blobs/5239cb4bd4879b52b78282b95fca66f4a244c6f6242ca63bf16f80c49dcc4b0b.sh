#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${1:?Physical runtime root required}
task_index=${2:-${SLURM_ARRAY_TASK_ID:?Refinement index required}}
task_geometries=(ellipsoid u)
test "$task_index" -ge 0 && test "$task_index" -lt 6
task_geometry=${task_geometries[task_index / 3]}
task_level=$((task_index % 3))
task_run="$task_root/fixed-poly-ref-$task_geometry-l$task_level-v1"
task_launcher="$task_run-launcher"
mkdir -p "$task_launcher/source"
sha256sum research/rigid_stress/scripts/run_fixed_poly_reference.sh > "$task_launcher/source.sha256"
cp --parents research/rigid_stress/scripts/run_fixed_poly_reference.sh "$task_launcher/source"
task_status=0
bash research/rigid_stress/scripts/run_load_models.sh "$task_run" \
    --mesh "$task_root/fixed-poly-$task_geometry-v1/meshes/level$task_level.npz" \
    --geometry "$task_geometry" --reference-only --reference-solver amg-cg || task_status=$?
if test "$task_status" -eq 0; then
    bash research/rigid_stress/scripts/sample_load_fields.sh "$task_run-probes" \
        --run "$task_run/results" --geometry "$task_geometry" \
        --mesh "$task_root/fixed-poly-$task_geometry-v1/meshes/level$task_level.npz" || task_status=$?
fi
sha256sum research/rigid_stress/scripts/run_fixed_poly_reference.sh > "$task_launcher/source-after.sha256"
cmp "$task_launcher/source.sha256" "$task_launcher/source-after.sha256"
printf '%s\n' "$task_status" > "$task_launcher/returncode.txt"
exit "$task_status"
