#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${1:?Physical runtime root required}
task_index=${2:-${SLURM_ARRAY_TASK_ID:?Fine reference index required}}
task_geometries=(ellipsoid capsule u thin_neck cavity narrow_gap egg)
test "$task_index" -ge 0 && test "$task_index" -lt 21
task_geometry=${task_geometries[task_index / 3]}
task_level=$((task_index % 3))
case "$task_geometry" in
    ellipsoid|capsule|thin_neck) task_sizes=(0.00175 0.00125 0.000875) ;;
    u|narrow_gap) task_sizes=(0.00125 0.000875 0.000625) ;;
    cavity) task_sizes=(0.11 0.08 0.06) ;;
    egg) task_sizes=(0.06 0.04 0.03) ;;
esac
task_mesh="$task_root/fine-matrix-mesh-$task_geometry-v1/meshes/h${task_sizes[task_level]}.npz"
task_run="$task_root/fine-ref-$task_geometry-l$task_level-v1"
task_launch="$task_run-launcher"
mkdir -p "$task_launch/source"
sha256sum research/rigid_stress/scripts/run_fine_reference_matrix.sh > "$task_launch/source.sha256"
cp --parents research/rigid_stress/scripts/run_fine_reference_matrix.sh "$task_launch/source"
task_status=0
bash research/rigid_stress/scripts/run_load_models.sh "$task_run" \
    --geometry "$task_geometry" --mesh "$task_mesh" --reference-only --reference-solver amg-cg || task_status=$?
if test "$task_status" -eq 0; then
    bash research/rigid_stress/scripts/sample_load_fields.sh "$task_run-probes" \
        --run "$task_run/results" --mesh "$task_mesh" --geometry "$task_geometry" || task_status=$?
fi
sha256sum research/rigid_stress/scripts/run_fine_reference_matrix.sh > "$task_launch/source-after.sha256"
cmp "$task_launch/source.sha256" "$task_launch/source-after.sha256"
printf '%s\n' "$task_status" > "$task_launch/returncode.txt"
exit "$task_status"
