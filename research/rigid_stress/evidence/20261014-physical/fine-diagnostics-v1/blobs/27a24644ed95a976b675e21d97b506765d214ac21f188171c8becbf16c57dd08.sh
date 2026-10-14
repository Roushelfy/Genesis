#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_geometry=${1:?Geometry required}
task_version=${2:-v1}
case "$task_geometry" in ellipsoid|capsule|u|thin_neck|cavity|narrow_gap|egg) ;; *) exit 2 ;; esac
task_root="$RIGID_STRESS_DATA_ROOT/runs"
task_epoch="$task_root/20261014-physical-validation"
task_tools="$task_epoch/fine-reference-publication-$task_geometry-$task_version"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{verify_fine_reference_fields,summarize_fine_references}.py
    research/rigid_stress/scripts/publish_fine_reference_checkpoint.sh
    research/rigid_stress/scripts/{publish_nonconvex_checkpoint,verify_nonconvex_publication}.py)
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
git rev-parse HEAD > "$task_tools/base-revision.txt"
hostname > "$task_tools/node.txt"
"$task_env/bin/python" -m research.rigid_stress.verify_fine_reference_fields \
    --runtime-root "$task_epoch" --geometry "$task_geometry" --output "$task_tools/summary.json"
task_runs=()
for task_level in 0 1 2; do
    task_run="$task_epoch/fine-ref-$task_geometry-l$task_level-v1"
    task_runs+=("$task_run" "$task_run-probes" "$task_run-launcher")
done
task_runs+=("$task_tools")
task_dest="research/rigid_stress/evidence/20261014-physical/fine-reference-$task_geometry-$task_version"
"$task_env/bin/python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root" --destination "$task_dest" --staging-root "$task_epoch/publication-staging" \
    --run "${task_runs[@]}" --summary "$task_tools/summary.json" \
    --limitations "Independent finite-traction FEM references on three finer CAD meshes. This combines geometry and volume changes; other error axes remain separate. Every full tensor, raw corner peak/location, work identity, original traction snapshot and source closure is retained. Complete residuals are computed by the independently assembled FEM driver. Mesh assets are already published in physical-axes-checkpoint-v1. Native finer-field, contact, profile/history and accuracy/cost acceptance remain required."
sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"
cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"
"$task_env/bin/python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
