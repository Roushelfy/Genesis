#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_root="$RIGID_STRESS_DATA_ROOT/runs"
task_epoch="$task_root/20261014-physical-validation"
task_attempt=${1:-v1}
task_tools="$task_epoch/load-publication-tools-$task_attempt"
test ! -e "$task_tools"
mkdir -p "$task_tools/source" "$task_tools/tmp"
export TMPDIR="$task_tools/tmp" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{audit_load_checkpoint,load_model_validation,sample_load_fields,finite_traction_reference,native_load_samples,skfem_reference,contact_reference,bonded_block_reference,compare_contact_controls,check_probe_reconstruction,plot_load_sheet_control,summarize_load_refinement}.py
    research/rigid_stress/scripts/{publish_nonconvex_checkpoint,verify_nonconvex_publication}.py
    research/rigid_stress/scripts/publish_load_model_checkpoint.sh)
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
git rev-parse HEAD > "$task_tools/base-revision.txt"
git diff HEAD > "$task_tools/source.patch"
task_names=(mesh-smoke-v2 pressure-sheet-before-v3 pressure-sheet-cpu-after-v1 pressure-sheet-cuda-after-v1
    pressure-module-cpu-v1 pressure-module-cuda-v1 independent-transfer-cpu-v4
    independent-transfer-cpu-v5 independent-transfer-cuda-v5
    load-u-after-v1 probes-u-after-v1 load-sheet-figure-v1 load-sheet-figure-v2
    load-matrix-capsule-l0-v2 load-matrix-capsule-l0-v2-probes
    contact-friction-smoke-v1 contact-friction-damped-v2 contact-friction-small-step-v3
    contact-friction-controls-v4 bonded-controls-v5 friction-comparison-v5 probe-reconstruction-v1
    shell-amg-h11-v1 shell-amg-h08-v1 shell-symmetric-h08-v1
    shell-amg-h11-v2 shell-amg-h08-v2 shell-symmetric-h08-v2 shell-amg-h06-v1
    load-refinement-summary-v3 egg-pcg-diagnostic-v2 egg-pcg-diagnostic-v3)
for task_geometry in ellipsoid capsule u thin_neck cavity narrow_gap egg; do
    task_names+=("matrix-mesh-$task_geometry-v1" "load-matrix-$task_geometry-l0-v1")
done
for task_geometry in ellipsoid u; do
    for task_smoke_version in 1 2 3 4 5; do task_names+=("load-$task_geometry-smoke-v$task_smoke_version"); done
done
for task_geometry in ellipsoid capsule u thin_neck cavity narrow_gap; do
    for task_level in 0 1 2; do
        task_names+=("load-matrix-$task_geometry-l$task_level-v3" "load-matrix-$task_geometry-l$task_level-v3-probes")
    done
done
for task_resolution in 3 4 5; do
    for task_level in 0 1; do task_names+=("contact-friction-tail-n$task_resolution-l$task_level-v1"); done
done
for task_name in "${task_names[@]}"; do test -f "$task_epoch/$task_name/source-after.sha256"; done
"$task_env/bin/python" -m research.rigid_stress.audit_load_checkpoint \
    --runtime-root "$task_epoch" --run-names "${task_names[@]}" --output "$task_tools/audit"
task_jobs=(712150 712151 712165 712166 712188 712189 712197 712198 712219 712220
    712222 712230 712233 712236 712238 712239 712240 712241 712244 712245 712246 712247
    712249 712170 712171 712172 712173 712174 712175 712176 712100 712101 712102
    712284 712290 712294 712317 712324 712325 712341 712342 712343 712347
    712348 712384 712420 712429 712435 712448 712451)
task_job_ids=$(IFS=,; printf '%s' "${task_jobs[*]}")
sacct -X -j "$task_job_ids" -o JobIDRaw,State,ExitCode,Elapsed -n -P > "$task_tools/jobs.txt"
sacct -X -j "$task_job_ids" -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
sacct -j 712347,712435 -o JobIDRaw,State,MaxRSS,Elapsed -n -P > "$task_tools/memory.txt"
task_runs=()
for task_name in "${task_names[@]}"; do task_runs+=("$task_epoch/$task_name"); done
task_runs+=("$task_tools")
task_extra=()
for task_job in "${task_jobs[@]}"; do
    for task_path in "$task_epoch"/*"$task_job"_*.log "$task_epoch"/*"$task_job"_*.err; do
        if test -f "$task_path" && ! [[ "$task_job" = 712347 ]]; then task_extra+=("$task_path"); fi
    done
done
task_dest="research/rigid_stress/evidence/20261014-physical/load-model-checkpoint-$task_attempt"
"$task_env/bin/python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root" --destination "$task_dest" --staging-root "$task_epoch/publication-staging" \
    --run "${task_runs[@]}" --extra "${task_extra[@]}" --summary "$task_tools/audit/summary.json" \
    --limitations "Physical-validation progress. Independent complete equilibrium, tensor, energy and raw peak are recomputed for343 native fields:342 on six shapes and three useful mesh levels, plus one full-egg pressure solve;18 pressure free-moment inputs remain unsupported. All six-shape physical refinement tails remain above the initial combined3% diagnostic. Pressure violates local compression/friction in the narrow gap and full egg. The nearest material-sheet defect is fixed with matched frozen inputs and33CPU/54CUDA module passes plus8CPU/8CUDA independent boundary-transfer passes. Synthetic contact snapshots are not a Franka/deformable-grasp validation. Initial failures, near-cancellation rejection, invalid observer assertions, unqualified displacement gauge comparisons and all source snapshots remain visible. Supported IPC sticking controls and independent bonded controls are included, with regularization and spatial error retained. Default2000-iteration egg recovery and the10000-iteration follow-up reject the elastic solve; a50000 cap converges in14583 iterations without changing the residual budget. The direct follow-up is tracked separately. The h0.06 analytic shell passes the combined tail diagnostic but geometry/volume error separation remains incomplete. Finite footprints do not establish complete physical convergence, separated error axes or full egg/interior-hotspot contact acceptance."
sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"
cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"
"$task_env/bin/python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
