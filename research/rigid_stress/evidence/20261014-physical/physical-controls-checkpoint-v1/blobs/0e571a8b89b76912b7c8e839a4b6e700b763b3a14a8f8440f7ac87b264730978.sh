#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_root="$RIGID_STRESS_DATA_ROOT/runs"
task_epoch="$task_root/20261014-physical-validation"
task_tools="$task_epoch/controls-publication-tools-v1"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{skfem_reference,validation_meshes,spherical_shell_validation,contact_reference,diagnose_contact_reference,plot_physical_controls}.py
    tests/rigid/test_stress_independent_fem.py
    research/rigid_stress/scripts/{summarize_physical_controls,summarize_physical_initial,verify_nonconvex_publication,publish_nonconvex_checkpoint}.py
    research/rigid_stress/scripts/{check_physical_independent,run_shell_validation,run_contact_reference,run_contact_convergence,diagnose_contact_reference,plot_physical_controls,setup_physical_amg,publish_physical_controls}.sh)
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
git rev-parse HEAD > "$task_tools/base-revision.txt"
git diff HEAD > "$task_tools/source.patch"
task_jobs=(711916 711933 711956 711962 711971 711980 711981 711985 711986
    711988 711989 711990 711993 711997 712024 712030 712032 712034 712038 712043 712051)
task_job_ids=$(IFS=,; printf '%s' "${task_jobs[*]}")
sacct -X -j "$task_job_ids" -o JobIDRaw,State,ExitCode,Elapsed -n -P > "$task_tools/jobs.txt"
sacct -X -j "$task_job_ids" -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
sacct -j 711830,711993 -o JobIDRaw,State,MaxRSS,Elapsed -n -P > "$task_tools/shell-memory.txt"
task_names=(independent-probes-cpu-v2 independent-gauges-cpu-v3 independent-gauges-cuda-v3
    spherical-shell-v2 spherical-shell-pinned-v3 contact-blocks-v3
    contact-blocks-convergence-v1 contact-blocks-convergence-v2 contact-blocks-convergence-v3
    contact-blocks-convergence-v4 contact-blocks-convergence-v5 contact-blocks-diagnostic-v1
    contact-blocks-n5-d1p25-v4 contact-blocks-n5-d1p25-v5 contact-blocks-n5-d1p25-v6
    contact-n5-diagnostic-v3 contact-n5-diagnostic-v4 setup-amg-v1
    control-figures-v1 control-figures-v2 control-figures-v3)
"$task_env/bin/python" research/rigid_stress/scripts/summarize_physical_controls.py \
    --runtime-root "$task_epoch" --run-names "${task_names[@]}" --output "$task_tools/summary.json"
task_runs=()
for task_name in "${task_names[@]}"; do task_runs+=("$task_epoch/$task_name"); done
task_runs+=("$task_tools")
task_extra=()
for task_job in "${task_jobs[@]}" 711830; do
    for task_path in "$task_epoch"/*"$task_job"_0.log "$task_epoch"/*"$task_job"_0.err; do
        if [[ -f "$task_path" ]]; then task_extra+=("$task_path"); fi
    done
done
task_dest=research/rigid_stress/evidence/20261014-physical/physical-controls-checkpoint-v1
"$task_env/bin/python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root" --destination "$task_dest" \
    --staging-root "$task_epoch/publication-staging" --run "${task_runs[@]}" \
    --extra "${task_extra[@]}" --summary "$task_tools/summary.json" \
    --limitations "Independent pinned/mass rigid gauges and off-centroid probes pass8CPU/8CUDA cases. The five-level pinned shell includes a successful h0.08 field, while the previous dense-gauge memory failure remains preserved. Contact normal controls retain all12-case scans and failures; the new roundoff-root variants regress and are not selected. The best retained/current contact scan has11accepted/12, with its n5/d1.25micrometre case unresolved. Figures retain raw maxima, common colour limits, failed markers and exact data/source provenance. Solver-scaling trials are excluded. Full shape/load-model/friction/egg/interior-hotspot physical acceptance remains unfinished."
sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"
cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"
"$task_env/bin/python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals \
    --output "$task_epoch/controls-publication-verification/result.json"
