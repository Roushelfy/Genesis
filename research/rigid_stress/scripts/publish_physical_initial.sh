#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_root="$RIGID_STRESS_DATA_ROOT/runs"
task_epoch="$task_root/20261014-physical-validation"
task_tools="$task_epoch/initial-publication-tools-v1"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{skfem_reference,validation_meshes,spherical_shell_validation}.py
    tests/rigid/test_stress_independent_fem.py
    research/rigid_stress/scripts/{summarize_physical_initial,verify_nonconvex_publication,publish_nonconvex_checkpoint}.py
    research/rigid_stress/scripts/{check_physical_independent,inspect_physical_env,setup_physical_env,generate_physical_meshes,run_shell_validation,publish_physical_initial}.sh)
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
git rev-parse HEAD > "$task_tools/base-revision.txt"
git diff HEAD > "$task_tools/source.patch"
sacct -X -j 711773,711776,711782,711785,711786,711794,711811,711824,711850,711890,711895,711905 \
    -o JobIDRaw,State,ExitCode,Elapsed -n -P > "$task_tools/jobs.txt"
"$task_env/bin/python" research/rigid_stress/scripts/summarize_physical_initial.py \
    --runtime-root "$task_epoch" --output "$task_tools/summary.json"
task_runs=()
for task_name in environment-v1 setup-v1 setup-v2 independent-cpu-v1 independent-cuda-v1 \
    mesh-smoke-v1 mesh-smoke-v2 spherical-shell-v1 contact-setup-v1 contact-setup-v2 \
    contact-blocks-v1 contact-blocks-v2; do
    task_runs+=("$task_epoch/$task_name")
done
task_runs+=("$task_tools")
task_extra=()
for task_job in 711773 711776 711782 711785 711786 711794 711811 711824 711850 711890 711895 711905; do
    for task_path in "$task_epoch"/*"$task_job"_0.log "$task_epoch"/*"$task_job"_0.err; do
        if [[ -f "$task_path" ]]; then task_extra+=("$task_path"); fi
    done
done
task_dest=research/rigid_stress/evidence/20261014-physical/independent-fem-initial-checkpoint-v1
"$task_env/bin/python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root" --destination "$task_dest" \
    --staging-root "$task_epoch/publication-staging" --run "${task_runs[@]}" \
    --extra "${task_extra[@]}" --summary "$task_tools/summary.json" \
    --limitations "Initial qualification only:8 CPU/8 CUDA cases and32 independent fields;7 coarse physical geometry assets. Test-signature formatting is verified by whole-module AST equality. The first spherical sweep is unconverged and includes duplicate mesh levels. Failed bootstrap, meshing and contact attempts are retained. Contact-blocks-v1 was explicitly cancelled for the Python callback/GIL problem; its closure was checked after cancellation before changing code. The full load-model, convergence and coupled deformable-contact stage is unfinished. Live finer/contact trials are excluded."
sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"
cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"
"$task_env/bin/python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals \
    --output "$task_epoch/initial-publication-verification/result.json"
