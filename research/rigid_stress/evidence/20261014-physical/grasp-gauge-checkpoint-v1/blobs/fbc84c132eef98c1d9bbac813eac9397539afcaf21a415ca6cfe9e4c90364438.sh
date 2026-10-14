#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_python=${PUBLICATION_PYTHON:-python3}
task_epoch="$RIGID_STRESS_DATA_ROOT/runs/20261014-physical-validation"
task_tools="$task_epoch/grasp-gauge-publication-v1"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{audit_grasp_gauge,summarize_fine_references}.py
    research/rigid_stress/scripts/{audit_grasp_gauge,publish_grasp_gauge_checkpoint}.sh
    research/rigid_stress/scripts/{publish_nonconvex_checkpoint,verify_nonconvex_publication}.py)
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
cp "$task_epoch/grasp-gauge-audit-v1/results/summary.json" "$task_tools/summary.json"
sacct -X -j 713885 -o JobIDRaw,State,ExitCode,Elapsed -n -P > "$task_tools/jobs.txt"
sacct -X -j 713885 -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
task_dest=research/rigid_stress/evidence/20261014-physical/grasp-gauge-checkpoint-v1
"$task_python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$RIGID_STRESS_DATA_ROOT/runs" --destination "$task_dest" \
    --staging-root "$task_epoch/publication-staging" \
    --run "$task_epoch/grasp-gauge-audit-v1" "$task_tools" \
    --extra "$task_epoch/grasp-gauge-audit-v1-713885_0.log" "$task_epoch/grasp-gauge-audit-v1-713885_0.err" \
    --summary "$task_tools/summary.json" \
    --limitations "The weakest retained eight-coordinate Schur mode is common translation of the free body and both x-free backplates. It is a zero-net-x-force gauge, not evidence of wrench incompatibility. Raw and gauge-removed hold increments are both retained. No contact equations, constraints or acceptance thresholds change; an explicit gauge or nullspace-aware solve remains future implementation work."
sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"
cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"
"$task_python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
