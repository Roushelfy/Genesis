#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_root="$RIGID_STRESS_DATA_ROOT/runs"
task_epoch="$task_root/20261014-physical-validation"
task_version=${1:-v1}
task_tools="$task_epoch/fine-diagnostics-publication-$task_version"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{diagnose_grasp_friction,contact_reference,grasp_contact_reference,force_control_blocks,summarize_fine_references,plot_fine_references}.py
    research/rigid_stress/scripts/publish_fine_diagnostics.sh
    research/rigid_stress/scripts/{publish_nonconvex_checkpoint,verify_nonconvex_publication}.py)
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
git rev-parse HEAD > "$task_tools/base-revision.txt"
hostname > "$task_tools/node.txt"
task_names=(fine-six-summary-v1 fine-six-plots-v1 fine-six-plots-v2
    contact-roundoff-friction-regression-v3 contact-final-legacy-regression-v1
    contact-updated-friction-control-v1 force-control-updated-normal-v1
    grasp-u-coupled-v4 grasp-u-updated-friction-v1 grasp-u-updated-friction-v2 grasp-u-updated-friction-v3
    grasp-u-rigid-basis-z-v1 grasp-u-rigid-basis-y-v1
    grasp-friction-diagnostic-v1 grasp-friction-diagnostic-v2
    grasp-newton-direction-old-v1 grasp-newton-direction-rigid-v1 grasp-rigid-schur-v1)
for task_geometry in ellipsoid capsule u thin_neck cavity narrow_gap; do
    task_names+=("fine-reference-publication-$task_geometry-v1")
done
TASK_DIAGNOSTIC_ROOT="$task_epoch" TASK_DIAGNOSTIC_NAMES="${task_names[*]}" "$task_env/bin/python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
from pathlib import Path
root = Path(os.environ["TASK_DIAGNOSTIC_ROOT"])
records = []
for name in os.environ["TASK_DIAGNOSTIC_NAMES"].split():
    run = root / name
    before = (run / "source.sha256").read_bytes()
    assert before == (run / "source-after.sha256").read_bytes(), name
    for line in before.decode().splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((run / "source" / relative).read_bytes()).hexdigest() == expected, (name, relative)
    records.append({"name": name, "returncode": int((run / "returncode.txt").read_text())})
curves = json.loads((root / "fine-six-summary-v1/results/summary.json").read_text())
schur = json.loads((root / "grasp-rigid-schur-v1/results/result.json").read_text())
print(json.dumps({"scope": "Finer FEM diagnostics, plots, passing contact controls and complete retained failed grasp/publication attempts",
                  "source_closures": records, "refinement_fields": len(curves["rows"]),
                  "tails_below_initial_target": sum(t["all_last_changes_below_diagnostic_target"] for t in curves["tails"]),
                  "rigid_schur": schur["rigid_schur"], "goal_completion": False}, indent=2))
PY
task_runs=()
for task_name in "${task_names[@]}"; do task_runs+=("$task_epoch/$task_name"); done
task_runs+=("$task_tools")
task_jobs=(712874 712881 712883 712839 712813 712888 712890 712913 712916 712945
    712946 712947 712948 712963 712964 712984 713012 713023 713024 713060)
task_ids=$(IFS=,; printf '%s' "${task_jobs[*]}")
sacct -X -j "$task_ids" -o JobIDRaw,State,ExitCode,Elapsed -n -P > "$task_tools/jobs.txt"
sacct -X -j "$task_ids" -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
task_extra=()
for task_job in "${task_jobs[@]}"; do
    for task_path in "$task_epoch"/*"$task_job"*.log "$task_epoch"/*"$task_job"*.err; do
        if test -f "$task_path"; then task_extra+=("$task_path"); fi
    done
done
task_dest="research/rigid_stress/evidence/20261014-physical/fine-diagnostics-$task_version"
"$task_env/bin/python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root" --destination "$task_dest" --staging-root "$task_epoch/publication-staging" \
    --run "${task_runs[@]}" --extra "${task_extra[@]}" --summary "$task_tools/summary.json" \
    --limitations "Six-shape six-level independent FEM diagnostics, ten of thirty complete initial3% last-change diagnostics and three current-source contact controls pass. Original plotting/publisher failures remain exact. Complete free-body U grasp attempts remain unqualified despite updated friction derivatives and alternative numerical coordinates; the preserved near-singular rigid/contact modes and large Newton directions do not alone prove wrench infeasibility. Finer full fields are separately archived by geometry. Egg fine reference tasks, converged resolved egg/Franka/interior-hotspot contact, profile/count/history and warmed native accuracy/cost acceptance remain required."
sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"
cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"
"$task_env/bin/python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
