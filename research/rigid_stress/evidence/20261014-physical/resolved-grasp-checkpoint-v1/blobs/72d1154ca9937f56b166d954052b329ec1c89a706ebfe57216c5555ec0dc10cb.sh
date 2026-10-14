#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_python=${PUBLICATION_PYTHON:-python3}
task_root="$RIGID_STRESS_DATA_ROOT/runs"
task_epoch="$task_root/20261014-physical-validation"
task_version=${1:-v1}
task_tools="$task_epoch/resolved-grasp-publication-$task_version"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{audit_grasp_support,replay_resolved_grasp,plot_resolved_grasp}.py
    research/rigid_stress/scripts/publish_resolved_grasp_checkpoint.sh
    research/rigid_stress/scripts/{publish_nonconvex_checkpoint,verify_nonconvex_publication}.py)
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
git rev-parse HEAD > "$task_tools/base-revision.txt"
task_names=(grasp-u-spatial-y-l1-v1 grasp-u-spatial-zero-l1-v1 grasp-support-coarse-v1 grasp-support-coarse-v2
    grasp-u-replay-y-l1-v1 grasp-u-replay-zero-l1-v1 grasp-u-replay-y-l1-v2 grasp-u-replay-zero-l1-v2
    grasp-u-plots-v1 grasp-u-plots-v2 grasp-u-plots-v3)
TASK_GRASP_EPOCH="$task_epoch" TASK_GRASP_NAMES="${task_names[*]}" "$task_python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["TASK_GRASP_EPOCH"])
records = []
for name in os.environ["TASK_GRASP_NAMES"].split():
    run = root / name
    before = (run / "source.sha256").read_bytes()
    assert before == (run / "source-after.sha256").read_bytes(), name
    for line in before.decode().splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((run / "source" / relative).read_bytes()).hexdigest() == expected, (name, relative)
    records.append({"name": name, "returncode": int((run / "returncode.txt").read_text())})
replays = {case: json.loads((root / f"grasp-u-replay-{case}-l1-v2/results/summary.json").read_text()) for case in ("y", "zero")}
support = json.loads((root / "grasp-support-coarse-v2/results/summary.json").read_text())
print(json.dumps({"scope": "Full free-body U contact pilots, bounded fixed-stencil wrench certificates, exact raw-tensor native replay and pad strain limitations",
                  "source_closures": records, "replays": replays, "fixed_stencil_support": support,
                  "goal_completion": False}, indent=2))
PY
task_runs=()
for task_name in "${task_names[@]}"; do task_runs+=("$task_epoch/$task_name"); done
task_runs+=("$task_tools")
task_extra=("$task_epoch/grasp-u-spatial-v1.sbatch" "$task_epoch/grasp-u-spatial-v1.json"
    "$task_epoch/grasp-u-replay-v1.sbatch" "$task_epoch/grasp-u-replay-v2.sbatch")
task_jobs=(713653 713694 713696 713701 713844 713853 713858 713859 713863)
task_ids=$(IFS=,; printf '%s' "${task_jobs[*]}")
sacct -X -j "$task_ids" -o JobIDRaw,State,ExitCode,Elapsed -n -P > "$task_tools/jobs.txt"
sacct -X -j "$task_ids" -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
for task_job in 713694 713696 713844 713853 713858 713859 713863; do
    for task_path in "$task_epoch"/*"$task_job"*.log "$task_epoch"/*"$task_job"*.err; do
        if test -f "$task_path"; then task_extra+=("$task_path"); fi
    done
done
for task_index in 0 2; do
    task_extra+=("$task_epoch/grasp-u-spatial-v1-713653_$task_index.log"
        "$task_epoch/grasp-u-spatial-v1-713653_$task_index.err")
done
task_dest="research/rigid_stress/evidence/20261014-physical/resolved-grasp-checkpoint-$task_version"
"$task_python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root" --destination "$task_dest" --staging-root "$task_epoch/publication-staging" \
    --run "${task_runs[@]}" --extra "${task_extra[@]}" --summary "$task_tools/summary.json" \
    --limitations "Two full3D U free-body force-controlled contact pilots meet unchanged equilibrium gates after spatial refinement. Actual raw native full tensors and nodal replay match independent contact-body elasticity. Fixed-stencil cone certificates do not exclude contact birth or pose changes. Original interpreter-exit and figure-QA failures remain exact. Local pads reach5–6% linear strain, so nonlinear compliance, contact spatial/regularization/history tails, finite-footprint replay, rigid feedback and full egg/Franka matching remain required. Live finer-U and egg jobs are outside this immutable checkpoint."
sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"
cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"
"$task_python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
