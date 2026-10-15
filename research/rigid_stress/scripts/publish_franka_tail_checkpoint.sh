#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_python=${PUBLICATION_PYTHON:-python3}
task_epoch="$RIGID_STRESS_DATA_ROOT/runs/20261014-physical-validation"
task_version=${1:-v1}
task_tools="$task_epoch/franka-tail-publication-$task_version"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{capture_franka_fixture,capture_franka_reference,audit_full_franka_controls,audit_franka_contact_regularization,refine_franka_reference,dynamic_franka_reference,qualify_dynamic_franka,sparse_franka_inertia,qualify_sparse_franka,recover_egg_initialization}.py
    research/rigid_stress/sparse_franka_driver.patch
    research/rigid_stress/scripts/{capture_franka_fixture.sh,capture_franka_reference.sh,audit_full_franka_controls.sh,audit_franka_contact_regularization.sh,dynamic_franka_reference.sh,run_sparse_franka.sh,recover_egg_initialization.sh,publish_franka_tail_checkpoint.sh,publish_nonconvex_checkpoint.py,verify_nonconvex_publication.py})
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
task_names=(franka-contact-tail-0-v1 franka-contact-tail-1-v1 franka-contact-tail-2-v1
    franka-contact-tail-audit-v1 franka-full-contact-input-0-v1 franka-full-contact-input-1-v1
    franka-full-contact-input-audit-v1 franka-full-control-audit-v1
    franka-spatial-qualification-0-v1 franka-spatial-qualification-1-v1
    franka-spatial-cold-0-v1 franka-sparse-mass-qualification-v1 franka-sparse-spatial-0-v1
    grasp-egg-spatial-l0-v2)
TASK_TAIL_EPOCH="$task_epoch" TASK_TAIL_NAMES="${task_names[*]}" "$task_python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
from pathlib import Path
root=Path(os.environ['TASK_TAIL_EPOCH']);closures=[]
for name in os.environ['TASK_TAIL_NAMES'].split():
    run=root/name;before=(run/'source.sha256').read_bytes()
    assert before==(run/'source-after.sha256').read_bytes(),name
    for line in before.decode().splitlines():
        expected,relative=line.split('  ',1)
        assert hashlib.sha256((run/'source'/relative).read_bytes()).hexdigest()==expected,(name,relative)
    closures.append({'run':name,'returncode':int((run/'returncode.txt').read_text())})
for name in ('franka-sparse-mass-qualification-v1','franka-sparse-spatial-0-v1'):
    run=root/name;assert (run/'input-source.sha256').read_bytes()==(run/'input-source-after.sha256').read_bytes()
eq=json.loads((root/'franka-sparse-spatial-0-v1/exact-array-equivalence.json').read_text())
assert eq['passed'] and eq['npz_archives_with_exact_array_payloads']==83
mass=json.loads((root/'franka-sparse-mass-qualification-v1/results/summary.json').read_text())
assert mass['max_mass_entry_difference']==mass['max_active_force_difference_N']==mass['max_active_tangent_entry_difference']==0
report={'scope':'Fixed-weld finer contact force tails, complete actual full-arm inputs, fixed-polyhedron qualifications and exact sparse-inertia equivalence',
        'source_closures':closures,'contact_tail':json.loads((root/'franka-contact-tail-audit-v1/results/summary.json').read_text()),
        'full_application_controls':json.loads((root/'franka-full-control-audit-v1/results/summary.json').read_text()),
        'full_input_geometry':json.loads((root/'franka-full-contact-input-audit-v1/results/summary.json').read_text()),
        'sparse_constructor':mass,'cold_protocol_exact_arrays':eq,
        'full_egg_failure':'Terminal lagged-friction initialization failure at increment384. Original input/current/previous states and exact logs are retained; same-state current-force numerical recovery is live and excluded.',
        'live_jobs_excluded':['714003_0','714444_2','714449_1','714462_1','714465_0'],'goal_completion':False}
print(json.dumps(report,indent=2))
PY
task_jobs=(714426 714431 714443 714444 714448 714449 714454 714459 714462)
task_ids=$(IFS=,;printf '%s' "${task_jobs[*]}")
sacct -X -j "$task_ids" -o JobIDRaw,State,ExitCode,Elapsed,End,NodeList -n -P > "$task_tools/jobs.txt"
sacct -X -j "$task_ids" -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
sacct -X -j 714003_3 -o JobIDRaw,State,ExitCode,Elapsed,End,NodeList -n -P > "$task_tools/egg-failed-job.txt"
sacct -X -j 714003_3 -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/egg-failed-command.txt"
task_runs=()
for task_name in "${task_names[@]}";do task_runs+=("$task_epoch/$task_name");done
task_extra=("$task_tools/source.sha256" "$task_tools/jobs.txt" "$task_tools/job-commands.txt" "$task_tools/summary.json"
    "$task_tools/egg-failed-job.txt" "$task_tools/egg-failed-command.txt"
    "$task_epoch/grasp-spatial-next-v1-714003_3.log" "$task_epoch/grasp-spatial-next-v1-714003_3.err"
    "$task_epoch/matrix-mesh-egg-v1/meshes/h0.15.npz")
while IFS= read -r -d '' task_source;do task_extra+=("$task_source");done < <(find "$task_tools/source" -type f -print0)
for task_job in 714426 714431 714443 714448 714454 714459;do
    for task_path in "$task_epoch"/*"$task_job"*.log "$task_epoch"/*"$task_job"*.err;do
        if test -f "$task_path";then task_extra+=("$task_path");fi
    done
done
for task_pair in 714444_0 714444_1 714449_0 714462_0;do
    for task_path in "$task_epoch"/*"$task_pair"*.log "$task_epoch"/*"$task_pair"*.err;do
        if test -f "$task_path";then task_extra+=("$task_path");fi
    done
done
task_dest="research/rigid_stress/evidence/20261014-physical/franka-tail-checkpoint-$task_version"
"$task_python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$RIGID_STRESS_DATA_ROOT/runs" --destination "$task_dest" \
    --staging-root "$task_epoch/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_tools/summary.json" \
    --limitations "Finer normal-force tails are measured at fixed actual weld/integration parameters; they are not full stress/contact-area/shear or spatial acceptance. Complete full-arm actual inputs are qualified but not a deformable full-arm solve. Two fixed-polyhedron spatial qualifications and exact sparse-constructor/83-NPZ-array cold-protocol equivalence are qualified. Fine spatial tasks and original U remain live/excluded. Original full-egg initialization fails, and its same-state recovery remains unverified. Full matched support/history/arm/contact/spatial, load-model/feedback, extrema and physical accuracy/cost acceptance remain required."
"$task_python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
