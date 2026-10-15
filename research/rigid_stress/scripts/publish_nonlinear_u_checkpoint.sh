#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/nonlinear-u-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from research.rigid_stress.interrupted_source_verification import verify_interrupted_source

p, meta = map(Path, sys.argv[1:])
natural = {
    'u-nonlinear-rigid-init-spatial-1-v1': '715802_0',
    'u-nonlinear-rigid-init-spatial-2-v1': '715802_1',
    'u-nonlinear-two-level-fields-v1': '716092_0',
    'u-nonlinear-two-level-fields-v2': '716108_0',
    'grasp-u-objective-spatial-2-v1': '716089_0',
    'grasp-u-objective-replay-v3': '714198_0',
    'u-objective-models-coarse-v1': '716117_0',
    'u-objective-models-medium-v1': '716118_0',
    'u-nonlinear-spatial-model-figure-v1': '716138_0',
    'u-nonlinear-spatial-model-figure-v2': '716256_0',
    'u-nonlinear-spatial-model-figure-v3': '716282_0',
    'u-cad-geometry-measures-v1': '716276_0',
    'franka-original-interrupted-prefix-audit-v1': '716294_0',
    'franka-original-audited-prefix-resume-check-v1': '716300_0',
    'franka-original-audited-prefix-resume-check-audit-v1': '716301_0',
}
observations = sorted((p / 'interrupted-source-observations-v1').glob('*.json'))
interrupted = [json.loads(path.read_text()) for path in observations]
extra_jobs = ['716255_0', '716066_0']
job_ids = sorted({job.split('_')[0] for job in [*natural.values(), *extra_jobs,
                 *(row['job'] for row in interrupted)]})
jobs = subprocess.check_output([
    'sacct', '-n', '-P', '-X', '-j', ','.join(job_ids),
    '--format=JobID%40,State,ExitCode,Elapsed,Start,End,NodeList',
], text=True)
(meta / 'jobs.txt').write_text(jobs)
states = {line.split('|')[0].strip(): line.split('|')[1].strip()
          for line in jobs.splitlines() if line.strip()}
(meta / 'job-commands.txt').write_text(subprocess.check_output([
    'sacct', '-n', '-P', '-X', '-j', ','.join(job_ids), '--format=JobID%40,SubmitLine%16000',
], text=True))
closures, records, extras = [], [], set(observations)
for name, job in natural.items():
    run = p / name
    code = int((run / 'returncode.txt').read_text())
    is_failed_figure = name == 'u-nonlinear-spatial-model-figure-v1'
    assert states[job] == ('FAILED' if is_failed_figure else 'COMPLETED'), (job, states[job])
    assert (code != 0) == is_failed_figure
    for kind in ['source', 'input-source']:
        before = run / (kind + '.sha256')
        if not before.exists():
            continue
        assert before.read_bytes() == (run / (kind + '-after.sha256')).read_bytes(), (name, kind)
        for line in before.read_text().splitlines():
            sha, relative = line.split('  ', 1)
            assert hashlib.sha256((run / kind / relative).read_bytes()).hexdigest() == sha
            closures.append({'run': name, 'identity': kind, 'file': relative, 'sha256': sha})
    records.append({'run': name, 'job': job, 'scheduler_state': states[job],
                    'closure_kind': 'natural_producer_exit', 'returncode': code})
for observation_path, observation in zip(observations, interrupted):
    run = Path(observation['source_run'])
    assert states[observation['job']] == observation['scheduler_state'] == 'FAILED'
    verify_interrupted_source(run, observation_path)
    closures.extend({'run': run.name, **row, 'closure_kind': 'separate_post_termination_observation'}
                    for row in observation['identities'])
    records.append({'run': run.name, 'job': observation['job'], 'scheduler_state': 'FAILED',
                    'closure_kind': 'separate_post_termination_observation', 'returncode': None,
                    'observation_sha256': hashlib.sha256(observation_path.read_bytes()).hexdigest()})
for record in records:
    extras.update(p.glob('*' + record['job'] + '.log'))
for job in extra_jobs:
    extras.update(p.glob('*' + job + '.log'))

def report(name):
    return json.loads((p / name / 'results/summary.json').read_text())

nonlinear = report('u-nonlinear-two-level-fields-v2')
assert len(nonlinear['rows']) == 2 and nonlinear['declared_spatial_axis'] == 'coupled'
for row in nonlinear['rows']:
    for key in ['independent_body_residual_N', 'independent_original_mapped_residual_N',
                'maximum_recorded_original_residual_N']:
        assert row[key] <= 1e-9
for path, sha in nonlinear['input_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
    extras.add(Path(path))
geometry = report('u-cad-geometry-measures-v1')
assert len(geometry['rows']) == 5
for row in geometry['rows']:
    path = Path(row['mesh'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['mesh_sha256']
    extras.add(path)
prefix = report('franka-original-interrupted-prefix-audit-v1')
resumed = report('franka-original-audited-prefix-resume-check-audit-v1')
assert prefix['steps_checked'] == 1875 and not prefix['complete_requested_history']
assert prefix['producer_returncode'] is None
assert prefix['source_closure_kind'] == 'separate_post_termination_observation'
assert resumed['steps_checked'] == 1880 and resumed['complete_requested_history']
assert resumed['requested_steps'] == 1880
for row in resumed['rows']:
    assert row['body_dynamic_residual_N'] <= 1e-9
    assert row['actual_robot_step_effort_residual'] <= 1e-9
qa = json.loads((p / 'meta/nonlinear-u-figure-qa-v1.json').read_text())
assert qa['passed'] and qa['rendered_previews_inspected'] and qa['source_CSV_values_verified'] == 39
qualification = json.loads((p / 'meta/audited-original-prefix-resume-qualification-v1.json').read_text())
assert qualification['all_original1875_saved_arrays_byte_identical']
assert qualification['all1880_independently_audited_original_budget_states']
summary = {
    'scope': 'Two nonlinear U fields, CAD geometry measures, six objective native load-model fields, '
             'interrupted-source observations and qualified original-history prefix resume',
    'runs': records,
    'source_hash_records': closures,
    'nonlinear_U_fields': nonlinear,
    'geometry_measures': geometry,
    'medium_objective_replay': report('grasp-u-objective-spatial-2-v1'),
    'coarse_load_models': report('u-objective-models-coarse-v1'),
    'medium_load_models': report('u-objective-models-medium-v1'),
    'independent1875_interrupted_prefix': prefix,
    'independent1880_short_prefix': resumed,
    'resume_qualification': qualification,
    'figure_qa': qa,
    'delivery_figure_version': 'v3',
    'rejected_figure_versions': {'v1': 'Text-overlap check fails.',
                                'v2': 'Completed render has cropped physical canvas dimensions.'},
    'startup_failures': [{'job': job, 'scheduler_state': states[job],
                          'scope': 'No producer result; source closure is not invented.'} for job in extra_jobs],
    'live_jobs_excluded': ['716302_0', '716303_0', '716308_0', '716309_0', '716310_0',
                           '716311_0', '716316_0', '716317_0', '716318_0', '716319_0',
                           '716320_0', '716321_0'],
    'previous_goal_turn_classification': 'no progress: document/status inspection only; '
                                         'this turn archives and verifies completed evidence',
    'goal_completion': False,
}
(meta / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
extras.update([meta / 'summary.json', meta / 'jobs.txt', meta / 'job-commands.txt',
               p / 'interrupted-source-observations-v1/jobs.txt',
               p / 'meta/nonlinear-u-followup-commands-v1.json',
               p / 'meta/nonlinear-u-figure-programmatic-qa-v1.json',
               p / 'meta/nonlinear-u-figure-qa-v1.json',
               p / 'meta/audited-original-prefix-resume-qualification-v1.json'])
(meta / 'runs.txt').write_text(''.join(str(p / row['run']) + '\n' for row in records))
(meta / 'extras.txt').write_text(''.join(str(path) + '\n' for path in sorted(extras)))
print(json.dumps({'closed_runs': len(natural), 'interrupted_runs': len(interrupted),
                  'source_hash_records': len(closures), 'audited_prefix_states': 1880}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extras < "$task_meta/extras.txt"
task_destination="research/rigid_stress/evidence/20261014-physical/nonlinear-u-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_destination" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extras[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'Two nonlinear U levels retain44.58% tensor change and coupled geometry/contact errors. Native load-model differences are conditional on the fixed3mm footprint and unresolved reference. Seven interrupted sources have separate post-termination observations, not natural EXIT records or invented return codes. All1875 interrupted states and1880 short-resume states are independently audited;235 macros do not complete the original330-macro history. Only figurev3 is accepted for delivery. Live continuations and retries are excluded. Full physical convergence, original application, remaining model/feedback and accuracy/cost gates remain open.' \
    > "$task_meta/publication.log" 2>&1
python3 research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_destination/manifest.json" --originals --output "$task_meta/verification.json" \
    > "$task_meta/verification.log" 2>&1
cp "$task_meta/verification.json" "$task_destination/verification.json"
python3 - "$task_root/runs" "$task_base/publication-staging" "$task_destination" "$task_meta" <<'PY'
import gzip
import hashlib
import json
from pathlib import Path
import sys

root, staging, destination, meta = map(Path, sys.argv[1:])
manifest_path = destination / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
stage = staging / destination.name
verified, generated_chunks = 0, []
for row in manifest['files']:
    original = Path(row['original'])
    if original.is_relative_to(stage):
        actual = root / original.relative_to(stage)
        if not actual.exists():
            assert original.name.endswith('.chunks.json') or '.part' in original.name
            generated_chunks.append(str(original))
            continue
    else:
        actual = original
    digest, size = hashlib.sha256(), 0
    with actual.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
            size += len(data)
    assert digest.hexdigest() == row['sha256'] and size == row['bytes'], actual
    artifact = destination / row['artifact']
    opener = gzip.open if row['encoding'] == 'gzip' else Path.open
    restored = hashlib.sha256()
    with opener(artifact, 'rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            restored.update(data)
    assert restored.hexdigest() == digest.hexdigest(), actual
    verified += 1
result = {
    'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    'true_runtime_original_entries_verified': verified,
    'generated_chunk_entries': generated_chunks,
    'source_hash_records_verified': len(manifest['summary']['source_hash_records']),
    'standard_stored_and_reconstructed_bytes_verification': 'verification.json',
    'passed': True,
    'goal_completion': False,
}
(meta / 'true-runtime-verification.json').write_text(json.dumps(result, indent=2) + '\n')
(destination / 'true-runtime-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
PY
