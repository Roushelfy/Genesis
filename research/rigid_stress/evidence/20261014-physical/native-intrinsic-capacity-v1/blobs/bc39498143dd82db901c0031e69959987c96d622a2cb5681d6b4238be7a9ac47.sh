#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${RIGID_STRESS_DATA_ROOT:-/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery}
task_base="$task_root/runs/20261014-physical-validation"
task_version=${1:-v1}
task_meta="$task_base/contact-limit-publication-$task_version"
test ! -e "$task_meta"
mkdir -p "$task_meta"
python3 - "$task_base" "$task_meta" <<'PY'
import hashlib, json, pathlib, subprocess, sys
p, meta = map(pathlib.Path, sys.argv[1:])
runs = {
    "u-cached-newton-fine-v1": "714883_1",
    "u-cached-newton-fine-equivalence-v1": "715056_0",
    "franka-contact-vertex-geometry-v1": "715064_0",
    "franka-contact-vertex-geometry-v2": "715068_0",
    "franka-contact-vertex-geometry-v3": "715075_0",
    "franka-articulated-regularization-0-pilot-v1": "715058_0",
    "franka-articulated-regularization-1-pilot-v1": "715058_1",
    "franka-articulated-regularization-2-pilot-v1": "715058_2",
    "franka-articulated-regularization-0-hold-v1": "715065_0",
    "franka-articulated-regularization-1-hold-v1": "715065_1",
    "franka-articulated-regularization-2-hold-v1": "715065_2",
    "franka-articulated-regularization-pilot-audit-v1": "715069_0",
    "franka-articulated-regularization-hold-audit-v1": "715070_0",
    "franka-fixed-point-load-control-v1": "715073_0",
    "franka-fixed-point-load-control-v2": "715076_0",
    "franka-tip-mesh-qualification-v2": "715081_0",
    "franka-tip-mesh-qualification-v3": "715082_0",
    "franka-tip-mesh-qualification-v4": "715084_0",
    "franka-tip-contact-geometry-v1": "715085_0",
    "franka-contact-limit-figure-v1": "715089_0",
    "franka-contact-limit-figure-v2": "715096_0",
    "franka-local-coupled-9-v1": "714874_3",
    "franka-local-frozen-9-v1": "714874_7",
    "franka-local-four-level-coupled-fields-v1": "715078_0",
    "franka-local-four-level-frozen-fields-v1": "715078_1",
    "franka-local-four-level-coupled-tractions-v2": "715101_0",
    "franka-local-four-level-frozen-tractions-v2": "715101_1",
    "franka-tip-coupled-0-v1": "715083_0"
}
ids = ','.join(sorted({j.split('_')[0] for j in runs.values()} | {'715079','715100'}))
jobs = subprocess.check_output(['sacct', '-X', '-j', ids, '-o', 'JobID%40,State,ExitCode,Elapsed,End,NodeList', '-n', '-P'], text=True)
(meta/'jobs.txt').write_text(jobs)
commands = subprocess.check_output(['sacct', '-X', '-j', ids, '-o', 'JobID%40,SubmitLine%12000', '-n', '-P'], text=True)
(meta/'job-commands.txt').write_text(commands)
states = {r.split('|')[0].strip(): r.split('|')[1].strip() for r in jobs.splitlines() if r.strip()}
closures, extras = [], set()
for name, job in runs.items():
    run = p/name
    code = int((run/'returncode.txt').read_text())
    assert states[job] == ('COMPLETED' if code == 0 else 'FAILED'), (name, states[job], code)
    for prefix in ['source', 'input-source']:
        before = run/(prefix+'.sha256')
        if not before.exists(): continue
        assert before.read_bytes() == (run/(prefix+'-after.sha256')).read_bytes()
        for line in before.read_text().splitlines():
            expected, relative = line.split('  ', 1)
            source = run/prefix/relative
            if pathlib.Path(relative).is_absolute() and prefix == 'input-source':
                source = run/prefix/'nature_mpl_style.py'
            assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, source
            closures.append({'run': name, 'identity': prefix, 'file': relative, 'sha256': expected})
    for suffix in ['log', 'err']:
        extras.update(p.glob('*'+job+'.'+suffix))
dependencies = []
for name in ['feedback-time-checkpoint-v1','local-tail-articulated-checkpoint-v1','local-refinement-checkpoint-v1']:
    manifest = pathlib.Path('research/rigid_stress/evidence/20261014-physical')/name/'manifest.json'
    dependencies.append({'manifest': str(manifest), 'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()})
def report(name):
    return json.loads((p/name/'results/summary.json').read_text())
summary = {
    'scope': 'Four-level unresolved contact stress, geometric support diagnosis, fixed-point control and contact-distance tails',
    'runs': [{'run': n,'job': j,'state': states[j],'returncode': int((p/n/'returncode.txt').read_text())} for n,j in runs.items()],
    'source_closures': closures, 'dependencies': dependencies,
    'regularization_audit': report('franka-articulated-regularization-hold-audit-v1'),
    'geometry_audit': report('franka-contact-vertex-geometry-v3'),
    'point_load_control': report('franka-fixed-point-load-control-v2'),
    'wide_spatial_fields': [report('franka-local-four-level-'+c+'-fields-v1') for c in ['coupled','frozen']],
    'wide_traction_audits': [report('franka-local-four-level-'+c+'-tractions-v2') for c in ['coupled','frozen']],
    'tip_mesh_qualification': report('franka-tip-mesh-qualification-v4'),
    'tip_contact_geometry': report('franka-tip-contact-geometry-v1'),
    'finer_u_cache_equivalence': report('u-cached-newton-fine-equivalence-v1'),
    'coarse_payload_equivalence': json.loads((p/'tip-coarse-array-equivalence.json').read_text()),
    'figure_qa': json.loads((p/'franka-contact-limit-figure-v2/figure/rendered_qa.json').read_text()),
    'submission_corrections': json.loads((p/'contact-limit-dependency-journal.json').read_text()),
    'failures': ['Geometry v1 used capture directly instead of following the archived closure configuration; source and traceback retained.',
                 'Point control v1 unpacked three values from a two-value stress evaluator; corrected without changing force budgets.',
                 'Tip qualification v2/v3 reach the unchanged60000-cell guard at iteration20 with140103 cells. The second command retained its old level list because escaped-string replacement failed.',
                 'Startup wrapper failures715078_2/3 and cancelled invalid715079/715100_1 have no scientific run directory; Slurm records, logs and correction journal are included.',
                 'Figure v1 completes but fails manual overlap QA; v2 corrects minor tick labels and passes.'],
    'excluded_live_work': ['714462_1','714503_0','714840_0','715021_0','715083_1','715083_2','715083_3','715100_0','715102_0'],
    'goal_completion': False,
}
extras.update([p/'contact-limit-dependency-journal.json',p/'tip-coarse-array-equivalence.json'])
for job in ['715078_2','715078_3','715079_0']:
    extras.update(p.glob('*'+job+'.log'))
for name in ['articulated-regularization-pilot-v1-array.sh','articulated-regularization-hold-v1-array.sh',
             'local-four-level-audit-v1-array.sh','local-four-level-tractions-v2-array.sh',
             'tip-contact-spatial-v1-array.sh','tip-contact-spatial-audit-v1-array.sh']:
    extras.add(p/name)
for level in [0,3,6,9]:
    extras.add(p/f'franka-local-mesh-qualification-v5/results/mesh{level:02d}.npz')
extras.update([p/'franka-local-coupled-0-v1/results/reference_geometry.npz',
               p/'franka-local-coupled-0-v1/results/step0080.npz',
               p/'franka-local-coupled-fields-v1/results/common_probes.npz'])
(meta/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(meta/'runs.txt').write_text(''.join(str(p/n)+'\n' for n in runs))
extras.update([meta/'summary.json',meta/'jobs.txt',meta/'job-commands.txt'])
(meta/'extras.txt').write_text(''.join(str(f)+'\n' for f in sorted(extras)))
print(json.dumps({'terminal_runs':len(runs),'source_identities':len(closures)}))
PY
mapfile -t task_runs < "$task_meta/runs.txt"
mapfile -t task_extra < "$task_meta/extras.txt"
task_dest="research/rigid_stress/evidence/20261014-physical/contact-limit-checkpoint-$task_version"
python3 research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root/runs" --destination "$task_dest" \
    --staging-root "$task_base/publication-staging" --run "${task_runs[@]}" --extra "${task_extra[@]}" \
    --summary "$task_meta/summary.json" \
    --limitations 'The fourth wide-band tensor tail remains32.75% and global peaks reach30.8MPa. Fixed point loads are a separately declared singular-load control, not finite-area contact validation. Contact-distance tails are scoped to one coarse setting. Tip geometry/mesh qualification and exact coarse payloads do not count live finer solves as accepted results. Full original geometry/history/contact/model/feedback/physical accuracy-cost requirements remain active. Exact failed scientific diagnostics and startup corrections are retained.' \
    > "$task_meta/publication.log" 2>&1
python3 research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_meta/verification.json" \
    > "$task_meta/verification.log" 2>&1
cp "$task_meta/verification.json" "$task_dest/verification.json"
python3 - "$task_dest/manifest.json" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);d=json.loads(p.read_text())
print(json.dumps({'files':len(d['files']),'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
PY
