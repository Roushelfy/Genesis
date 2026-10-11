"""Archive only terminal JSON cases and explicitly completed test/source files."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

run = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261010-contact-repair')
out = Path('research/rigid_stress/evidence/20261010-contact-repair')
manifest = out / 'manifest.json'
previous = json.loads(manifest.read_text())
known = {entry['runtime_path'] for entry in previous['artifacts']}
names = set()
patterns = (
    'native-controller-*-v3.json',
    'native-episode-extended-*-v2.json',
    'native-rollout-profile-*-v4.json',
    'native-reset-*-v1.json',
    'policy-graph-*-v1.json',
    'pipeline-fastcache-*-v1.json',
    'native-observation-equivalence-*-v1.observation-variant.json',
    'native-scheduling-original-*-v6.json',
    'native-scheduling-combined-*-v6.json',
    'native-scheduling-original-combined-*-v6-pair-summary.json',
    'native-scheduling-*-live-b32768-seed*-v6.json',
    'native-scheduling-ablation-*-v6-summary.json',
    'native-scheduling-partial-v6-summary.json',
    'native-production-*-v7.json',
    'native-production-*-v8.json',
    'native-contact-motion-*-v1.json',
    'native-index-reset-equivalence-*-v1.json',
    'native-observation-*-v1.json',
    'native-observation-*-v2.json',
    'native-observation-*-v2-summary.json',
    'native-observation-*-v2.observation-variant.json',
    'native-indices-*-v1.json',
    'native-indices-*-v1-summary.json',
    'native-episode-extended-all-v2-summary.json',
    'native-rollout-profile-all-v4-summary.json',
)
for pattern in patterns:
    for path in run.glob(pattern):
        if str(path) in known:
            continue
        # Benchmark JSON is written at completion. Adjacent generated sources,
        # traces, metadata and logs are closed before the next case starts.
        stem = path.name.removesuffix('.json')
        if stem.endswith(('.policy-variant', '.reset-variant', '.fastcache-variant', '.observation-variant')):
            stem = stem.rsplit('.', 1)[0]
        names.add(path.name)
        for suffix in ('.log', '.trace.json', '.reset-variant.json', '.set-state.py', '.policy-variant.json', '.fastcache-variant.json', '.pipeline.py', '.observation-variant.json', '.indices-variant.json', '.controller.py'):
            candidate = run / (stem + suffix)
            if candidate.is_file():
                names.add(candidate.name)

explicit = (
    'native-production-gpu-v7.log', 'native-production-cpu-v7.log',
    'native-production-integration-v7.patch', 'native-production-controller-v7.py',
    'native-production-recovery-v7.py',
    'native-shared-contact-moments-original-v2.log',
    'native-shared-link-regression-original-v2.patch',
    'native-link-invalidation-gpu-v5.log', 'native-link-invalidation-cpu-v5.log',
    'native-link-invalidation-gpu-v4.log', 'native-shared-contact-moments-v1.log',
    'native-controller-input-oracle-frozen-v3.py',
    'native-controller-production-frozen-v3.py', 'native-episode-probe-frozen-v2.py',
    'native-observation-probe-frozen-v1.py', 'native-reset-profile-frozen-v1.py',
    'native-reset-probe-frozen-v1.py', 'native-rollout-profile-probe-frozen-v1.py',
    'pipeline-fastcache-probe-frozen-v1.py', 'policy-graph-probe-frozen-v1.py',
    'moment-checkpoint-v7-gzip-evidence-verification.json',
    'moment-checkpoint-v7-gzip-evidence-verification.log',
    'moment-checkpoint-v7-trace-gzip-archive.log',
    'native-production-gpu-v8.log', 'native-production-cpu-v8.log',
    'native-indices-production-v8.patch', 'native-indices-production-controller-v8.py',
    'native-indices-production-benchmark-v8.py',
    'native-index-reset-oracle-frozen-v1.py', 'native-index-reset-oracle-frozen-v2.py',
    'native-indices-probe-frozen-v1.py', 'native-observation-compact-probe-frozen-v2.py',
    'native-checkpoint-v8-selection.json', 'native-checkpoint-v8-verification.json',
    'native-checkpoint-v8-archive.log',
)
for name in explicit:
    if (run / name).is_file():
        names.add(name)
names = sorted(name for name in names if str(run / name) not in known)
selection = {
    'names': names,
    'previous_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
    'selector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'note': 'Completed heterogeneous checkpoints. Exact source revisions/hashes are in each artifact; current production v7 is not claimed as the source of earlier prototypes.',
}
(run / 'native-checkpoint-v9-selection.json').write_text(json.dumps(selection, indent=2) + '\n')
subprocess.run([sys.executable, 'research/rigid_stress/scripts/archive_completed_evidence.py', '--run', str(run), '--output', str(out), '--revision', 'f4965c70-checkpoint-v9-see-per-artifact-source', *names], check=True)
subprocess.run([sys.executable, 'research/rigid_stress/scripts/verify_contact_evidence.py', '--manifest', str(manifest), '--originals', '--output', str(run / 'native-checkpoint-v9-verification.json')], check=True)
oversized = [(path.name, path.stat().st_size) for path in out.iterdir() if path.is_file() and path.stat().st_size >= 100_000_000]
assert not oversized, oversized
print('Checkpoint v9 archived and verified; no Git artifact reaches 100 MB.', flush=True)
