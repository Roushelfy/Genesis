"""Copy bounded terminal evidence without moving or rewriting runtime data."""

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
DESTINATION = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/normal-pad-reference-v1')
CAMPAIGNS = (
    'compliant-pad-joint-reference-v2', 'compliant-pad-joint-diagonal-v1',
    'full-source-joint-normal-fields-v1', 'pad-normal-qualification-v2',
    'pad-normal-figure-v1', 'solid-pad-stvk-v1', 'solid-normal-constitutive-fields-v1',
    'solid-pad-stvk-modal-v1', 'solid-pad-stvk-modal-v2', 'solid-pad-stvk-warm-v1',
    'solid-pad-matrix-free-v2', 'stvk-solver-equivalence-v1',
    'pad-cost-preflight-v1', 'pad-cost-preflight-audit-v1',
)
SUPPORT = (
    'artifacts', 'p2_stvk', 'solid_pad_step', 'condensed_pad_dynamics',
    'matrix_free_pad_dynamics', 'modal_pad_dynamics', 'common_column_peaks',
    'native_pad_recovery', 'pad_cost',
)


def identity(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    DESTINATION.mkdir(exist_ok=False)
    files, omitted = [], []

    def copy(path, relative):
        receipt = {'source': str(path), 'copy': str(relative), **identity(path)}
        if receipt['bytes'] > 1_000_000:
            omitted.append(receipt)
            return
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert identity(path) == identity(target)
        files.append(receipt)

    for campaign in CAMPAIGNS:
        folder = ROOT / campaign
        paths = set()
        for pattern in ('source*.json', 'source*.patch', 'extra-source*', 'submission.json',
                        'tasks.json', 'analysis-files.json', 'job.sh', 'workload-*.sh',
                        'group*/returncode.txt', 'group*/*.log', 'logs/*.log',
                        'group*/*/result.json', 'group*/*/start.json', 'group*/*/cost.json'):
            paths.update(p for p in folder.glob(pattern) if p.is_file())
        tasks = json.loads((folder / 'tasks.json').read_text())
        modules = {t['module'] for group in tasks for t in group}
        modules.update('research.rigid_stress.qipc_comparison.' + name for name in SUPPORT)
        for module in modules:
            path = folder / 'analysis-source' / (module.replace('.', '/') + '.py')
            if path.exists():
                paths.add(path)
        if campaign == 'pad-normal-qualification-v2':
            paths.add(folder / 'group0/qualification/errors.csv')
        if campaign == 'pad-normal-figure-v1':
            paths.update(p for p in (folder / 'group0/figure').iterdir() if p.is_file())
        for path in sorted(paths):
            copy(path, path.relative_to(ROOT))
    copy(Path(__file__), Path('publication-source.py'))
    manifest = {
        'scope': 'Completed normal reference change qualification and conditional comparisons; objective transient controls; instrumented preflight costs and retained failed audits. Full fields and journals remain on data; active refinements are excluded.',
        'files': files, 'large_files_retained_on_data': omitted,
        'verified_file_count': len(files), 'verified_bytes': sum(f['bytes'] for f in files),
        'whole_goal_complete': False,
    }
    (DESTINATION / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for receipt in files:
        assert identity(DESTINATION / receipt['copy']) == {'bytes': receipt['bytes'], 'sha256': receipt['sha256']}
    print(json.dumps({key: manifest[key] for key in ('verified_file_count', 'verified_bytes')}))


if __name__ == '__main__':
    main()
