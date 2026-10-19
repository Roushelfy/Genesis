"""Copy bounded terminal evidence without moving or rewriting runtime data."""

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
DESTINATION = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/normal-spatial-cost-v1')
CAMPAIGNS = ('pad-normal-surface-fields-v1', 'pad-spatial-figure-v1', 'pad-spatial-figure-v2', 'profile-p2-stvk-v1', 'pad-primary-sparse-audit-v1')

SUPPORT = ('artifacts', 'common_column_peaks', 'p2_stvk', 'pad_primary_cost', 'native_pad_recovery', 'pad_cost')


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
        if campaign == 'profile-p2-stvk-v1':
            paths.add(folder / 'group0/force/profile.txt')
        if campaign == 'pad-spatial-figure-v2':
            paths.update(p for p in (folder / 'group0/figure').iterdir() if p.is_file())
        for path in sorted(paths):
            copy(path, path.relative_to(ROOT))
    # The earlier packet already retains primary every-step groups0/1.
    folder = ROOT / 'pad-primary-cost-v1'
    for pattern in ('source*.json', 'source*.patch', 'extra-source*', 'submission.json', 'tasks.json', 'analysis-files.json', 'job.sh', 'workload-*.sh'):
        for path in sorted(folder.glob(pattern)):
            if path.is_file(): copy(path, path.relative_to(ROOT))
    for group in (2, 3):
        assert (folder / f'group{group}/returncode.txt').read_text().strip() == '0'
        for pattern in ('returncode.txt', '*.log', '*/start.json', '*/result.json', '*/cost.json'):
            for path in sorted((folder / f'group{group}').glob(pattern)):
                copy(path, path.relative_to(ROOT))
    for module in ('plate_native_pad', 'plate_shell_coupled', 'audit_pad_primary_cost', *SUPPORT):
        path = folder / 'analysis-source/research/rigid_stress/qipc_comparison' / (module + '.py')
        if path.exists(): copy(path, path.relative_to(ROOT))
    copy(Path(__file__), Path('publication-source.py'))
    manifest = {
        'scope': 'Executed full-surface and actual-traction maps, retained failed layout QA, admitted B1 sparse full/max primary costs and nonlinear CPU profile. Larger arrays remain on data; ongoing B4/memory and friction refinements are excluded.',
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
