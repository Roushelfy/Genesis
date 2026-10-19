"""Copy bounded terminal evidence without moving or rewriting runtime data."""

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
DESTINATION = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/common-peaks-batch-v1')
CAMPAIGNS = ('common-pad-peaks-v1', 'common-pad-peak-reference-v1', 'common-pad-peak-loads-v1', 'pad-common-peak-qualification-v2', 'shell-paired-output-v1', 'pad-cost-sparse-preflight-v1', 'pad-cost-sparse-audit-v1', 'native-pad-batch-control-v1', 'native-pad-batch-motion-v1', 'native-pad-batch-audit-v1', 'native-pad-batch-scr-audit-v1', 'shell-pad-batch-motion-v1', 'shell-pad-batch-audit-v1', 'solid-pad-stvk-half-dt-v1', 'solid-pad-stvk-contact-quadrature-v1', 'stvk-time-quadrature-fields-v1', 'solid-pad-stvk-spatial500-v1', 'solid-pad-stvk-direct-control-v1', 'stvk-direct-equivalence-v1', 'pad-primary-every-audit-v1', 'pad-response-figure-v1', 'pad-response-figure-v2')

SUPPORT = (
    'artifacts', 'p2_stvk', 'solid_pad_step', 'condensed_pad_dynamics',
    'matrix_free_pad_dynamics', 'modal_pad_dynamics', 'common_column_peaks',
    'native_pad_recovery', 'pad_cost', 'pad_primary_cost', 'shell_pad_audit',
    'native_pad_batch_recovery', 'native_pad_batch_mechanics', 'plates',
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
        if campaign == 'pad-common-peak-qualification-v2':
            paths.add(folder / 'group0/qualification/common-peaks.csv')
        if campaign in ('pad-response-figure-v1', 'pad-response-figure-v2'):
            paths.update(p for p in (folder / 'group0/figure').iterdir() if p.is_file())
        for path in sorted(paths):
            copy(path, path.relative_to(ROOT))
    # Only completed immutable leaves from still-active campaign arrays.
    partials = {
        'pad-primary-cost-v1': [f'group{g}/{m}' for g in (0, 1) for m in ('dpr', 'scr', 'shell', 'dpr-audit', 'scr-audit', 'shell-audit')],
        'pad-primary-batch4-v1': [f'group{g}/{m}' for g in (0, 1) for m in ('dpr', 'dpr-audit')],
        'pad-memory-profile-v1': [f'group{g}/{m}' for g in (0, 1) for m in ('dpr', 'dpr/run', 'dpr-audit')],
    }
    for campaign, leaves in partials.items():
        folder = ROOT / campaign
        paths = set()
        for pattern in ('source*.json', 'source*.patch', 'extra-source*', 'submission.json', 'tasks.json', 'analysis-files.json', 'job.sh', 'workload-*.sh'):
            paths.update(p for p in folder.glob(pattern) if p.is_file())
        for leaf in leaves:
            assert (folder / leaf / 'result.json').exists()
            for name in ('result.json', 'start.json', 'cost.json'):
                path = folder / leaf / name
                if path.exists(): paths.add(path)
        modules = {t['module'] for group in json.loads((folder / 'tasks.json').read_text()) for t in group}
        modules.update('research.rigid_stress.qipc_comparison.' + name for name in SUPPORT)
        modules.update('research.rigid_stress.qipc_comparison.' + name for name in ('plate_native_pad_batch', 'plate_shell_pad_batch', 'pad_memory_profile'))
        modules.add('research.rigid_stress.shell_comparison.memory')
        for module in modules:
            path = folder / 'analysis-source' / (module.replace('.', '/') + '.py')
            if path.exists(): paths.add(path)
        for path in sorted(paths): copy(path, path.relative_to(ROOT))
    for path in sorted((ROOT / 'meta').glob('memory-b*-v1.json')):
        copy(path, path.relative_to(ROOT))
    copy(Path(__file__), Path('publication-source.py'))
    manifest = {
        'scope': 'Completed common-peak qualification, actual B4 conformance, transient time/contact/direct controls, and force-closure figure. Selected completed primary B1 every-step leaves, B4 DPR and B1 DPR memory leaves are explicitly partial campaign evidence. Other active leaves are excluded. Full fields and journals remain on data.',
        'files': files, 'large_files_retained_on_data': omitted,
        'verified_file_count': len(files), 'verified_bytes': sum(f['bytes'] for f in files),
        'whole_goal_complete': False, 'partial_campaign_completed_leaves': partials,
    }
    (DESTINATION / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for receipt in files:
        assert identity(DESTINATION / receipt['copy']) == {'bytes': receipt['bytes'], 'sha256': receipt['sha256']}
    print(json.dumps({key: manifest[key] for key in ('verified_file_count', 'verified_bytes')}))


if __name__ == '__main__':
    main()
