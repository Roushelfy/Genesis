"""Publish verified terminal receipts; leave runtime fields and journals on data."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
DESTINATION = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/complete-pad-cost-v1')
CAMPAIGNS = ('pad-primary-batch4-v1', 'pad-cost-matrix-full-v2', 'pad-phase-profile-v1')
SUPPORT = ('artifacts', 'p2_stvk', 'plate_solid_pad', 'solid_pad_contact', 'solid_pad_modal', 'common_column_peaks', 'contact_fem_exact_columns', 'pad_primary_cost', 'pad_cost', 'native_pad_batch_recovery', 'plate_native_pad_batch', 'plate_shell_pad_batch', 'audit_pad_batch', 'pad_cost_matrix', 'pad_phase_profile', 'plate_native_pad', 'plate_shell_coupled', 'native_pad_recovery', 'audit_native_pad', 'audit_coupled_shell')

def identity(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    # Check all terminal groups before creating the publication.
    for campaign in CAMPAIGNS:
        tasks = json.loads((ROOT / campaign / 'tasks.json').read_text())
        for i in range(len(tasks)):
            assert (ROOT / campaign / f'group{i}/returncode.txt').read_text().strip() == '0'
    DESTINATION.mkdir(exist_ok=False)
    files, omitted, copied = [], [], set()
    def copy(path, relative):
        if str(relative) in copied:
            return
        copied.add(str(relative))
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
        for pattern in ('source*.json', 'source*.patch', 'extra-source*', 'submission.json', 'tasks.json', 'analysis-files.json', 'job.sh', 'workload-*.sh', 'group*/returncode.txt', 'group*/*.log', 'logs/*.log', 'group*/*/result.json', 'group*/*/start.json', 'group*/*/cost.json', 'group*/*/run/start.json', 'group*/*/run/result.json', 'group*/*/run/cost.json', 'group*/*/audit/result.json', 'group*/*/phases.json', 'group*/*/phases.csv', 'group*/*/timings.csv', 'group*/*/memory.csv'):
            paths.update(p for p in folder.glob(pattern) if p.is_file())
        tasks = json.loads((folder / 'tasks.json').read_text())
        modules = {t['module'] for group in tasks for t in group}
        modules.update('research.rigid_stress.qipc_comparison.' + name for name in SUPPORT)
        modules.add('research.rigid_stress.shell_comparison.memory')
        for module in modules:
            path = folder / 'analysis-source' / (module.replace('.', '/') + '.py')
            if path.exists(): paths.add(path)
        for group in tasks:
            for task in group:
                if '--task' in task['args']:
                    path = Path(task['args'][task['args'].index('--task') + 1])
                    copy(path, path.relative_to(ROOT))
        for path in sorted(paths): copy(path, path.relative_to(ROOT))
    # Retain the two premature dependent analyses as failures/partial diagnostics.
    for campaign in ('pad-cost-matrix-full-v1', 'stvk-spatial500-fields-v1'):
        folder = ROOT / campaign
        for pattern in ('submission.json', 'tasks.json', 'scheduler-dependency.txt', 'group*/returncode.txt', 'group*/*.log'):
            for path in sorted(folder.glob(pattern)):
                if path.is_file(): copy(path, path.relative_to(ROOT))
    path = ROOT / 'meta/premature-dependent-analyses-v1.json'
    copy(path, path.relative_to(ROOT))
    copy(Path(__file__), Path('publication-source.py'))
    manifest = {'scope': 'Terminal B4 primary every/sparse full/max timings, verified B1/B4 timing-memory aggregation and separately admitted intrusive solver-phase profiles. Earlier B1 timing and memory evidence remain in their preceding packets; large fields and journals remain on data. Two prematurely scheduled analyses are retained as failures/partial diagnostics and cannot qualify their still-running parent references. Fine friction reference qualification and final cross-case comparison remain open.', 'files': files, 'large_files_retained_on_data': omitted, 'verified_file_count': len(files), 'verified_bytes': sum(f['bytes'] for f in files), 'whole_goal_complete': False}
    (DESTINATION / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for r in files: assert identity(DESTINATION / r['copy']) == {'bytes': r['bytes'], 'sha256': r['sha256']}
    print(json.dumps({k:manifest[k] for k in ('verified_file_count','verified_bytes')}))

if __name__ == '__main__':
    main()
