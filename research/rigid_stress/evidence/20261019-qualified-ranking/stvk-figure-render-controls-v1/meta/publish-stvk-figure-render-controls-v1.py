"""Freeze terminal renderer controls, inspected previews and guarded final figure inputs."""
import gzip
import hashlib
import io
import json
import pathlib
import subprocess

RUN = pathlib.Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
WORK = pathlib.Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
DEST = WORK / 'research/rigid_stress/evidence/20261019-qualified-ranking'
META = RUN / 'meta'
SCRIPT = pathlib.Path(__file__)


def identity(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hasher.hexdigest()}


def freeze(name, files, external, scope):
    target = DEST / name
    target.mkdir(exist_ok=False)
    rows = []
    for source in sorted(set(files)):
        original = source.read_bytes()
        relative = source.relative_to(RUN)
        encoded = original
        compressed = bool(original) and (source.suffix == '.log' or (source.suffix == '.json' and len(original) > 8192) or source.suffix == '.patch')
        if compressed:
            memory = io.BytesIO()
            with gzip.GzipFile(filename='', mode='wb', fileobj=memory, mtime=0) as stream:
                stream.write(original)
            encoded = memory.getvalue()
            relative = pathlib.Path(str(relative) + '.gz')
        copy = target / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(encoded)
        rows.append({'source': str(source), 'copy': str(relative), 'bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest(), 'encoding': 'gzip' if compressed else 'identity', 'original_bytes': len(original), 'original_sha256': hashlib.sha256(original).hexdigest()})
    manifest = {'files': rows, 'external_files': [identity(path) for path in external], 'scope': scope, 'payloads': len(rows), 'payload_bytes': sum(row['bytes'] for row in rows)}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (target / 'README.md').write_text('# ' + name + '\n\n' + scope + '\n\nManifest records encoded and original SHA-256/size. Gzip uses mtime0 and an empty filename. Decode to a separate working directory when replaying; immutable archive bytes stay unchanged. Large trajectory/field arrays remain at the explicit data-side locations in the receipts. The original1ff39b077 engine can be reconstructed from the public-base overlay in [the source packet](../native-public-source-overlay-v1/README.md). Captured worker modules retain publication-time variants. Live fine histories and pending method/ranking/figure outputs are excluded.\n')
    return {'packet': name, 'payloads': len(rows), 'bytes': manifest['payload_bytes'], 'external_arrays': len(external)}


def captured(campaign, modules):
    root = RUN / campaign
    files = [root / n for n in ('submission.json', 'source.json', 'source.patch', 'analysis-files.json', 'tasks.json', 'job.sh')]
    files += list(root.glob('workload-*.sh'))
    for module in modules:
        files.append(root / 'analysis-source/research/rigid_stress' / module)
    return files


def terminal(campaign):
    root = RUN / campaign
    return [p for p in root.glob('group*/**/*') if p.is_file() and p.suffix in ('.json', '.txt', '.log') and 'cache' not in p.parts and 'tmp' not in p.parts] + list(root.glob('logs/*.log'))


expected = {'729875_0': ('FAILED', '1:0'), '729877_0': ('FAILED', '1:0'), '729882_0': ('FAILED', '1:0'), '729932_0': ('COMPLETED', '0:0'), '729930_0': ('FAILED', '1:0')}
accounting = subprocess.check_output(['sacct', '-j', '729875,729877,729882,729932,729930', '-P', '--noheader', '--format=JobID,State,ExitCode'], text=True)
observed = {parts[0]: tuple(parts[1:3]) for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
assert all(observed.get(job) == status for job, status in expected.items()), observed
passed = json.loads((RUN / 'stvk-figure-render-controls-v4/group0/figures/result.json').read_text())
negative = json.loads((RUN / 'stvk-final-figure-admission-negative-v1/group0/unqualified-normal-reference/result.json').read_text())
inspection = json.loads((META / 'stvk-rendered-preview-inspection-v1.json').read_text())
assert passed['passed'] and len(passed['checks']) == 15 and all(passed['checks'].values())
assert not negative['passed'] and negative['figures'] == {} and negative['error'] == 'explicit_fine_friction_reference_qualified'
assert inspection['preview_inspected'] and inspection['visual_qa_passed'] and len(inspection['figures']) == 4
for row in inspection['figures'].values():
    assert identity(pathlib.Path(row['preview']['path'])) == row['preview'] and not row['text_issues']
contract = json.loads((META / 'canonical-stvk-figure-contract-v1.json').read_text())
for name, digest in contract['source_sha256'].items():
    assert identity(RUN / 'canonical-stvk-figures-v1/analysis-source/research/rigid_stress/qipc_comparison' / name)['sha256'] == digest
files = [SCRIPT] + [META / name for name in ('stvk-render-controls-terminal-v1.json', 'stvk-rendered-preview-inspection-v1.json', 'canonical-stvk-figure-contract-v1.json', 'stvk-figure-render-controls-v1-tasks.json', 'stvk-final-figure-admission-negative-v1-tasks.json', 'canonical-stvk-figures-v1-tasks.json')]
external = []
control_modules = ['qipc_comparison/' + name for name in ('artifacts.py', 'pad_spatial_figure.py', 'stvk_figure_render.py', 'check_stvk_figure_rendering.py')]
for version in range(1, 5):
    campaign = 'stvk-figure-render-controls-v' + str(version)
    files += captured(campaign, control_modules) + terminal(campaign)
    figures = RUN / campaign / 'group0/figures'
    files += list(figures.glob('*.md'))
    for path in figures.glob('*'):
        if not path.is_file():
            continue
        if version == 4 and path.name.endswith('_preview.png'):
            files.append(path)
        elif path.suffix in ('.svg', '.pdf', '.png', '.npz', '.csv'):
            external.append(path)
files.append(RUN / 'stvk-figure-render-controls-v4/group0/figures/nature_mpl_style.py')
final_modules = ['qipc_comparison/' + name for name in ('artifacts.py', 'pad_spatial_figure.py', 'stvk_figure_render.py', 'plot_stvk_pad_figures.py')]
files += captured('stvk-final-figure-admission-negative-v1', final_modules) + terminal('stvk-final-figure-admission-negative-v1')
files += captured('canonical-stvk-figures-v1', final_modules)
row = freeze('stvk-figure-render-controls-v1', files, external, 'Terminal15 renderer/data/export controls and four visually inspected previews using already qualified normal fields and mechanically admitted500um friction data. Every material face, force-bearing sample, signed tangential component and actual-mass mechanical history remains retained. A normal-only qualification is intentionally rejected by the guarded final renderer. Original failed unit comparison and out-of-history tick warnings, their exact source variants, positive exports/QA and the immutable pending fine-render plan are preserved. Full plotted arrays and SVG/PDF/high-resolution PNG exports stay on data with hashes; final fine friction qualification, errors/ranks, artwork and matched consolidation remain pending. These controls do not complete the active goal.')
(META / 'stvk-figure-render-controls-publication-v1.json').write_text(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))
