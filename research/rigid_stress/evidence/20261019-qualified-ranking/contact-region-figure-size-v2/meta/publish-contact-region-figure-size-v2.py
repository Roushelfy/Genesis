"""Retain the exact-size pressure figure beside the original cropped export."""
import ast
import datetime
import json
import pathlib
import re
import struct
import subprocess

RUN = pathlib.Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
META = RUN / 'meta'
HELPER = META / 'publish-canonical125-resilient-recovery-v1.py'
exec(compile(HELPER.read_text().split('\nimport ast\n', 1)[0], str(HELPER), 'exec'))
read = lambda p: json.loads(p.read_text())
campaign = 'contact-region-figure-v2'
root = RUN / campaign
figure = root / 'group0/figure'
result = read(figure / 'result.json')
assert result['passed'] and not result['rendered_text_issues'] and result['svg_text_editable']
assert (root / 'group0/returncode.txt').read_text().strip() == '0'
job = str(read(root / 'submission.json')['job']).split(';')[0]
accounting = subprocess.check_output(['sacct', '-j', job + '_0', '-P', '--noheader',
                                     '--format=JobID,State,ExitCode,Elapsed,End'], text=True)
assert any(line.split('|')[:3] == [job + '_0', 'COMPLETED', '0:0'] for line in accounting.splitlines())

def measure(path):
    svg = path.read_text()
    match = re.search(r'<svg\b[^>]*width="([0-9.]+)pt"[^>]*height="([0-9.]+)pt"', svg)
    assert match
    return [float(value) * 25.4 / 72 for value in match.groups()]

old = RUN / 'contact-region-figure-v1/group0/figure'
actual_mm = measure(figure / 'contact-pressure.svg')
old_mm = measure(old / 'contact-pressure.svg')
assert all(abs(actual - expected) <= 1e-6 for actual, expected in zip(actual_mm, [183, 145]))
assert any(abs(actual - expected) > 1 for actual, expected in zip(old_mm, [183, 145]))
contract = read(figure / 'figure-contract.json')
previous = read(old / 'figure-contract.json')
assert contract == previous, 'Pressure data, claims, axes, scale or panel selection changed'
source = root / 'analysis-source/research/rigid_stress/qipc_comparison/plot_contact_region_study.py'
assert source.read_bytes() == (WORK / 'research/rigid_stress/qipc_comparison/plot_contact_region_study.py').read_bytes()
visual = META / 'contact-region-figure-visual-qa-v2.json'
qa = read(visual)
assert qa['passed'] and identity(pathlib.Path(qa['preview']['path'])) == qa['preview']
png = (figure / 'contact-pressure.png').read_bytes()
pixels = list(struct.unpack('>II', png[16:24]))
assert all(abs(actual - expected / 25.4 * 600) <= 1 for actual, expected in zip(pixels, [183, 145]))
assert '<text' in (figure / 'contact-pressure.svg').read_text()
assert b'/FontFile2' in (figure / 'contact-pressure.pdf').read_bytes()
receipt = {'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'passed': True, 'actual_scheduler_accounting': accounting,
    'original_cropped_dimensions_mm': old_mm, 'corrected_dimensions_mm': actual_mm,
    'png_pixels_at600dpi': pixels, 'source': identity(source),
    'pressure_contract_and_source_data_identical': True,
    'zero_pressure_background_replaces_only_off_canvas_zero_polygons': True,
    'all_exact_nonzero_parent_facets_inside_limits': True,
    'svg_text_editable': True, 'pdf_embedded_truetype_marker_present': True,
    'original_frozen_packet_not_rewritten': True, 'visual_qa': identity(visual),
    'whole_goal_complete': False}
p = META / 'contact-region-figure-size-verified-v2.json'
with p.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files = [pathlib.Path(__file__), HELPER, visual, p, META / 'contact-region-nature-style-v1.py']
files += captured(campaign, ['qipc_comparison/plot_contact_region_study.py', 'qipc_comparison/artifacts.py'])
files += terminal(campaign)
for row in result['exports'] + [result['preview']]:
    assert identity(pathlib.Path(row['path'])) == row
    files.append(pathlib.Path(row['path']))
row = freeze('contact-region-figure-size-v2', files, [],
    'Correct the pressure diagnostic export dimensions without changing its data, claims, panel selection, common kPa scale or physical axes. The shared style default savefig.bbox=tight cropped the original183x145mm canvas to163.055x139.449mm; that frozen original and its earlier readability inspection remain intact. '
    'The renderer now explicitly disables automatic cropping and draws all nonzero parent facets inside verified polar limits against the matching zero-pressure background, excluding only zero polygons outside the figure. Actual SVG dimensions are183x145mm;600dpi PNG dimensions, editable SVG text, embedded TrueType PDF marker and fresh rendered visual/text checks pass. '
    'This companion supersedes only the first figure size claim/artwork. All original CPU support, double-spreading and paired elastic results remain unchanged. No physical simulation or production engine is rerun or edited, and the extended goal remains active.')
with (META / 'contact-region-figure-size-publication-v2.json').open('x') as stream:
    stream.write(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))
