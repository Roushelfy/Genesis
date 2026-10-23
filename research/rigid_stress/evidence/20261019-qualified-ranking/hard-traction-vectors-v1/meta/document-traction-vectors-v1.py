"""Document measured traction contraction costs and retained numerical checks."""
from pathlib import Path
import json
import textwrap

n = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
w = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
r = json.loads((n / 'meta/traction-vector-terminal-v1.json').read_text())
p = json.loads((n / 'meta/traction-vector-publication-v1.json').read_text())
assert r['passed']
rows = {(row['candidate']['batch'], row['candidate']['mode']): row for row in r['paired_timings']}
profiles = {(row['batch'], row['kind']): row for row in r['profiles']}

def prose(text):
    return textwrap.fill(text, width=88, break_long_words=False, break_on_hyphens=False) + '\n\n'

section = '\n## Boundary traction vector contractions\n\n'
section += prose('The B1024 profile of source f6c4ff2b0 identifies boundary traction as61.85% of measured GPU kernel time (253.730ms mean), combined-field evaluation as28.90% (118.555ms), and source resolution as0.49% (2.029ms). These are intrusive measurements from the same admitted100--129 window. They identify the next optimization target separately from primary throughput.')
section += prose('Candidate e2979aed2 changes only the association of the boundary expression -(B S B^T n)w to -B(S(B^T n))w. Here B is the contact basis, S the local Boussinesq/Cerruti tensor, n the surface normal and w the quadrature weight. Three matrix-vector products replace two matrix-matrix products followed by one matrix-vector product. The quadrature, pressure/friction laws, source geometry, reductions, state buffers and acceptance conditions remain unchanged. The complete-field output still forms its required global tensors.')
section += prose('CPU64 and CUDA64 paired controls retain analytic/estimated curved normal recovery and supplied sharp/friction recovery. Both implementations pass the existing independent elasticity/quadrature readouts, full/max, refusal isolation, release and restoration checks. Ten paired output files compare every field: stress relative1e-7/absolute1e-6Pa, position/displacement absolute1e-12m, force absolute1e-12N and dimensionless fields absolute1e-12; integer and mask outputs compare exactly. These controls verify the implementation on supplied loads. They add no independent contact-pressure qualification or32-bit scene qualification.')
stress, wrench = [], []
for control in r['normal_and_hard_controls']:
    for row in control['complete_output_comparison']['rows']:
        stress += [field for field in row['fields'] if field['field'].endswith('_pa')]
        wrench += [field for field in row['fields'] if field['field'] == 'wrench']
section += prose(f"The largest direct paired stress difference is {max(field['maximum_absolute_difference'] for field in stress):.4g}Pa, with relative L2 difference at most {max(field['relative_l2_difference'] or 0 for field in stress):.4g}. The near-zero wrench-residual diagnostic has relative differences up to {100 * max(field['relative_l2_difference'] for field in wrench):.2f}%, while its largest absolute difference is {max(field['maximum_absolute_difference'] for field in wrench):.4g}. Its original residual gate and the declared absolute comparison both pass. The ratio of two near-zero residuals is retained separately from stress error.")
section += '| Batch | Engine | Traction kernel mean ms | Main combined-field kernel mean ms |\n| --- | --- | ---: | ---: |\n'
for batch in [4, 1024]:
    for kind in ['baseline', 'candidate']:
        means = profiles[batch, kind]['kernel_means_ms']
        assert len(means['kernel_hertz_residual_traction']) == 1 and means['kernel_hertz_combined_field']
        section += f"| {batch} | {kind} | {means['kernel_hertz_residual_traction'][0]:.3f} | {max(means['kernel_hertz_combined_field']):.3f} |\n"
section += '\n' + prose('Each row pair shares a GPU allocation with source-private compiler caches. The baseline/candidate profiler results identify component changes; the following profiler-off wall-clock medians measure complete scenes, including runtime acceptance, controller, dispatch and final synchronization. Reset, pre-window history and audit/export remain outside timing. All modes use64-bit precision and surface order16, with130-frame admission and three100--129 timing repeats.')
section += '| Batch | Engine | Pure rigid ms/step | Max ms/step | Full ms/step | Max/full env-step/s |\n| --- | --- | ---: | ---: | ---: | ---: |\n'
for batch in [4, 1024]:
    for kind in ['baseline', 'candidate']:
        a, b, c = [rows[batch, mode][kind] for mode in ['rigid', 'max', 'full']]
        section += f"| {batch} | {kind} | {a['median_seconds'] * 1000:.3f} | {b['median_seconds'] * 1000:.3f} | {c['median_seconds'] * 1000:.3f} | {b['environment_steps_per_second']:.2f} / {c['environment_steps_per_second']:.2f} |\n"
section += '\n'
for batch in [4, 1024]:
    a, b = [rows[batch, mode] for mode in ['max', 'full']]
    section += prose(f"At B{batch}, max/full median scene-cost reductions are {100*a['scene_cost_reduction_fraction']:.3f}% / {100*b['scene_cost_reduction_fraction']:.3f}%. Candidate inclusive added costs over its own pure-rigid mode are {1000*a['candidate']['added_seconds_per_scene_step']:.3f} / {1000*b['candidate']['added_seconds_per_scene_step']:.3f}ms.")
separated = [f'B{batch} {mode}' for batch in [4, 1024] for mode in ['max', 'full'] if rows[batch, mode]['candidate_repeat_range_below_baseline']]
overlap = [f'B{batch} {mode}' for batch in [4, 1024] for mode in ['max', 'full'] if not rows[batch, mode]['candidate_repeat_range_below_baseline']]
section += prose('Candidate repeat ranges lie entirely below baseline for ' + ', '.join(separated) + '.' if separated else 'No candidate repeat range lies entirely below baseline.')
if overlap: section += prose('This separation does not hold for ' + ', '.join(overlap) + '.')
section += prose('These three repeat blocks supply no statistical confidence interval or portable speedup. B1024 tiles the four starting conditions256 times. Its short admitted window remains separate from independently diverse populations, best-batch/memory limits and full-cycle admission. Primary max/full admit360 timed environment steps each at B4 and92,160 each at B1024. Complete public observations and repeated load/edge counts pass across the paired sources.')
summary = {}
for row in r['full_field_diagnostic']['rows']:
    for field in row['fields']:
        values = summary.setdefault(field['field'], {'failed': 0, 'absolute': 0, 'relative': 0})
        values['failed'] += not field['conformant']
        values['absolute'] = max(values['absolute'], field['maximum_absolute_difference'])
        values['relative'] = max(values['relative'], field['relative_l2_difference'] or 0)
assert {name for name, values in summary.items() if values['failed']} == {'probe_barycentric', 'probe_tensor_pa', 'corner_tensor_pa'}
section += prose('The separate130-frame full capture retains all52 fields and identical discrete statuses. Original raw1e-9 relative/1e-12 absolute comparisons still fail for probe barycentrics in128 frames, probe tensors in126 and corner tensors in128. Their maximum absolute differences are ' + ', '.join(f"{summary[key]['absolute']:.4g}" for key in ['probe_barycentric', 'probe_tensor_pa', 'corner_tensor_pa']) + ' respectively (the tensor values are in Pa). Probe/corner tensor relative L2 differences are at most ' + ' / '.join(f"{summary[key]['relative']:.4g}" for key in ['probe_tensor_pa', 'corner_tensor_pa']) + '. Arithmetic order changes in the traction contraction, and the complete scene also retains upstream curvature/probe-coordinate differences. These raw failures remain visible alongside the passing direct recovery and public output checks; no assertion is relaxed.')
section += prose('A continuous429-step run again admits frames0--427 and reproduces the original frame428 refusal. The sharp-edge singularity, unsupported contact-wrench boundary and independent complete-shell physical-error requirements remain unchanged.')
section += prose('The [traction-vector packet](../evidence/20261019-qualified-ranking/hard-traction-vectors-v1/README.md) retains the initial profile, paired controls/profiles/timings, complete-field differences and original refusal. Captured task and workload scripts record the executed source and arguments. A missing tracked root file in the initial sparse checkout was detected before submission; that partial capture and the verified cone-mode repair are retained. All55 frozen source heads and12 original unfinished files are preserved. Runtime data/logs remain under the established data root, and the measured native source is pinned separately from the original task engine baseline.')
doc = w / 'research/rigid_stress/docs/SHARP_EDGE_PRESSURE_20261022.md'
text = doc.read_text(); assert '## Boundary traction vector contractions' not in text
doc.write_text(text.rstrip() + '\n' + section)
a, b = [rows[1024, mode] for mode in ['max', 'full']]
update = prose(f"The [boundary traction contraction](docs/SHARP_EDGE_PRESSURE_20261022.md#boundary-traction-vector-contractions) follows the B1024 profile that places61.85% of GPU kernel time in boundary traction. A single native expression uses three matrix-vector products. CPU64/CUDA64 complete recovery and public scene checks pass; measured B1024 max/full costs change by {-100*a['scene_cost_reduction_fraction']:+.2f}% / {-100*b['scene_cost_reduction_fraction']:+.2f}%. Strict raw field differences remain recorded, and frame428 still refuses. The profile, primary timing and independent physical-error requirements retain their separate scopes.")
readme = w / 'research/rigid_stress/README.md'
text = readme.read_text(); anchor = 'The preceding [contact-application study]'; assert anchor in text
readme.write_text(text.replace(anchor, update + anchor, 1))
data = w / 'DATA_LOCATION.md';text = data.read_text();header = '# Rigid-stress data and artifact locations\n\n';assert text.startswith(header)
update = prose(f"The `hard-traction-vectors-v1` packet retains {p['payloads']} compact payloads ({p['bytes']:,} bytes) and {p['external_arrays']} external receipts from four terminal campaigns. Initial B1024 profiling, paired normal/sharp recovery, B4/B1024 profiling/timing, full-field captures and frame428 replay remain in the data root below under `hard-large-profile-v1`, `hard-traction-controls-v2`, `hard-traction-profile-v1` and `hard-traction-timing-v1`. The unsubmitted `hard-traction-controls-v1` partial capture remains in place. Source `hard_contact_traction_vectors` is pinned in workspace at e2979aed2. All55 source heads and12 original unfinished files are preserved. Scientific work used Slurm `rtx-mid`, image `1_26`, the existing data-side environment and source-private node caches. Source/docs and compact evidence stay in workspace; no new home fallback output or environment migration occurred.")
data.write_text(header + update + text[len(header):])
print(json.dumps({'documented': True, 'packet': p, 'separated_repeat_ranges': separated, 'other_repeat_ranges': overlap}))
