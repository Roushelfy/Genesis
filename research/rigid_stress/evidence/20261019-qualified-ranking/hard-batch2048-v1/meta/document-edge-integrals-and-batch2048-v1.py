"""Insert terminal measurements into the existing reports without changing old evidence."""
from pathlib import Path
import json

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
W = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
read = lambda p: json.loads(p.read_text())
integrals = read(N / 'edge-shell-integrals-inspection-v4/group0/inspection/result.json')
batch = read(N / 'hard-batch-scaling2048-inspection-v1/group0/inspection/result.json')
assert integrals['execution_completed'] and batch['passed']
v = integrals['volume']
windows = [read(N / p / 'result.json') for p in [
    'edge-shell-volume-window-v3/group0/window', 'edge-shell-volume-window-v2/group1/window']]
text = '\nThe cell-aligned pair uses angular/radial/along orders8/8/4 and12/8/6,\n'
text += 'with' + ' /'.join(f'{p["quadrature"]["points"]:,}' for p in windows) + ' positive-weight points. '
text += 'Both radial rules have order8.\n'
text += 'Their minimum positive depths are' + ' /'.join(f'{1e9*p["quadrature"]["minimum_depth_m"]:.6g}nm' for p in windows) + ';\n'
text += 'maximum analytic moment errors are' + ' /'.join(f'{max(p["quadrature"]["analytic_moment_relative_errors"]):.3g}' for p in windows) + '.\n'
text += 'These are quadrature scales for the declared singular elastic model,\nnot validated material length scales.\n'
text += 'The first lower-order attempt used radial order6 and failed its analytic\n'
text += 'first-depth/transverse-second-moment checks (6.63e-8 /2.94e-7,\n'
text += 'above1e-8) before evaluating stress. It remains recorded; the lower\n'
text += 'rule was rerun at order8 without changing the higher job or the gate.\n\n'
text += '| Integration comparison | Reference-norm change | Candidate-error-norm change |\n| --- | ---: | ---: |\n'
for key, label in [('local_rule','Local window'), ('polar_rule_on_complete_integral','Complete volume, local-rule refinement'),
                   ('remaining_global_partition','Remaining global partition')]:
    row = v[key]
    text += f'| {label} | {100*row["reference_norm_relative_change"]:.6g}% | {100*row["error_norm_relative_change"]:.6g}% |\n'
text += '\nPressure-difference norm changes under local/complete rule refinement are\n'
text += ' /'.join(f'{100*v["pressure_difference_norm_relative_changes"][key]:.5g}%' for key in ['local_polar_rule','complete_polar_rule']) + '. '
text += f'The finer corrected coarse-candidate volume error is{100*v["candidate_errors"]["coarse"]:.5g}%.\n'
text += f'The pressure-grid36/48 reference change is{100*v["reference_changes"]["reference36"]:.5g}%;\n'
text += f'the observed candidate-plus-reference variation sum is{100*v["observed_variation_sum"]:.5g}%.\n'
text += f'The joint conditional volume gate is **{"true" if v["conditional_targets_met"] else "false"}**. '
text += 'Observed changes are not continuum bounds.\n'
if not v['conditional_targets_met']:
    text += 'This result does not qualify the cheap law\'s complete-volume physical error.\n'
text += 'The ideal-edge model, all material contributions and original solver\nacceptance criteria remain unchanged.\n'
p = W / 'research/rigid_stress/docs/SHARP_EDGE_SHELL_CONTROL_20261023.md'
s = p.read_text(); anchor = '\n### Refined positive-depth peaks\n'
assert anchor in s and 'The cell-aligned pair uses' not in s
s = s.replace(anchor, text + anchor)
reproduction = '''The [integral-refinement packet](../evidence/20261019-qualified-ranking/sharp-edge-shell-integrals-v1/README.md)
retains all ten terminal campaigns, including both failed column variants,
the failed radial-order6 window and both cancelled pending inspectors.
The workers are study_edge_shell_exact_columns, study_edge_volume_window
and inspect_edge_shell_integrals. Use their captured task files with the
same CPU Slurm wrapper, explicit data output and private node caches.
The final volume inspector deliberately combines lower window-v3/group0
with higher window-v2/group1; these paths must not be replaced by the
failed window-v2/group0. No new pressure or elastic solve is introduced.

'''
s = s.replace('Next resolve the error-integral sensitivity', reproduction + 'Next resolve the error-integral sensitivity')
p.write_text(s)

rows = {row['mode']: row for row in batch['primary_rows']}
text = '\n| Batch | Pure rigid ms/step | Max ms/step | Full ms/step | Max/full env-step/s |\n| --- | ---: | ---: | ---: | ---: |\n'
for i, size in enumerate(batch['batches']):
    r, m, f = [rows[mode]['timings'][i] for mode in ['rigid','max','full']]
    text += f'| {size} | {1000*r["median_seconds"]:.3f} | {1000*m["median_seconds"]:.3f} | {1000*f["median_seconds"]:.3f} | {m["environment_steps_per_second"]:.2f} / {f["environment_steps_per_second"]:.2f} |\n'
text += '\nRelative to the same-allocation B1024 run, B2048 changes aggregate\nmax/full throughput by'
text += ' /'.join(f'{100*(rows[mode]["aggregate_rate_ratio"]-1):+.3f}%' for mode in ['max','full']) + '.\n'
text += 'All primary max/full modes admit90 times their batch size in timed\nenvironment steps; the separate audit covers130 times the batch.\n'
text += 'These three repeat blocks do not supply confidence intervals or a\nportable speedup. Each admitted mode retains all loaded/sharp counts.\n'
p = W / 'research/rigid_stress/docs/SHARP_EDGE_PRESSURE_20261022.md'
s = p.read_text(); anchor = '\n## B2048 throughput and sampled memory\n'
assert anchor in s and '| 2048 |' not in s.split(anchor)[1]
s = s.replace(anchor, anchor + text)
s += '''
The [B2048 packet](../evidence/20261019-qualified-ranking/hard-batch2048-v1/README.md)
retains terminal timing/memory job743187 and independent inspection743191,
captured workers, all timing repeats, complete public audit arrays and
memory polling receipts. Reproduce time_hard_franka and memory_hard_franka
from the captured task files, then inspect_hard_batch_scaling; bind fresh
output paths before using shell_comparison.submit. Scientific execution
uses Slurm rtx-mid, image1_26, the existing data-side physical environment
and private node caches. All55 engine heads and12 original unfinished
files remain preserved. Runtime data stay under the established data root.
'''
p.write_text(s)
print(json.dumps({'volume_conditional': v['conditional_targets_met'], 'batch_inspection': batch['passed']}))
