"""Prepare larger-batch timing and separate fresh-process memory tasks."""
from pathlib import Path
import json

n = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
w = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees')
old = json.loads((n / 'hard-source-warps-timing-v1/tasks.json').read_text())[1]
for batch in [256, 1024]:
    name = f'hard-batch-scaling{batch}-v1'
    root = n / name / 'group0'
    args = list(old[0]['args'])
    args[args.index('--batches') + 1] = str(batch)
    memory_tasks = {}
    for mode in ['rigid', 'max', 'full']:
        task = {'module': 'research.rigid_stress.qipc_comparison.time_hard_franka',
                'args': args + ['--modes', mode, '--repeats', '1'], 'batch': batch, 'mode': mode}
        path = n / 'meta' / f'{name}-memory-{mode}-task.json'
        with path.open('x') as stream:
            json.dump(task, stream, indent=2); stream.write('\n')
        memory_tasks[mode] = {'name': 'memory-' + mode,
                              'module': 'research.rigid_stress.qipc_comparison.memory_hard_franka',
                              'args': ['--task', str(path)]}
    group = [memory_tasks['full']]
    for kind, source in [('baseline', 'hard_contact_edge_projection'), ('candidate', 'hard_contact_source_warps')]:
        group.append({'name': kind, 'module': 'research.rigid_stress.qipc_comparison.time_hard_franka',
                      'args': args, 'source': str(w / source)})
    group.append({'name': 'observations', 'module': 'research.rigid_stress.qipc_comparison.compare_hard_observations',
                  'args': ['--baseline', str(root / 'baseline'), '--candidate', str(root / 'candidate'),
                           '--batch', str(batch), '--check-repeated-counts']})
    group += [memory_tasks['rigid'], memory_tasks['max']]
    group.append({'name': 'memory-observations', 'module': 'research.rigid_stress.qipc_comparison.compare_hard_observations',
                  'args': ['--baseline', str(root / 'candidate'), '--candidate', str(root),
                           '--candidate-mode-subdir', 'memory-{mode}/driver', '--batch', str(batch),
                           '--check-repeated-counts']})
    with (n / 'meta' / f'{name}-tasks.json').open('x') as stream:
        json.dump([group], stream, indent=2); stream.write('\n')
    print(name, [task['name'] for task in group])
