"""Check timing provenance, matching inputs and measured coverage with the standard library."""

import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent.parent
REPO = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery')
PACKET = REPO / 'research/rigid_stress/evidence/20261023-automatic-dpr-throughput-v1'


def main():
    rows = []
    for campaign in ('automatic-dpr-opt-matched-v3', 'automatic-dpr-opt-batches-v3'):
        directory = ROOT / campaign
        groups = json.loads((directory / 'tasks.json').read_text())
        for index, tasks in enumerate(groups):
            uuids, positions, conditions = set(), set(), set()
            for task in tasks:
                path = directory / f'group{index}' / task['name']
                assert path.with_name(path.name + '.returncode.txt').read_text().strip() == '0', path
                raw = json.loads(path.read_text())
                assert raw['source_revision'] == ('40775c1d0' if task['name'] == 'original-hertz-full' else 'db84114d1')
                assert raw['precision'] == '64' and raw['quadrature'] == 16
                assert raw['trajectory_metrics'] is None
                assert raw['dt'] == 0.01 and raw['substeps'] == 1
                assert raw['condition_count'] == (8 if raw['envs'] == 8 else 4)
                assert raw['conditions_sha256'] == (
                    'a9a9ecfa409f39acd159ee0e15205102710312ab76c5de539bd624f7676d807c' if raw['envs'] == 8
                    else 'ab451473db350152f812f095c64e8595a558a5ea07e4d884ba8b55d7680ff9a8')
                if raw['scope'] != 'rigid':
                    assert raw['dofs'] == 2430 and raw['tetrahedra'] == 480
                    assert raw['method_selected'] == 'inverse'
                    assert raw['stress_configurations'][0]['box_edge_policy'] == 'planar'
                    if raw['pressure_profile'] == 'hertz' and raw['source_revision'] == 'db84114d1':
                        assert raw['effective_contact_moment_reuse'] == [True]
                assert len(raw['repeats']) == (5 if campaign.endswith('matched-v3') and raw['envs'] != 16384 else 3)
                for repeat in raw['repeats']:
                    assert repeat['steps'] == (160 if raw['envs'] == 8 else 130)
                    assert repeat['invalid_environment_steps'] == 0
                    expected_fallbacks = 37 if campaign.endswith('matched-v3') and index == 3 and task['name'].startswith('manual-') else 0
                    assert repeat['scatter_overflow_calls'] == expected_fallbacks
                uuids.add(raw['gpu_uuid'])
                positions.add(hashlib.sha256(json.dumps(raw['egg_position_m']).encode()).hexdigest())
                conditions.add(raw['conditions_sha256'])
                samples = [1000 * repeat['seconds'] / repeat['steps'] for repeat in raw['repeats']]
                median = statistics.median(samples)
                rows.append({'campaign': campaign, 'group': index, 'task': task['name'], 'envs': raw['envs'],
                             'median_ms': median, 'range_ms': [min(samples), max(samples)],
                             'environment_steps_per_second': 1000 * raw['envs'] / median,
                             'native_stress_bytes': raw.get('native_buffer_bytes'),
                             'gpu_used_at_end_bytes': raw['gpu_total_bytes'] - raw['device_free_bytes'],
                             'scatter_fallback_calls': [repeat['scatter_overflow_calls'] for repeat in raw['repeats']],
                             'sampled_environment_steps': sum(raw['envs'] * repeat['steps'] for repeat in raw['repeats'])})
            assert len(uuids) == len(positions) == len(conditions) == 1, (campaign, index, uuids, positions, conditions)
    result = {'passed': True, 'production_source': 'db84114d1',
              'all_final_primary_repeats_zero_invalid': True,
              'same_gpu_and_exact_final_positions_within_each_group': True, 'rows': rows}
    path = ROOT / 'meta/automatic-dpr-opt-final-timing-validation-v2.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
