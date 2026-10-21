"""Collect paired input-only costs and cache diagnostics from terminal runs."""

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')


def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def describe(values):
    return {'count': len(values), 'sum': sum(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'p95': statistics.quantiles(values, n=100, method='inclusive')[94],
            'maximum': max(values)}


rows = []
cache_rows = []
for variant, campaign, batches in [
    ('published', 'hard-input-precomputed-v3', [4, 64]),
    ('published', 'hard-input-scaling-baseline-v1', [256, 1024]),
    ('cache_v1', 'hard-input-cache-v1', [4, 64]),
    ('cache_v1', 'hard-input-scaling-v1', [256, 1024]),
    ('cache_v2', 'hard-input-cache-v2', [4, 64, 256, 1024]),
    ('closest_bounds', 'hard-input-bounds-v1', [4, 64, 256, 1024]),
]:
    for i, batch in enumerate(batches):
        group = ROOT / campaign / f'group{i}'
        assert (group / 'returncode.txt').read_text().strip() == '0'
        result = json.loads((group / 'inputs/result.json').read_text())
        comparison = json.loads((group / 'comparison/result.json').read_text())
        assert comparison['motion_matches'] and result['execution_completed']
        assert all(value == 0 for row in result['grouping_errors'] for value in row)
        assert all(not row['research_audit_in_timing'] for row in result['rollouts'][1:])
        inputs = comparison['timing']['inputs']['median_scene_step_seconds']
        rigid = comparison['timing']['rigid']['median_scene_step_seconds']
        assert inputs == statistics.median(row['seconds'] / result['steps'] for row in result['rollouts'][1:])
        rows.append({
            'variant': variant, 'campaign': campaign, 'envs': batch,
            'input_seconds_per_step': inputs, 'rigid_seconds_per_step': rigid,
            'added_seconds_per_step': inputs - rigid,
            'raw_aggregate_env_steps_per_second': batch / inputs,
            'warm_samples': comparison['timing'],
            'loaded_environment_steps': sum(comparison['loaded_environment_steps']),
            'valid_loaded_environment_steps': sum(comparison['valid_loaded_environment_steps']),
            'source_status_counts': comparison['source_status_counts'],
            'phase_profile': result['phase_profile'],
            'curvature_build_seconds_including_first_compile': result['curvature_build_seconds_including_first_compile'],
            'inputs_result': identity(group / 'inputs/result.json'),
            'comparison_result': identity(group / 'comparison/result.json'),
        })
        if variant not in ('cache_v1', 'cache_v2'):
            continue
        kinds = Counter()
        hits = 0
        counts = [[], []]
        per_step = []
        history_path = group / 'inputs/history.jsonl.gz'
        with gzip.open(history_path, 'rt') as stream:
            for line in stream:
                observation = json.loads(line)
                kinds.update(observation['cache_hint_kind'])
                hits += sum(observation['cache_hit'])
                per_step.append(sum(sum(pair) for pair in observation['query_evaluations']))
                for pair in observation['query_evaluations']:
                    for j in range(2):
                        counts[j].append(pair[j])
        cache_rows.append({
            'variant': variant, 'envs': batch, 'hint_counts': dict(kinds),
            'temporal_hint_fraction': kinds[1] / sum(kinds.values()),
            'temporal_hint_hits': hits, 'temporal_hit_rate': hits / kinds[1],
            'closest_full_candidate_evaluations': describe(counts[0]),
            'normal_root_full_candidate_evaluations': describe(counts[1]),
            'total_candidate_evaluations_per_step': describe(per_step),
            'history': identity(history_path),
        })
for batch in (4, 64, 256, 1024):
    selected = [row for row in rows if row['envs'] == batch]
    for row in selected[1:]:
        for key in ('loaded_environment_steps', 'valid_loaded_environment_steps', 'source_status_counts'):
            assert row[key] == selected[0][key], (batch, row['variant'], key)
with (ROOT / 'meta/hard-input-scaling-summary-v1.json').open('x') as stream:
    json.dump({
        'passed': True, 'rows': rows, 'cache_rows': cache_rows,
        'scope': 'Matched warmed input-only/pure-rigid scene repeats on independently evolving batches. All environments contribute to raw aggregate throughput, including rejected geometry. No stress-update throughput or best possible batch/memory bound. Face-evaluation counts exclude bounding-box visits; even a temporal hint hit retains all potentially closer candidates. Hit rate is conditional on supplying a temporal hint, not the fraction of all contacts handled without global search. Actual moving-scene collision provenance supplied no usable vertex hint in these runs. Exclusive trace timings remain separate.',
        'whole_goal_complete': False,
    }, stream, indent=2)
    stream.write('\n')
