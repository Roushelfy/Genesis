"""Boundary-address diagnostics, not a relaxation of the engine array guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from genesis.utils.misc import qd_to_numpy


@qd.kernel(graph=True)
def scalar_test(values: qd.Tensor, coordinates: qd.types.ndarray(), result: qd.Tensor):
    for i in range(coordinates.shape[0]):
        values[coordinates[i, 0], coordinates[i, 1], coordinates[i, 2]] = 17.0 + i
    for i in range(coordinates.shape[0]):
        result[i] = values[coordinates[i, 0], coordinates[i, 1], coordinates[i, 2]]


@qd.kernel(graph=True)
def vector_test(values: qd.Tensor, coordinates: qd.types.ndarray(), result: qd.Tensor):
    for i in range(coordinates.shape[0]):
        values[coordinates[i, 0], coordinates[i, 1] // 3, coordinates[i, 2]][coordinates[i, 1] % 3] = 17.0 + i
    for i in range(coordinates.shape[0]):
        result[i] = values[coordinates[i, 0], coordinates[i, 1] // 3, coordinates[i, 2]][coordinates[i, 1] % 3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('scalar', 'vector'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    gs.init(backend=gs.gpu, precision='64', logging_level='warning')
    shape = (2328, 15, 65536)
    total = int(np.prod(shape))
    flat = [0, 1, 2**31 - 2, 2**31 - 1, 2**31, 2**31 + 1, total - 2, total - 1]
    # Canonical, environment-last addressing and physical batch-first layout.
    canonical = np.asarray([np.unravel_index(index, shape) for index in flat], dtype=np.int32)
    physical = np.asarray([(c, d, b) for b, c, d in (np.unravel_index(index, (65536, 2328, 15)) for index in flat)], dtype=np.int32)
    coordinates = np.unique(np.concatenate((canonical, physical)), axis=0)
    result = {
        'mode': args.mode, 'shape': shape, 'total_scalar_elements': total,
        'coordinates': coordinates.tolist(), 'stage': 'before_allocation', 'passed': False,
        'gs_qd_int': str(gs.qd_int), 'default_ip': str(qd.lang.impl.current_cfg().default_ip),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'gpu': torch.cuda.get_device_name(), 'job': os.environ.get('SLURM_JOB_ID'),
        'scope': 'Independent synthetic index-boundary diagnostic. No scene, contact, load model or production guard is changed.',
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    values = qd.tensor(dtype=qd.f64, shape=shape, layout=(2, 0, 1)) if args.mode == 'scalar' else qd.tensor(dtype=qd.types.vector(3, qd.f64), shape=(2328, 5, 65536), layout=(2, 0, 1))
    result['stage'] = 'allocated_before_access'
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    device_coordinates = torch.as_tensor(coordinates, dtype=torch.int32, device=gs.device)
    observed = qd.tensor(dtype=qd.f64, shape=(len(coordinates),))
    (scalar_test if args.mode == 'scalar' else vector_test)(values, device_coordinates, observed)
    qd.sync()
    actual = qd_to_numpy(observed)
    np.testing.assert_array_equal(actual, 17.0 + np.arange(len(coordinates)))
    result.update(stage='complete', passed=True, observed=actual.tolist())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
