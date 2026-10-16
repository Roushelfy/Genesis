"""Check graph capture of sequential offloaded passes inside native functions."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import quadrants as qd

from genesis.utils.misc import qd_to_numpy


@qd.func
def func_pass(values: qd.types.ndarray(), mode: qd.template()):
    for i in range(values.shape[0]):
        if qd.static(mode == 0):
            values[i] = i * 0.5
        else:
            values[i] += values[(i + 1) % values.shape[0]] * 0.0 + 1.0


@qd.kernel(graph=True)
def kernel_pass(values: qd.types.ndarray(), mode: qd.template()):
    func_pass(values, mode)


@qd.kernel(graph=True)
def kernel_fused(values: qd.types.ndarray()):
    func_pass(values, 0)
    func_pass(values, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qd.init(arch=qd.cuda, default_fp=qd.f64)
    values = qd.ndarray(dtype=qd.f64, shape=1024 * 1024)
    operations = (
        ("separate", lambda: (kernel_pass(values, 0), kernel_pass(values, 1))),
        ("combined", lambda: kernel_fused(values)),
    )
    result = {}
    for name, operation in operations:
        operation()
        np.testing.assert_array_equal(qd_to_numpy(values), np.arange(values.shape[0]) * 0.5 + 1)
        qd.sync()
        started = time.perf_counter()
        for _ in range(1000):
            operation()
        qd.sync()
        result[name] = 1e3 * (time.perf_counter() - started) / 1000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
