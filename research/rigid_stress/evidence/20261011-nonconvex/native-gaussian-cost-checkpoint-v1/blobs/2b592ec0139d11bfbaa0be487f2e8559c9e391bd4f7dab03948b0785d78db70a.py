"""Isolated Gaussian-profile precision cost at captured sample counts; no production selection or rollout claim."""

import argparse
import json
import time
from pathlib import Path

import genesis as gs
import numpy as np
import quadrants as qd
from genesis.utils.array_class import V
from genesis.utils.misc import qd_to_numpy


@qd.kernel
def initialize(relative: qd.types.ndarray(), weight: qd.types.ndarray()):
    for i_q in range(relative.shape[0]):
        relative[i_q] = (i_q % 1024 + 0.5) / 1024.0
        weight[i_q] = 0.25 + (i_q % 17) / 17.0


@qd.kernel
def gaussian(relative: qd.types.ndarray(), weight: qd.types.ndarray(), output: qd.types.ndarray(), low: qd.template()):
    for i_q in range(relative.shape[0]):
        distance = relative[i_q]
        argument = -0.5 * (distance / 0.45) ** 2
        gaussian_weight = gs.qd_float(0.0)
        if qd.static(low):
            gaussian_weight = qd.exp(qd.cast(argument, qd.f32))
        else:
            gaussian_weight = qd.exp(argument)
        output[i_q] = weight[i_q] * gaussian_weight * (1.0 - distance * distance) ** 2


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--capture", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
with np.load(args.capture) as capture:
    samples = capture["work"][:, :, 3].sum(axis=1)
    sample_mean, sample_maximum = int(np.ceil(samples.mean())), int(samples.max())
    actual_mean_step_ms = float(capture["step_seconds"].mean() * 1000)
gs.init(backend=gs.gpu, precision="64", logging_level="warning")
measurements = []
for count in dict.fromkeys((sample_mean, sample_maximum)):
    relative, weight, output = (V(dtype=gs.qd_float, shape=(max(1, count),)) for i in range(3))
    initialize(relative, weight)
    results = []
    for low in (False, True, False):
        for i in range(20):
            gaussian(relative, weight, output, low)
        qd.sync()
        repeat_seconds = []
        for i in range(5):
            qd.sync()
            start = time.perf_counter()
            for j in range(100):
                gaussian(relative, weight, output, low)
            qd.sync()
            repeat_seconds.append(time.perf_counter() - start)
        actual = qd_to_numpy(output, transpose=True)
        distance, area = qd_to_numpy(relative, transpose=True), qd_to_numpy(weight, transpose=True)
        expected = area * np.exp(-0.5 * (distance / 0.45) ** 2) * (1.0 - distance * distance) ** 2
        relative_error = float(np.max(abs(actual - expected) / expected))
        results.append({"gaussian_fp32": low, "repeat_seconds_per_100": repeat_seconds,
                        "mean_kernel_ms": float(np.mean(repeat_seconds) * 10),
                        "independent_profile_relative_error_max": relative_error})
    baseline = (results[0]["mean_kernel_ms"] + results[2]["mean_kernel_ms"]) * .5
    saving = baseline - results[1]["mean_kernel_ms"]
    measurements.append({"sample_count": count, "routes": results,
                         "isolated_saved_ms": saving, "captured_fenced_mean_step_fraction": saving / actual_mean_step_ms})
value = {"captured_sample_mean": sample_mean, "captured_sample_maximum": sample_maximum,
         "captured_fenced_mean_step_ms": actual_mean_step_ms, "measurements": measurements,
         "limitations": __doc__ + " Counts sum retained per-source candidate counts; aggregate diagnostic duplication can overestimate Gaussian calls. This interleaved off/on/off microkernel isolates contiguous64-bit input/output plus Gaussian computation and cannot establish whole-sampler register/occupancy or end-to-end gains, or final stress consistency. No contact/load/acceptance change is made."}
args.output.write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps(value, indent=2), flush=True)
