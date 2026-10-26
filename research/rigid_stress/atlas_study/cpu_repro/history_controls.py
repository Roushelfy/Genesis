"""Check history state algebra and the cached gate's threshold safeguard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from .history import BatchHistory, RecoverySystem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("history_operator_controls.json"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(421)
        matrix = rng.normal(size=(30, 30))
        stiffness = sparse.csr_matrix(matrix.T @ matrix + np.eye(30))
        stress = sparse.csr_matrix(rng.normal(size=(60, 30)))
        scale = 1.0 / np.sqrt(stiffness.diagonal())
        diagonal = sparse.diags(scale)
        system = RecoverySystem(stiffness, stress, scale,
                                splu((diagonal @ stiffness @ diagonal).tocsc()), np.arange(30), np.ones(60))
        engines = (BatchHistory(system, 8, rotation=False), BatchHistory(system, 8),
                   BatchHistory(system, 8, cached_product=True))
        initial = rng.normal(size=(30, 8))
        velocity = 0.001 * rng.normal(size=(30, 8))
        maximum_difference = stress_consistency = 0.0
        decisions = 0
        for frame in range(320):
            displacement = initial + frame * velocity + 0.0001 * frame**2 * velocity
            forces = stiffness @ displacement
            active = np.ones(8, dtype=bool)
            if 5 <= frame < 8:
                active[2] = False
                forces[:, 2] = 0.0
            reset = np.array([3]) if frame in (13, 91, 200) else np.empty(0, dtype=np.int64)
            results = [engine.step(forces, active, reset=reset, audit_cache=True) for engine in engines]
            for result in results:
                maximum_difference = max(maximum_difference, float(np.max(np.abs(result.stress - results[0].stress))))
                stress_consistency = max(stress_consistency, float(np.max(np.abs(stress @ result.displacement - result.stress))))
                decisions += np.count_nonzero(result.accepted_prediction != results[0].accepted_prediction)
                decisions += result.cache_decision_disagreement
                assert np.max(np.abs(result.stress[:, ~active]), initial=0.0) < 1e-30
        assert maximum_difference < 1e-12 and stress_consistency < 1e-10 and decisions == 0

        direct = BatchHistory(system, 1)
        cached = BatchHistory(system, 1, cached_product=True)
        q = rng.normal(size=(30, 1))
        sigma, product = stress @ q, stiffness @ q
        for engine in (direct, cached):
            engine.depth[:] = 2
            engine.q_buffers[0][:] = q
            engine.q_buffers[1][:] = q
            engine.s_buffers[0][:] = sigma
            engine.s_buffers[1][:] = sigma
        cached.g_buffers[0][:] = product * (1.0 + 1e-12)
        cached.g_buffers[1][:] = product
        forces = product / (1.0 - direct.tolerance)
        direct_result = direct.step(forces, np.ones(1, dtype=bool))
        cached_result = cached.step(forces, np.ones(1, dtype=bool), audit_cache=True)
        assert cached_result.cache_direct_rechecks == 1
        assert np.array_equal(direct_result.accepted_prediction, cached_result.accepted_prediction)
        np.testing.assert_allclose(direct_result.stress, cached_result.stress, rtol=1e-13, atol=1e-13)
    result = {
        "scope": "Manufactured positive-definite operator/state control, separate from the physical shell trajectory",
        "frames": 320, "batch": 8, "copy_rotation_cache_absolute_difference": maximum_difference,
        "stress_displacement_absolute_difference": stress_consistency,
        "decision_disagreements": int(decisions), "near_threshold_direct_rechecks": cached_result.cache_direct_rechecks,
        "near_threshold_decision_equal": True, "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
