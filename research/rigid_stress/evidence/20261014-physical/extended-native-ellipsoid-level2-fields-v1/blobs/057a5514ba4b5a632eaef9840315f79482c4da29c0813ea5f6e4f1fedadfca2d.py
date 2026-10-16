"""Measure full rigid-contact tangents and stencil changes at rejected original-equation continuation states."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .original_franka_ground import build_original
from .run_original_rigid_franka import OriginalRigidEggContact
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.run, args.continuation]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) != 0
    config = json.loads((args.run / "results/configuration.json").read_text())
    assert hashlib.sha256(args.mjcf.read_bytes()).digest() == hashlib.sha256(Path(config["mjcf"]).read_bytes()).digest()
    static, frame, capture, protocol = build_original(
        Path(config["capture"]), Path(config["protocol_run"]), args.mjcf, config["dt"], config["dhat"]
    )
    system = OriginalRigidEggContact(static, frame, capture, args.mjcf, config["dt"])
    paths = sorted((args.run / "results").glob("step[0-9][0-9][0-9][0-9][0-9].npz"))
    with np.load(paths[-1]) as saved:
        previous, q, velocity, body_velocity = [
            saved[key].copy() for key in ["displacement", "joint_position", "joint_velocity", "body_velocity"]
        ]
    substeps = round(protocol["macro_dt_s"] / config["dt"])
    control = [row for row in protocol["records"] if row["mode"] == "approach_lift"][len(paths) // substeps]
    system.target = np.array(control["target"])
    system.caps[7:] = control["finger_cap_N"]
    system.begin_dynamic_step(q, velocity, previous[: system.n] + config["dt"] * body_velocity)
    fixed = {
        key: np.asarray(getattr(system, key)).copy() for key in ["previous_q", "previous_velocity", "target", "caps"]
    }
    args.output.mkdir(parents=True, exist_ok=False)
    numerical = json.loads((args.continuation / "results/numerical_recovery/summary.json").read_text())
    accepted = [(i + 1, row) for i, row in enumerate(numerical["attempts"]) if row["numerical_stage_accepted"]]
    targets = [0.125, 0.28, max(row["residual_continuation_parameter"] for _, row in accepted)]
    selected = sorted(
        {min(accepted, key=lambda item, t=t: abs(item[1]["residual_continuation_parameter"] - t))[0] for t in targets}
    )
    cases = [(args.run / "results/failure.npz", None)]
    cases.extend(
        (args.continuation / f"results/numerical_recovery/stage{index:04d}.npz", numerical["attempts"][index - 1])
        for index in selected
    )
    scales = np.r_[np.full(3, 1e-4), np.full(3, 1e-2), np.full(9, 0.1)]
    base_steps = np.r_[np.full(3, 1e-12), np.full(3, 1e-10), np.full(9, 1e-8)]
    mesh = system.contact.mesh

    def evaluate(x, hessian=False):
        wide = np.asarray(x, dtype=np.longdouble)
        result = system.evaluate_rigid(wide[:6], wide[6:], previous, hessian)
        collisions, _ = system.contact.normal_collisions(result[2])
        fingerprint = []
        for index in range(len(collisions)):
            collision = collisions[index]
            ids = np.array(collision.vertex_ids(mesh.edges, mesh.faces))
            fingerprint.append((type(collision).__name__, *ids.tolist()))
            del collision
        return np.asarray(np.r_[result[0], result[1]], dtype=float), tuple(sorted(fingerprint)), result[-1]

    records = []
    for index, (path, stage) in enumerate(cases):
        with np.load(path) as saved:
            x = saved["unknown"].copy() if stage is not None else np.r_[saved["body_pose"], saved["robot_variable"]]
            offset = saved["offset"].copy() if stage is not None else np.zeros(15)
        value, fingerprint, blocks = evaluate(x, True)
        bb, bq, qb, qq = blocks
        analytic = np.block([[bb, bq], [qb, qq]])
        scaled = analytic * scales
        observations = []
        arrays = {"unknown": x, "original_residual": value, "artificial_offset": offset, "analytic": analytic}
        for level, multiplier in enumerate([10.0, 1.0, 0.1, 0.01]):
            forward, backward, changes = [], [], []
            for column, step in enumerate(base_steps * multiplier):
                direction = np.eye(15)[column] * step
                plus, plus_stencil, _ = evaluate(x + direction)
                minus, minus_stencil, _ = evaluate(x - direction)
                forward.append((plus - value) / step)
                backward.append((value - minus) / step)
                if plus_stencil != fingerprint or minus_stencil != fingerprint:
                    changes.append({"column": column, "plus": plus_stencil, "minus": minus_stencil})
            forward, backward = np.column_stack(forward), np.column_stack(backward)
            central = (forward + backward) / 2
            norm = np.linalg.norm(scaled)
            observations.append(
                {
                    "absolute_step_multiplier": multiplier,
                    "central_scaled_relative_difference": float(np.linalg.norm((central - analytic) * scales) / norm),
                    "one_sided_scaled_relative_difference": float(np.linalg.norm((forward - backward) * scales) / norm),
                    "scaled_central_singular_values": np.linalg.svd(central * scales, compute_uv=False).tolist(),
                    "stencil_changes": changes,
                }
            )
            arrays[f"forward{level}"] = forward
            arrays[f"backward{level}"] = backward
        np.savez_compressed(args.output / f"case{index:02d}.npz", **arrays)
        records.append(
            {
                "source": str(path),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "numerical_stage": stage,
                "original_body_residual_N": float(np.linalg.norm(value[:6])),
                "original_robot_effort_residual": float(np.linalg.norm(value[6:])),
                "artificial_equation_residual": float(np.linalg.norm(value - offset)),
                "stencil_fingerprint": fingerprint,
                "scaled_analytic_singular_values": np.linalg.svd(scaled, compute_uv=False).tolist(),
                "observations": observations,
            }
        )
        print(json.dumps({"case": index, "source": str(path), "last": observations[-1]}), flush=True)
    for key, value in fixed.items():
        np.testing.assert_array_equal(getattr(system, key), value)
    report = {
        "scope": __doc__,
        "original_dt_s": system.dt,
        "original_previous_state_and_controls_fixed": True,
        "unknown_column_scales": scales.tolist(),
        "base_absolute_difference_steps": base_steps.tolist(),
        "cases": records,
        "physical_states_added": 0,
        "goal_completion": False,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
