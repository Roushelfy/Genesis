"""Compare complete same-law rigid/deformable original protocols and separate resolved-load feedback from elastic recovery error."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np

from .audit_objective_frame import finite_frame
from .neohookean_reference import NeoHookeanBulk
from .original_franka_ground import build_original
from .reference_motion_audit import motion_moments
from .summarize_fine_references import closed_source


def source_wrenches(static, displacement, previous, saved_force, centre, dt):
    normal, positions = static.normal_collisions(displacement)
    tangential = ipctk.TangentialCollisions()
    tangential.build(static.mesh, positions, normal, static.barrier, static.friction)
    delta = static.mesh.vertices((displacement - previous).reshape(-1, 3))
    friction = ipctk.FrictionPotential(dt * 4e-6)
    ids = static.mesh.to_full_vertex_id()
    reconstructed = np.zeros_like(positions)
    records = {
        str(group): {
            "normal_force_on_body_N": np.zeros(3),
            "shear_force_on_body_N": np.zeros(3),
            "moment_on_body_about_current_com_Nm": np.zeros(3),
            "normal_stencils": 0,
        }
        for group in [1, 2, 3]
    }
    for kind, collisions in [("normal", normal), ("shear", tangential)]:
        for index in range(len(collisions)):
            collision = collisions[index]
            nodes = np.asarray(collision.vertex_ids(static.mesh.edges, static.mesh.faces))
            nodes = nodes[nodes >= 0]
            groups = static.groups[ids[nodes]]
            group = set(groups)
            assert group in ({0, 1}, {0, 2}, {0, 3})
            record = records[str(max(group))]
            if kind == "normal":
                centered = np.asarray(positions[nodes] - positions[nodes[0]], dtype=float)
                force = -static.barrier.gradient(collision, centered.reshape(-1)).reshape(-1, 3)
                record["normal_stencils"] += 1
            else:
                force = -friction.gradient(
                    collision, collision.dof(delta, static.mesh.edges, static.mesh.faces)
                ).reshape(-1, 3)
            np.add.at(reconstructed, nodes, force)
            body = groups == 0
            record[kind + "_force_on_body_N"] += force[body].sum(axis=0)
            record["moment_on_body_about_current_com_Nm"] += np.cross(positions[nodes[body]] - centre, force[body]).sum(
                axis=0
            )
            del collision
    reconstructed = static.mesh.to_full_dof(reconstructed.reshape(-1))
    np.testing.assert_allclose(reconstructed, saved_force, rtol=1e-12, atol=1e-12)
    return {
        key: {name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in row.items()}
        for key, row in records.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rigid", type=Path, required=True)
    parser.add_argument("--rigid-audit", type=Path, required=True)
    parser.add_argument("--deformable", type=Path, required=True)
    parser.add_argument("--deformable-audit", type=Path, required=True)
    parser.add_argument("--deformable-fields", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--allow-audited-prefix", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.rigid, args.rigid_audit, args.deformable, args.deformable_audit, args.deformable_fields]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    ra, da, fields = [
        json.loads((run / "results/summary.json").read_text())
        for run in [args.rigid_audit, args.deformable_audit, args.deformable_fields]
    ]
    assert Path(ra["run"]).resolve() == args.rigid.resolve()
    assert Path(da["run"]).resolve() == args.deformable.resolve()
    assert Path(fields["source_run"]).resolve() == args.deformable.resolve()
    count = ra["states_checked"]
    assert 0 < count == len(ra["rows"]) <= 2640
    complete_original = ra["complete_original_330_macro_history"] and count == 2640
    assert complete_original or args.allow_audited_prefix
    assert da["complete_requested_history"] and da["steps_checked"] == len(da["rows"]) == 2640
    assert fields["complete_original_330_macro_history"] and fields["states_checked"] == len(fields["rows"]) == 2640
    assert all(
        max(r["body_force_equivalent_residual_N"], r["fresh_actual_robot_effort_residual"]) <= 1e-9 for r in ra["rows"]
    )
    assert all(max(r["body_dynamic_residual_N"], r["actual_robot_step_effort_residual"]) <= 1e-9 for r in da["rows"])
    geometry = args.rigid / "results/reference_geometry.npz"
    assert geometry.read_bytes() == (args.deformable / "results/reference_geometry.npz").read_bytes()
    config = json.loads((args.rigid / "results/configuration.json").read_text())
    assert Path(config["matched_deformable"]).resolve() == args.deformable.resolve()
    assert hashlib.sha256(args.mjcf.read_bytes()).digest() == hashlib.sha256(Path(config["mjcf"]).read_bytes()).digest()
    static, _, _, protocol = build_original(
        Path(config["capture"]), Path(config["protocol_run"]), args.mjcf, config["dt"], config["dhat"]
    )
    body, dt = static.body, config["dt"]
    permutation, n = static.parts[0].permutation, body.basis.N
    mass = body.mass[permutation][:, permutation]
    bulk = NeoHookeanBulk(body)
    selected = [step for step in [11, 80, 866, 869, 1440, 2000, 2640] if step <= count]
    assert set(selected) <= set(fields["saved_snapshot_steps"])
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "body_geometry.npz",
        vertices=body.mesh.p.T,
        tetrahedra=body.mesh.t.T,
        component_permutation=permutation,
    )
    previous = {}
    for label, run in [("rigid", args.rigid), ("deformable", args.deformable)]:
        with np.load(run / "results/initial_state.npz") as saved:
            previous[label] = saved["full_displacement"].copy()
    rows, snapshots, inputs = [], [], {}

    def linear_quadrature(displacement):
        gradient = np.einsum("aie,ajeq->ijeq", displacement[bulk.dofs], bulk.gradients, optimize=True).transpose(
            2, 3, 0, 1
        )
        return bulk.mu * (gradient + gradient.swapaxes(-1, -2)) + bulk.lam * np.trace(gradient, axis1=-2, axis2=-1)[
            ..., None, None
        ] * np.eye(3)

    def error(first, second):
        weights = bulk.basis.dx
        assert first.shape == second.shape == (*weights.shape, 3, 3)
        assert np.all(weights > 0)
        absolute = float(np.sqrt(np.sum(weights[..., None, None] * (first - second) ** 2)))
        norm = float(np.sqrt(np.sum(weights[..., None, None] * second**2)))
        return {
            "volume_weighted_absolute_L2_Pa_m1p5": absolute,
            "reference_L2_Pa_m1p5": norm,
            "relative_L2": absolute / norm if norm > 0 else None,
        }

    for step in range(1, count + 1):
        frame = {}
        for label, run in [("rigid", args.rigid), ("deformable", args.deformable)]:
            path = run / f"results/step{step:05d}.npz"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if label == "rigid":
                assert ra["input_sha256"][str(path)] == digest
            else:
                assert fields["input_sha256"][str(path)] == digest
            inputs[str(path)] = digest
            with np.load(path) as saved:
                u, velocity, acceleration, force, applied, q, qv = [
                    saved[k].copy()
                    for k in [
                        "displacement",
                        "body_velocity",
                        "body_acceleration",
                        "contact_force_N",
                        "applied_force_N",
                        "joint_position",
                        "joint_velocity",
                    ]
                ]
            vector = np.empty(n)
            vector[permutation] = u[:n]
            rotation, translation, _local = finite_frame(body, vector)
            centre = rotation @ body.com + translation
            groups = source_wrenches(static, u, previous[label], force, centre, dt)
            frame[label] = {
                "centre_m": centre.tolist(),
                "rotation_world_from_material": rotation.tolist(),
                "source_wrenches_on_body": groups,
                "joint_position": np.asarray(q, dtype=float).tolist(),
                "joint_velocity": np.asarray(qv, dtype=float).tolist(),
            }
            if label == "rigid":
                material_contact, material_rhs = np.zeros(n), np.zeros(n)
                material_contact[body.component_dofs] = force[:n].reshape(-1, 3) @ rotation
                material_rhs[body.component_dofs] = (force[:n] + applied[:n] - mass @ acceleration).reshape(
                    -1, 3
                ) @ rotation
                motion = motion_moments(
                    body,
                    permutation,
                    np.asarray(static.rest[: n // 3] + u[:n].reshape(-1, 3), dtype=float),
                    np.asarray(velocity, dtype=float),
                )
                omega = rotation.T @ motion["angular_velocity_world_rad_s"]
                full_replay = body.solve(material_rhs)
                rigid_replay = body.solve(material_contact, omega)
                frame[label].update(
                    full_inertia_linear_peak_Pa=full_replay.peak_pa,
                    rigid_relief_linear_peak_Pa=rigid_replay.peak_pa,
                    full_rhs_balance_correction_N=float(np.linalg.norm(full_replay.rhs - material_rhs)),
                )
            previous[label] = u
        dfield = fields["rows"][step - 1]
        assert dfield["step"] == step
        row = {
            "step": step,
            "time_s": step * dt,
            **frame,
            "deformable_raw_global_peak_lower_Pa": dfield["raw_global_peak_lower_Pa"],
            "deformable_raw_global_peak_upper_Pa": dfield["raw_global_peak_upper_Pa"],
            "deformable_full_inertia_linear_peak_Pa": dfield["full_inertia_linear_peak_Pa"],
            "deformable_rigid_relief_linear_peak_Pa": dfield["rigid_relief_linear_peak_Pa"],
            "body_com_separation_m": float(
                np.linalg.norm(np.array(frame["rigid"]["centre_m"]) - frame["deformable"]["centre_m"])
            ),
            "joint_position_L2_difference": float(
                np.linalg.norm(np.array(frame["rigid"]["joint_position"]) - frame["deformable"]["joint_position"])
            ),
        }
        if step in selected:
            source = args.deformable_fields / f"results/step{step:05d}.npz"
            with np.load(source) as saved:
                deformable_full = linear_quadrature(saved["full_inertia_linear_displacement_m"])
                nonlinear = saved["objective_nonlinear_quadrature_tensor_Pa"].copy()
            rigid_full = linear_quadrature(full_replay.displacement)
            row["resolved_load_history_feedback_tensor"] = error(rigid_full, deformable_full)
            row["deformable_linear_recovery_vs_nonlinear_tensor"] = error(deformable_full, nonlinear)
            row["rigid_resolved_linear_vs_nonlinear_tensor"] = error(rigid_full, nonlinear)
            inputs[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
            np.savez_compressed(
                args.output / f"step{step:05d}.npz",
                material_contact_force_N=material_contact,
                material_full_inertia_rhs_N=material_rhs,
                omega_material_rad_s=omega,
                rigid_full_inertia_corner_tensor_Pa=full_replay.tensor_pa,
                rigid_full_inertia_quadrature_tensor_Pa=rigid_full,
                deformable_full_inertia_quadrature_tensor_Pa=deformable_full,
                deformable_nonlinear_quadrature_tensor_Pa=nonlinear,
            )
            snapshots.append(step)
        rows.append(row)
        if step % 80 == 0:
            print(json.dumps({"step": step, "com_separation_m": row["body_com_separation_m"]}), flush=True)
    assert inputs == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs}
    report = {
        "scope": __doc__,
        "rigid": str(args.rigid),
        "deformable": str(args.deformable),
        "states_checked": len(rows),
        "original_macro_controls": len([r for r in protocol["records"] if r["mode"] == "approach_lift"]),
        "complete_original_330_macro_history": complete_original,
        "requested_original_states": 2640,
        "audited_prefix_explicitly_allowed": args.allow_audited_prefix,
        "saved_snapshot_steps": snapshots,
        "rows": rows,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": f"All{count} audited matched original states share the same represented geometry, mass, material, full Panda model, ground, control sequence, timestep and IPC/friction law. Source wrenches are independently decomposed at actual body force points about each current body COM, including support and friction moments. Separate resolved-load linear replays isolate history/load feedback from deformable linear-versus-nonlinear recovery at{len(selected)} material-frame tensor snapshots; these error norms are not additive. Code0 and small algebra errors do not establish physical geometry/contact/time/regularization convergence, finite-footprint model accuracy or warmed accuracy/cost. The480-tet polyhedron's known geometry and concentration limitations remain. An explicitly allowed prefix is not complete original-history evidence.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
