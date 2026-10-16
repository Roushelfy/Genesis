"""Extract compact rigid-contact observations from qualified deformable Franka snapshots.

One packet per IPC stencil preserves its resultant, application point, free
moment and physical source. No new contact equilibrium is solved. The pullback
uses the proper fitted rigid frame; deformation remaining after that pullback
is retained in the application point and moment, rather than silently projected
onto a different reference surface.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_objective_frame import finite_frame
from .compare_original_contact_feedback import source_wrenches
from .original_franka_ground import build_original
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[11, 866, 869, 2640])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.run, args.fields, args.feedback]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def record(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    config = json.loads(record(args.run / "results/configuration.json").read_text())
    fields = json.loads(record(args.fields / "results/summary.json").read_text())
    feedback = json.loads(record(args.feedback / "results/summary.json").read_text())
    assert fields["complete_original_330_macro_history"] and fields["states_checked"] == 2640
    assert feedback["states_checked"] == 2640
    assert Path(fields["source_run"]).resolve() == args.run.resolve()
    assert Path(feedback["deformable"]).resolve() == args.run.resolve()
    assert set(args.steps) <= set(fields["saved_snapshot_steps"]) & set(feedback["saved_snapshot_steps"])
    mjcf = record(Path(config["mjcf"]))
    static, _, _, _ = build_original(
        Path(config["capture"]), Path(config["protocol_run"]), mjcf, config["dt"], config["dhat"]
    )
    body, permutation = static.body, static.parts[0].permutation
    n = body.basis.N
    mass = body.mass[permutation][:, permutation]
    np.savez_compressed(
        args.output / "body_mesh.npz",
        vertices=body.mesh.p.T,
        tetrahedra=body.mesh.t.T,
        surface_triangles=body.mesh.facets[:, body.mesh.boundary_facets()].T,
    )
    rows = []
    for step in args.steps:
        source = record(args.run / f"results/step{step:05d}.npz")
        with np.load(source) as saved:
            u = saved["displacement"].copy()
            force = saved["contact_force_N"].copy()
            acceleration = saved["body_acceleration"].copy()
            applied = saved["applied_force_N"].copy()
        previous_path = args.run / (f"results/step{step - 1:05d}.npz" if step > 1 else "results/initial_state.npz")
        with np.load(record(previous_path)) as saved:
            previous = saved["displacement" if step > 1 else "full_displacement"].copy()
        vector = np.empty(n)
        vector[permutation] = u[:n]
        rotation, translation, _ = finite_frame(body, vector)
        centre = rotation @ body.com + translation
        stencils = []
        groups = source_wrenches(static, u, previous, force, centre, config["dt"], stencils)
        old = feedback["rows"][step - 1]["deformable"]["source_wrenches_on_body"]
        assert groups == old, "The recording extension changed an already-qualified source wrench"
        normal = [r for r in stencils if r["kind"] == "normal"]
        shear = [r for r in stencils if r["kind"] == "shear"]
        assert len(normal) == len(shear) > 0
        packets = []
        reconstructed = np.zeros((n // 3, 3))
        for first, second in zip(normal, shear, strict=True):
            assert first["full_vertex_ids"] == second["full_vertex_ids"]
            assert first["body_vertex_ids"] == second["body_vertex_ids"]
            assert first["source"] == second["source"]
            positions = np.array(first["body_positions_m"])
            normal_force = np.array(first["body_forces_N"])
            tangent_force = np.array(second["body_forces_N"])
            np.testing.assert_array_equal(positions, second["body_positions_m"])
            weights = np.linalg.norm(normal_force, axis=1)
            assert weights.sum() > 0
            point = weights @ positions / weights.sum()
            resultant = (normal_force + tangent_force).sum(axis=0)
            moment = np.cross(positions - point, normal_force + tangent_force).sum(axis=0)
            inward = normal_force.sum(axis=0)
            assert np.linalg.norm(inward) > 0
            inward /= np.linalg.norm(inward)
            np.add.at(reconstructed, first["body_vertex_ids"], normal_force + tangent_force)
            # Coordinates and wrenches undergo the same proper rigid pullback.
            packets.append(
                {
                    "source": int(first["source"]),
                    "point_m": ((point - translation) @ rotation).tolist(),
                    "force_N": (resultant @ rotation).tolist(),
                    "free_moment_Nm": (moment @ rotation).tolist(),
                    "inward_normal": (inward @ rotation).tolist(),
                    "normal_force_N": (normal_force.sum(axis=0) @ rotation).tolist(),
                    "shear_force_N": (tangent_force.sum(axis=0) @ rotation).tolist(),
                    "friction": static.friction,
                }
            )
        np.testing.assert_allclose(reconstructed.reshape(-1), force[:n], rtol=1e-12, atol=1e-12)
        contact, rhs = np.zeros(n), np.zeros(n)
        contact[body.component_dofs] = force[:n].reshape(-1, 3) @ rotation
        rhs[body.component_dofs] = (force[:n] + applied[:n] - mass @ acceleration).reshape(-1, 3) @ rotation
        source_force = np.array([r["force_N"] for r in packets])
        source_points = np.array([r["point_m"] for r in packets])
        source_moments = np.array([r["free_moment_Nm"] for r in packets])
        np.testing.assert_allclose(
            source_force.sum(axis=0), contact[body.component_dofs].sum(axis=0), rtol=1e-12, atol=1e-12
        )
        moment = np.cross(source_points - body.com, source_force).sum(axis=0) + source_moments.sum(axis=0)
        current = (static.rest[: n // 3] + u[:n].reshape(-1, 3) - translation) @ rotation
        target = np.cross(current - body.com, force[:n].reshape(-1, 3) @ rotation).sum(axis=0)
        np.testing.assert_allclose(moment, target, rtol=1e-12, atol=1e-12)
        with np.load(record(args.fields / f"results/step{step:05d}.npz")) as saved:
            np.savez_compressed(
                args.output / f"step{step:05d}.npz",
                resolved_contact_N=contact,
                resolved_full_inertia_rhs_N=rhs,
                resolved_linear_displacement_m=saved["full_inertia_linear_displacement_m"],
                resolved_nonlinear_quadrature_tensor_Pa=saved["objective_nonlinear_quadrature_tensor_Pa"],
                rotation_world_from_material=rotation,
                translation_world_m=translation,
                current_material_positions_m=current,
            )
        (args.output / f"step{step:05d}-packets.json").write_text(json.dumps(packets, indent=2) + "\n")
        rows.append(
            {
                "step": step,
                "time_s": step * config["dt"],
                "packet_count": len(packets),
                "sources": groups,
                "source_wrenches_exactly_match_qualified_feedback": True,
                "current_minus_reference_wrench_Nm": (
                    target - np.cross(body.coordinates - body.com, contact[body.component_dofs]).sum(axis=0)
                ).tolist(),
            }
        )
        print(json.dumps({"step": step, "packets": len(packets)}), flush=True)
    assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "scope": __doc__,
                "run": str(args.run),
                "fields": str(args.fields),
                "feedback": str(args.feedback),
                "dt": config["dt"],
                "friction": static.friction,
                "rows": rows,
                "input_sha256": inputs,
                "goal_completion": False,
                "limitations": "Compact numerical-contact snapshots, not continuum or experimental truth. Finite rigid pullback preserves each actual stencil resultant and enabled free moment. Reference-versus-current material application geometry and fixed full-inertia RHS conventions must remain explicit in reconstructed replay. No footprint radius is inferred from a point wrench.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
